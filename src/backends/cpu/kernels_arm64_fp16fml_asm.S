// MLVC ARMv8.2-A (+fp16fml) 4x16 ping-pong double-buffered GEMM microkernel.
// Computes C[M x N] = A[M x K] * W_packed[K x N] + bias[N] + epilogue in FP32
// accumulators via FMLAL/FMLAL2 with ping-pong weight buffers (v0..v1 / v2..v3)
// hiding L1D load latency, plus zero-buffer in-register ReGLU fusion (v24..v31).

#if defined(__APPLE__)
#define MLVC_ASM_SYM(name) _##name
#else
#define MLVC_ASM_SYM(name) name
#endif

// struct Arm64AsmGemmArgs offsets (LP64, static_assert-verified in C++)
#define ARG_M_ROWS          0
#define ARG_N_COLS          8
#define ARG_K_DIM           16
#define ARG_LHS             24
#define ARG_LHS_STRIDE      32
#define ARG_RHS             40
#define ARG_DST             48
#define ARG_DST_STRIDE      56
#define ARG_RES             64
#define ARG_EPI_KIND        72
#define ARG_ALPHA           80
#define ARG_CLAMP_MIN       84
#define ARG_CLAMP_MAX       88
#define ARG_REGLU_LIN_OFF   96

#define EPI_NONE         0
#define EPI_CLAMP        1
#define EPI_LEAKY_RELU   2
#define EPI_RESIDUAL_ADD 3
#define EPI_REGLU        4

// Stack frame layout (256 bytes, 16-byte aligned):
//   [sp, #0  .. #31]  : sub-16 store scratch buffer
//   [sp, #32 .. #63]  : sub-16 residual load scratch buffer
//   [sp, #64]         : float alpha
//   [sp, #68]         : float clamp_min
//   [sp, #72]         : float clamp_max
//   [sp, #80]         : size_t reglu_linear_offset (bytes)
//   [sp, #96 .. #159] : d8 .. d15 callee-saved FP/SIMD
//   [sp, #160 .. #239]: x19 .. x28 callee-saved GPRs
//   [sp, #240 .. #255]: x29, x30 frame pointer & link register
#define FRAME_SIZE 256

.macro fmlal4_step wa, wb, lane
    fmlal   v8.4s,  \wa\().4h, v4.h[\lane]
    fmlal2  v9.4s,  \wa\().4h, v4.h[\lane]
    fmlal   v10.4s, \wb\().4h, v4.h[\lane]
    fmlal2  v11.4s, \wb\().4h, v4.h[\lane]

    fmlal   v12.4s, \wa\().4h, v5.h[\lane]
    fmlal2  v13.4s, \wa\().4h, v5.h[\lane]
    fmlal   v14.4s, \wb\().4h, v5.h[\lane]
    fmlal2  v15.4s, \wb\().4h, v5.h[\lane]

    fmlal   v16.4s, \wa\().4h, v6.h[\lane]
    fmlal2  v17.4s, \wa\().4h, v6.h[\lane]
    fmlal   v18.4s, \wb\().4h, v6.h[\lane]
    fmlal2  v19.4s, \wb\().4h, v6.h[\lane]

    fmlal   v20.4s, \wa\().4h, v7.h[\lane]
    fmlal2  v21.4s, \wa\().4h, v7.h[\lane]
    fmlal   v22.4s, \wb\().4h, v7.h[\lane]
    fmlal2  v23.4s, \wb\().4h, v7.h[\lane]
.endm

// Initialize x0..x3 row pointers clamped by mr (x7 = 1..4) from x16 & x23.
.macro setup_row_ptrs4
    mov     x0, x16
    add     x1, x0, x23
    add     x2, x1, x23
    add     x3, x2, x23
    cmp     x7, #4
    b.eq    1f
    cmp     x7, #2
    csel    x1, x1, x0, hs
    cmp     x7, #3
    csel    x2, x2, x0, hs
    mov     x3, x0
1:
.endm

// Compute v8..v23 = bias + LHS[4xK] * W[Kx16]. Inputs: x0..x3, x10 (wblk), x21 (K).
// Preserves v24..v31 and x6..x9, x15..x28 completely.
.macro run_4x16_k_loop id
    ldp     q0, q1, [x10], #32
    fcvtl   v8.4s,  v0.4h
    fcvtl2  v9.4s,  v0.8h
    fcvtl   v10.4s, v1.4h
    fcvtl2  v11.4s, v1.8h

    mov     v12.16b, v8.16b
    mov     v13.16b, v9.16b
    mov     v14.16b, v10.16b
    mov     v15.16b, v11.16b
    mov     v16.16b, v8.16b
    mov     v17.16b, v9.16b
    mov     v18.16b, v10.16b
    mov     v19.16b, v11.16b
    mov     v20.16b, v8.16b
    mov     v21.16b, v9.16b
    mov     v22.16b, v10.16b
    mov     v23.16b, v11.16b

    mov     x11, x21
    cmp     x11, #8
    b.lo    .Lk4_check_\id

    ldp     q0, q1, [x10], #32
.Lk8_loop_\id:
    prfm    pldl1keep, [x10, #256]
    ldr     q4, [x0], #16
    ldr     q5, [x1], #16
    ldr     q6, [x2], #16
    ldr     q7, [x3], #16

    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 0
    ldp     q0, q1, [x10], #32
    fmlal4_step v2, v3, 1
    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 2
    ldp     q0, q1, [x10], #32
    fmlal4_step v2, v3, 3
    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 4
    ldp     q0, q1, [x10], #32
    fmlal4_step v2, v3, 5
    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 6

    sub     x11, x11, #8
    cmp     x11, #8
    b.lo    .Lk8_last_\id
    ldp     q0, q1, [x10], #32
    fmlal4_step v2, v3, 7
    b       .Lk8_loop_\id

.Lk8_last_\id:
    fmlal4_step v2, v3, 7

.Lk4_check_\id:
    cmp     x11, #4
    b.lo    .Lk1_check_\id
    ldr     d4, [x0], #8
    ldr     d5, [x1], #8
    ldr     d6, [x2], #8
    ldr     d7, [x3], #8

    ldp     q0, q1, [x10], #32
    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 0
    ldp     q0, q1, [x10], #32
    fmlal4_step v2, v3, 1
    ldp     q2, q3, [x10], #32
    fmlal4_step v0, v1, 2
    fmlal4_step v2, v3, 3
    sub     x11, x11, #4

.Lk1_check_\id:
    cbz     x11, .Lk_done_\id
.Lk1_loop_\id:
    ldr     h4, [x0], #2
    ldr     h5, [x1], #2
    ldr     h6, [x2], #2
    ldr     h7, [x3], #2
    ldp     q0, q1, [x10], #32
    fmlal4_step v0, v1, 0
    subs    x11, x11, #1
    b.ne    .Lk1_loop_\id

.Lk_done_\id:
.endm

// Write q0, q1 (16 halfwords) to [x8] with sub-16 [sp] tail clamping (x15 = valid_n),
// advancing x8 += x26. Uses x12..x14, x17.
.macro store_q0q1_row id
    cmp     x15, #16
    b.lo    .Lstore_tail_\id
    stp     q0, q1, [x8]
    add     x8, x8, x26
    b       .Lrow_done_\id
.Lstore_tail_\id:
    stp     q0, q1, [sp]
    mov     x12, x8
    mov     x13, sp
    mov     x14, x15
.Lstore_copy_\id:
    ldrh    w17, [x13], #2
    strh    w17, [x12], #2
    subs    x14, x14, #1
    b.ne    .Lstore_copy_\id
    add     x8, x8, x26
.Lrow_done_\id:
.endm

// Narrow 4 FP32 vectors (r0..r3) -> FP16 (q0, q1), apply non-ReGLU epilogue, store.
.macro epilogue_and_store_row r0, r1, r2, r3, id
    fcvtn   v0.4h, \r0\().4s
    fcvtn2  v0.8h, \r1\().4s
    fcvtn   v1.4h, \r2\().4s
    fcvtn2  v1.8h, \r3\().4s

    cbz     x28, .Lstore_row_\id
    cmp     x28, #EPI_CLAMP
    b.eq    .Lepi_clamp_\id
    cmp     x28, #EPI_LEAKY_RELU
    b.eq    .Lepi_leaky_\id

    // EPI_RESIDUAL_ADD
    cmp     x15, #16
    b.lo    .Lres_tail_\id
    ldp     q2, q3, [x9]
    add     x9, x9, x26
    b       .Lres_add_\id
.Lres_tail_\id:
    stp     xzr, xzr, [sp, #32]
    stp     xzr, xzr, [sp, #48]
    mov     x12, x9
    add     x13, sp, #32
    mov     x14, x15
.Lres_copy_\id:
    ldrh    w17, [x12], #2
    strh    w17, [x13], #2
    subs    x14, x14, #1
    b.ne    .Lres_copy_\id
    add     x9, x9, x26
    ldp     q2, q3, [sp, #32]
.Lres_add_\id:
    fadd    v0.8h, v0.8h, v2.8h
    fadd    v1.8h, v1.8h, v3.8h
    b       .Lstore_row_\id

.Lepi_clamp_\id:
    fmin    v0.8h, v0.8h, v7.8h
    fmax    v0.8h, v0.8h, v6.8h
    fmin    v1.8h, v1.8h, v7.8h
    fmax    v1.8h, v1.8h, v6.8h
    b       .Lstore_row_\id

.Lepi_leaky_\id:
    fcmlt   v2.8h, v0.8h, #0.0
    fcmlt   v3.8h, v1.8h, #0.0
    fcvtl   v4.4s, v0.4h
    fcvtl2  v5.4s, v0.8h
    fmul    v4.4s, v4.4s, v6.4s
    fmul    v5.4s, v5.4s, v6.4s
    fcvtn   v4.4h, v4.4s
    fcvtn2  v4.8h, v5.4s
    bsl     v2.16b, v4.16b, v0.16b
    mov     v0.16b, v2.16b
    fcvtl   v4.4s, v1.4h
    fcvtl2  v5.4s, v1.8h
    fmul    v4.4s, v4.4s, v6.4s
    fmul    v5.4s, v5.4s, v6.4s
    fcvtn   v4.4h, v4.4s
    fcvtn2  v4.8h, v5.4s
    bsl     v3.16b, v4.16b, v1.16b
    mov     v1.16b, v3.16b

.Lstore_row_\id:
    store_q0q1_row \id
.endm

// Narrow linear FP32 accumulators (r0..r3) -> FP16 (q0, q1), multiply by pre-clamped
// gate FP16 pair (g0, g1) held in v24..v31, and store to [x8].
.macro reglu_mul_and_store_row r0, r1, r2, r3, g0, g1, id
    fcvtn   v0.4h, \r0\().4s
    fcvtn2  v0.8h, \r1\().4s
    fcvtn   v1.4h, \r2\().4s
    fcvtn2  v1.8h, \r3\().4s
    fmul    v0.8h, v0.8h, \g0\().8h
    fmul    v1.8h, v1.8h, \g1\().8h
    store_q0q1_row \id
.endm

    .text
    .align 4
    .globl MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_4x16)
#if !defined(__APPLE__)
    .type MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_4x16), %function
#endif
MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_4x16):
    hint    #25                     // paciasp + BTI c landing pad (NOP pre-ARMv8.3/v8.5)
    ldr     x1, [x0, #ARG_M_ROWS]
    cbz     x1, .Learly_exit
    ldr     x2, [x0, #ARG_N_COLS]
    cbz     x2, .Learly_exit

    sub     sp, sp, #FRAME_SIZE
    stp     x29, x30, [sp, #240]
    add     x29, sp, #240
    stp     x19, x20, [sp, #160]
    stp     x21, x22, [sp, #176]
    stp     x23, x24, [sp, #192]
    stp     x25, x26, [sp, #208]
    stp     x27, x28, [sp, #224]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    mov     x19, x1                 // m_rows
    mov     x20, x2                 // rem_n = n_cols (or gate_c for EPI_REGLU)
    ldr     x21, [x0, #ARG_K_DIM]
    ldr     x22, [x0, #ARG_LHS]
    ldr     x23, [x0, #ARG_LHS_STRIDE]
    ldr     x24, [x0, #ARG_RHS]
    ldr     x25, [x0, #ARG_DST]
    ldr     x26, [x0, #ARG_DST_STRIDE]
    ldr     x27, [x0, #ARG_RES]
    ldr     x28, [x0, #ARG_EPI_KIND]

    ldr     w3, [x0, #ARG_ALPHA]
    ldr     w4, [x0, #ARG_CLAMP_MIN]
    ldr     w5, [x0, #ARG_CLAMP_MAX]
    ldr     x12, [x0, #ARG_REGLU_LIN_OFF]
    str     w3, [sp, #64]
    str     w4, [sp, #68]
    str     w5, [sp, #72]
    str     x12, [sp, #80]

    cmp     x28, #EPI_RESIDUAL_ADD
    b.ne    .Lnb_loop
    cbnz    x27, .Lnb_loop
    mov     x28, #EPI_NONE

.Lnb_loop:
    mov     x15, #16
    cmp     x20, #16
    csel    x15, x20, x15, lo       // valid_n = min(16, rem_n)

    mov     x6, x19                 // rem_m = m_rows
    mov     x16, x22                // lhs_row_cur
    mov     x8, x25                 // dst_row_cur
    mov     x9, x27                 // res_row_cur

.Lmi_loop:
    mov     x7, #4
    cmp     x6, #4
    csel    x7, x6, x7, lo          // mr = min(4, rem_m)

    setup_row_ptrs4
    mov     x10, x24                // wblk (or wblk_gate for EPI_REGLU)

    cmp     x28, #EPI_REGLU
    b.eq    .Ldo_reglu

    // Standard non-ReGLU GEMM + epilogue
    run_4x16_k_loop norm

    cmp     x28, #EPI_CLAMP
    b.ne    .Lcheck_leaky_param
    ldr     s6, [sp, #68]           // clamp_min
    ldr     s7, [sp, #72]           // clamp_max
    fcvt    h6, s6
    fcvt    h7, s7
    dup     v6.8h, v6.h[0]
    dup     v7.8h, v7.h[0]
    b       .Lstore_rows

.Lcheck_leaky_param:
    cmp     x28, #EPI_LEAKY_RELU
    b.ne    .Lstore_rows
    ldr     s6, [sp, #64]           // alpha
    dup     v6.4s, v6.s[0]

.Lstore_rows:
    epilogue_and_store_row v8,  v9,  v10, v11, r0
    cmp     x7, #1
    b.ls    .Lmi_next
    epilogue_and_store_row v12, v13, v14, v15, r1
    cmp     x7, #2
    b.ls    .Lmi_next
    epilogue_and_store_row v16, v17, v18, v19, r2
    cmp     x7, #3
    b.ls    .Lmi_next
    epilogue_and_store_row v20, v21, v22, v23, r3
    b       .Lmi_next

.Ldo_reglu:
    // Pass 1: accumulate gate[4x16] in v8..v23, clamp to FP16 in v24..v31.
    run_4x16_k_loop reglu_gate

    ldr     s6, [sp, #68]           // clamp_min
    ldr     s7, [sp, #72]           // clamp_max
    fcvt    h6, s6
    fcvt    h7, s7
    dup     v6.8h, v6.h[0]
    dup     v7.8h, v7.h[0]

    fcvtn   v24.4h, v8.4s
    fcvtn2  v24.8h, v9.4s
    fcvtn   v25.4h, v10.4s
    fcvtn2  v25.8h, v11.4s
    fmin    v24.8h, v24.8h, v7.8h
    fmax    v24.8h, v24.8h, v6.8h
    fmin    v25.8h, v25.8h, v7.8h
    fmax    v25.8h, v25.8h, v6.8h

    fcvtn   v26.4h, v12.4s
    fcvtn2  v26.8h, v13.4s
    fcvtn   v27.4h, v14.4s
    fcvtn2  v27.8h, v15.4s
    fmin    v26.8h, v26.8h, v7.8h
    fmax    v26.8h, v26.8h, v6.8h
    fmin    v27.8h, v27.8h, v7.8h
    fmax    v27.8h, v27.8h, v6.8h

    fcvtn   v28.4h, v16.4s
    fcvtn2  v28.8h, v17.4s
    fcvtn   v29.4h, v18.4s
    fcvtn2  v29.8h, v19.4s
    fmin    v28.8h, v28.8h, v7.8h
    fmax    v28.8h, v28.8h, v6.8h
    fmin    v29.8h, v29.8h, v7.8h
    fmax    v29.8h, v29.8h, v6.8h

    fcvtn   v30.4h, v20.4s
    fcvtn2  v30.8h, v21.4s
    fcvtn   v31.4h, v22.4s
    fcvtn2  v31.8h, v23.4s
    fmin    v30.8h, v30.8h, v7.8h
    fmax    v30.8h, v30.8h, v6.8h
    fmin    v31.8h, v31.8h, v7.8h
    fmax    v31.8h, v31.8h, v6.8h

    // Pass 2: accumulate linear[4x16] into v8..v23 (v24..v31 untouched), multiply & store.
    setup_row_ptrs4
    ldr     x12, [sp, #80]
    add     x10, x24, x12
    run_4x16_k_loop reglu_lin

    reglu_mul_and_store_row v8,  v9,  v10, v11, v24, v25, gr0
    cmp     x7, #1
    b.ls    .Lmi_next
    reglu_mul_and_store_row v12, v13, v14, v15, v26, v27, gr1
    cmp     x7, #2
    b.ls    .Lmi_next
    reglu_mul_and_store_row v16, v17, v18, v19, v28, v29, gr2
    cmp     x7, #3
    b.ls    .Lmi_next
    reglu_mul_and_store_row v20, v21, v22, v23, v30, v31, gr3

.Lmi_next:
    cmp     x6, #4
    b.ls    .Lnb_next
    sub     x6, x6, #4
    add     x16, x16, x23, lsl #2   // + 4 * lhs_stride
    b       .Lmi_loop

.Lnb_next:
    add     x24, x24, x21, lsl #5
    add     x24, x24, #32           // wblk += 32 + k_dim * 32
    add     x25, x25, #32           // dst_col += 16 halfs
    add     x27, x27, #32           // res_col += 16 halfs
    subs    x20, x20, #16
    b.hi    .Lnb_loop

    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x19, x20, [sp, #160]
    ldp     x21, x22, [sp, #176]
    ldp     x23, x24, [sp, #192]
    ldp     x25, x26, [sp, #208]
    ldp     x27, x28, [sp, #224]
    ldp     x29, x30, [sp, #240]
    add     sp, sp, #FRAME_SIZE

.Learly_exit:
    hint    #29                     // autiasp (NOP pre-ARMv8.3)
    ret

#if !defined(__APPLE__)
    .size MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_4x16), .-MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_4x16)
#endif

#if defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__)
    .section .note.GNU-stack,"",%progbits
    .section .note.gnu.property,"a",%note
    .p2align 3, 0x0
    .long 4
    .long 16
    .long 5                         // NT_GNU_PROPERTY_TYPE_0
    .asciz "GNU"
    .p2align 3, 0x0
    .long 0xc0000000                // GNU_PROPERTY_AARCH64_FEATURE_1_AND
    .long 4
    .long 3                         // BTI (1) | PAC (2)
    .long 0
#endif
