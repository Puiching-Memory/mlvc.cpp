// MLVC ARMv8.2-A (+fp16fml) 6x16 FP16->FP32 GEMM microkernel.
// Computes C[M x N] = A[M x K] * W_packed[K x N] + bias[N] + epilogue,
// reading FP16 inputs/weights, accumulating in FP32 via FMLAL/FMLAL2 across
// all 8 K lanes per vector load, and storing FP16 outputs with exact bounds
// checking for arbitrary M >= 1, N >= 1, and K >= 0.

#if defined(__APPLE__)
#define MLVC_ASM_SYM(name) _##name
#else
#define MLVC_ASM_SYM(name) name
#endif

// struct Arm64AsmGemmArgs offsets (LP64, static_assert-verified in C++)
#define ARG_M_ROWS      0
#define ARG_N_COLS      8
#define ARG_K_DIM       16
#define ARG_LHS         24
#define ARG_LHS_STRIDE  32
#define ARG_RHS         40
#define ARG_DST         48
#define ARG_DST_STRIDE  56
#define ARG_RES         64
#define ARG_EPI_KIND    72
#define ARG_ALPHA       80
#define ARG_CLAMP_MIN   84
#define ARG_CLAMP_MAX   88

#define EPI_NONE         0
#define EPI_CLAMP        1
#define EPI_LEAKY_RELU   2
#define EPI_RESIDUAL_ADD 3

// Stack frame layout (240 bytes, 16-byte aligned):
//   [sp, #0  .. #31]  : sub-16 store scratch buffer
//   [sp, #32 .. #63]  : sub-16 residual load scratch buffer
//   [sp, #64]         : float alpha
//   [sp, #68]         : float clamp_min
//   [sp, #72]         : float clamp_max
//   [sp, #80 .. #143] : d8 .. d15 callee-saved FP/SIMD
//   [sp, #144 .. #223]: x19 .. x28 callee-saved GPRs
//   [sp, #224 .. #239]: x29, x30 frame pointer & link register
#define FRAME_SIZE 240

.macro fmlal_step lane
    ldp     q0, q1, [x10], #32
    fmlal   v8.4s,  v0.4h, v2.h[\lane]
    fmlal2  v9.4s,  v0.4h, v2.h[\lane]
    fmlal   v10.4s, v1.4h, v2.h[\lane]
    fmlal2  v11.4s, v1.4h, v2.h[\lane]

    fmlal   v12.4s, v0.4h, v3.h[\lane]
    fmlal2  v13.4s, v0.4h, v3.h[\lane]
    fmlal   v14.4s, v1.4h, v3.h[\lane]
    fmlal2  v15.4s, v1.4h, v3.h[\lane]

    fmlal   v16.4s, v0.4h, v4.h[\lane]
    fmlal2  v17.4s, v0.4h, v4.h[\lane]
    fmlal   v18.4s, v1.4h, v4.h[\lane]
    fmlal2  v19.4s, v1.4h, v4.h[\lane]

    fmlal   v20.4s, v0.4h, v5.h[\lane]
    fmlal2  v21.4s, v0.4h, v5.h[\lane]
    fmlal   v22.4s, v1.4h, v5.h[\lane]
    fmlal2  v23.4s, v1.4h, v5.h[\lane]

    fmlal   v24.4s, v0.4h, v6.h[\lane]
    fmlal2  v25.4s, v0.4h, v6.h[\lane]
    fmlal   v26.4s, v1.4h, v6.h[\lane]
    fmlal2  v27.4s, v1.4h, v6.h[\lane]

    fmlal   v28.4s, v0.4h, v7.h[\lane]
    fmlal2  v29.4s, v0.4h, v7.h[\lane]
    fmlal   v30.4s, v1.4h, v7.h[\lane]
    fmlal2  v31.4s, v1.4h, v7.h[\lane]
.endm

// Narrow 4 FP32 vectors (r0..r3) to 2 FP16 vectors (q0, q1), apply epilogue,
// and write x15 (1..16) halfwords to [x8], advancing x8 += x26 (and x9 += x26
// when residual add is active). Uses v0..v7 and x12..x14, x17.
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

    // EPI_RESIDUAL_ADD (x27 non-null guaranteed when x28 == EPI_RESIDUAL_ADD)
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
    // v6.8h = clamp_min, v7.8h = clamp_max (preloaded before row stores)
    fmin    v0.8h, v0.8h, v7.8h
    fmax    v0.8h, v0.8h, v6.8h
    fmin    v1.8h, v1.8h, v7.8h
    fmax    v1.8h, v1.8h, v6.8h
    b       .Lstore_row_\id

.Lepi_leaky_\id:
    // v6.4s = alpha (preloaded before row stores)
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

    .text
    .align 4
    .globl MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_6x16)
#if !defined(__APPLE__)
    .type MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_6x16), %function
#endif
MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_6x16):
    ldr     x1, [x0, #ARG_M_ROWS]
    cbz     x1, .Learly_exit
    ldr     x2, [x0, #ARG_N_COLS]
    cbz     x2, .Learly_exit

    sub     sp, sp, #FRAME_SIZE
    stp     x29, x30, [sp, #224]
    add     x29, sp, #224
    stp     x19, x20, [sp, #144]
    stp     x21, x22, [sp, #160]
    stp     x23, x24, [sp, #176]
    stp     x25, x26, [sp, #192]
    stp     x27, x28, [sp, #208]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]
    stp     d14, d15, [sp, #128]

    mov     x19, x1                 // m_rows
    mov     x20, x2                 // rem_n = n_cols
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
    str     w3, [sp, #64]
    str     w4, [sp, #68]
    str     w5, [sp, #72]

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
    mov     x7, #6
    cmp     x6, #6
    csel    x7, x6, x7, lo          // mr = min(6, rem_m)

    mov     x0, x16
    add     x1, x0, x23
    add     x2, x1, x23
    add     x3, x2, x23
    add     x4, x3, x23
    add     x5, x4, x23
    cmp     x7, #6
    b.eq    .Lrows_ready
    cmp     x7, #2
    csel    x1, x1, x0, hs
    cmp     x7, #3
    csel    x2, x2, x0, hs
    cmp     x7, #4
    csel    x3, x3, x0, hs
    cmp     x7, #5
    csel    x4, x4, x0, hs
    mov     x5, x0

.Lrows_ready:
    // Load FP16 bias[0..15] -> FP32 across all 6 rows' accumulators (v8..v31)
    ldp     q0, q1, [x24]
    add     x10, x24, #32
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
    mov     v24.16b, v8.16b
    mov     v25.16b, v9.16b
    mov     v26.16b, v10.16b
    mov     v27.16b, v11.16b
    mov     v28.16b, v8.16b
    mov     v29.16b, v9.16b
    mov     v30.16b, v10.16b
    mov     v31.16b, v11.16b

    mov     x11, x21
    cmp     x11, #8
    b.lo    .Lk4_check

.Lk8_loop:
    prfm    pldl1keep, [x10, #256]
    ldr     q2, [x0], #16
    ldr     q3, [x1], #16
    ldr     q4, [x2], #16
    ldr     q5, [x3], #16
    ldr     q6, [x4], #16
    ldr     q7, [x5], #16

    fmlal_step 0
    fmlal_step 1
    fmlal_step 2
    fmlal_step 3
    fmlal_step 4
    fmlal_step 5
    fmlal_step 6
    fmlal_step 7

    sub     x11, x11, #8
    cmp     x11, #8
    b.hs    .Lk8_loop

.Lk4_check:
    cmp     x11, #4
    b.lo    .Lk1_check
    ldr     d2, [x0], #8
    ldr     d3, [x1], #8
    ldr     d4, [x2], #8
    ldr     d5, [x3], #8
    ldr     d6, [x4], #8
    ldr     d7, [x5], #8

    fmlal_step 0
    fmlal_step 1
    fmlal_step 2
    fmlal_step 3
    sub     x11, x11, #4

.Lk1_check:
    cbz     x11, .Lk_done
.Lk1_loop:
    ldr     h2, [x0], #2
    ldr     h3, [x1], #2
    ldr     h4, [x2], #2
    ldr     h5, [x3], #2
    ldr     h6, [x4], #2
    ldr     h7, [x5], #2

    fmlal_step 0
    subs    x11, x11, #1
    b.ne    .Lk1_loop

.Lk_done:
    // Preload epilogue parameters into v6/v7 before per-row narrows/stores
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
    cmp     x7, #4
    b.ls    .Lmi_next
    epilogue_and_store_row v24, v25, v26, v27, r4
    cmp     x7, #5
    b.ls    .Lmi_next
    epilogue_and_store_row v28, v29, v30, v31, r5

.Lmi_next:
    cmp     x6, #6
    b.ls    .Lnb_next
    sub     x6, x6, #6
    add     x12, x23, x23, lsl #1   // 3 * lhs_stride
    add     x16, x16, x12, lsl #1   // + 6 * lhs_stride
    b       .Lmi_loop

.Lnb_next:
    add     x24, x24, x21, lsl #5
    add     x24, x24, #32           // wblk += 32 + k_dim * 32
    add     x25, x25, #32           // dst_col += 16 halfs
    add     x27, x27, #32           // res_col += 16 halfs
    subs    x20, x20, #16
    b.hi    .Lnb_loop

    ldp     d8,  d9,  [sp, #80]
    ldp     d10, d11, [sp, #96]
    ldp     d12, d13, [sp, #112]
    ldp     d14, d15, [sp, #128]
    ldp     x19, x20, [sp, #144]
    ldp     x21, x22, [sp, #160]
    ldp     x23, x24, [sp, #176]
    ldp     x25, x26, [sp, #192]
    ldp     x27, x28, [sp, #208]
    ldp     x29, x30, [sp, #224]
    add     sp, sp, #FRAME_SIZE

.Learly_exit:
    ret

#if !defined(__APPLE__)
    .size MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_6x16), .-MLVC_ASM_SYM(mlvc_arm64_gemm_fp16fml_6x16)
#endif

#if defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__)
    .section .note.GNU-stack,"",%progbits
#endif
