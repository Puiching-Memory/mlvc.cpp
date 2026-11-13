// MLVC baseline ARMv8.0-A NEON 4x16 FP16->FP32 GEMM microkernel.
// Uses only mandatory ARMv8.0-A AArch64 NEON instructions (no +fp16/+fp16fml
// required) with ping-pong double-buffered weight loads/conversions and exact
// bounds checking for arbitrary M >= 1, N >= 1, and K >= 0.

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

#define FRAME_SIZE 240

.macro neon_fmla4_a lane
    fmla    v8.4s,  v0.4s, v24.s[\lane]
    fmla    v9.4s,  v1.4s, v24.s[\lane]
    fmla    v10.4s, v2.4s, v24.s[\lane]
    fmla    v11.4s, v3.4s, v24.s[\lane]
    fmla    v12.4s, v0.4s, v25.s[\lane]
    fmla    v13.4s, v1.4s, v25.s[\lane]
    fmla    v14.4s, v2.4s, v25.s[\lane]
    fmla    v15.4s, v3.4s, v25.s[\lane]
    fmla    v16.4s, v0.4s, v26.s[\lane]
    fmla    v17.4s, v1.4s, v26.s[\lane]
    fmla    v18.4s, v2.4s, v26.s[\lane]
    fmla    v19.4s, v3.4s, v26.s[\lane]
    fmla    v20.4s, v0.4s, v27.s[\lane]
    fmla    v21.4s, v1.4s, v27.s[\lane]
    fmla    v22.4s, v2.4s, v27.s[\lane]
    fmla    v23.4s, v3.4s, v27.s[\lane]
.endm

// Interleaved step consuming buffer A (v0..v3) while loading+converting next k into B (v4..v7)
.macro neon_step_a_prefetch_b lane
    ldp     q4, q6, [x10], #32
    fmla    v8.4s,  v0.4s, v24.s[\lane]
    fmla    v9.4s,  v1.4s, v24.s[\lane]
    fmla    v10.4s, v2.4s, v24.s[\lane]
    fmla    v11.4s, v3.4s, v24.s[\lane]
    fmla    v12.4s, v0.4s, v25.s[\lane]
    fmla    v13.4s, v1.4s, v25.s[\lane]
    fmla    v14.4s, v2.4s, v25.s[\lane]
    fmla    v15.4s, v3.4s, v25.s[\lane]
    fcvtl2  v5.4s,  v4.8h
    fcvtl   v4.4s,  v4.4h
    fcvtl2  v7.4s,  v6.8h
    fcvtl   v6.4s,  v6.4h
    fmla    v16.4s, v0.4s, v26.s[\lane]
    fmla    v17.4s, v1.4s, v26.s[\lane]
    fmla    v18.4s, v2.4s, v26.s[\lane]
    fmla    v19.4s, v3.4s, v26.s[\lane]
    fmla    v20.4s, v0.4s, v27.s[\lane]
    fmla    v21.4s, v1.4s, v27.s[\lane]
    fmla    v22.4s, v2.4s, v27.s[\lane]
    fmla    v23.4s, v3.4s, v27.s[\lane]
.endm

// Interleaved step consuming buffer B (v4..v7) while loading+converting next k into A (v0..v3)
.macro neon_step_b_prefetch_a lane
    ldp     q0, q2, [x10], #32
    fmla    v8.4s,  v4.4s, v24.s[\lane]
    fmla    v9.4s,  v5.4s, v24.s[\lane]
    fmla    v10.4s, v6.4s, v24.s[\lane]
    fmla    v11.4s, v7.4s, v24.s[\lane]
    fmla    v12.4s, v4.4s, v25.s[\lane]
    fmla    v13.4s, v5.4s, v25.s[\lane]
    fmla    v14.4s, v6.4s, v25.s[\lane]
    fmla    v15.4s, v7.4s, v25.s[\lane]
    fcvtl2  v1.4s,  v0.8h
    fcvtl   v0.4s,  v0.4h
    fcvtl2  v3.4s,  v2.8h
    fcvtl   v2.4s,  v2.4h
    fmla    v16.4s, v4.4s, v26.s[\lane]
    fmla    v17.4s, v5.4s, v26.s[\lane]
    fmla    v18.4s, v6.4s, v26.s[\lane]
    fmla    v19.4s, v7.4s, v26.s[\lane]
    fmla    v20.4s, v4.4s, v27.s[\lane]
    fmla    v21.4s, v5.4s, v27.s[\lane]
    fmla    v22.4s, v6.4s, v27.s[\lane]
    fmla    v23.4s, v7.4s, v27.s[\lane]
.endm

.macro neon_fmla4_b lane
    fmla    v8.4s,  v4.4s, v24.s[\lane]
    fmla    v9.4s,  v5.4s, v24.s[\lane]
    fmla    v10.4s, v6.4s, v24.s[\lane]
    fmla    v11.4s, v7.4s, v24.s[\lane]
    fmla    v12.4s, v4.4s, v25.s[\lane]
    fmla    v13.4s, v5.4s, v25.s[\lane]
    fmla    v14.4s, v6.4s, v25.s[\lane]
    fmla    v15.4s, v7.4s, v25.s[\lane]
    fmla    v16.4s, v4.4s, v26.s[\lane]
    fmla    v17.4s, v5.4s, v26.s[\lane]
    fmla    v18.4s, v6.4s, v26.s[\lane]
    fmla    v19.4s, v7.4s, v26.s[\lane]
    fmla    v20.4s, v4.4s, v27.s[\lane]
    fmla    v21.4s, v5.4s, v27.s[\lane]
    fmla    v22.4s, v6.4s, v27.s[\lane]
    fmla    v23.4s, v7.4s, v27.s[\lane]
.endm

.macro neon_epilogue_and_store_row r0, r1, r2, r3, id
    fcvtn   v0.4h, \r0\().4s
    fcvtn2  v0.8h, \r1\().4s
    fcvtn   v1.4h, \r2\().4s
    fcvtn2  v1.8h, \r3\().4s

    cbz     x28, .Lneon_store_row_\id

    fcvtl   v2.4s, v0.4h
    fcvtl2  v3.4s, v0.8h
    fcvtl   v4.4s, v1.4h
    fcvtl2  v5.4s, v1.8h

    cmp     x28, #EPI_CLAMP
    b.eq    .Lneon_epi_clamp_\id
    cmp     x28, #EPI_LEAKY_RELU
    b.eq    .Lneon_epi_leaky_\id

    // EPI_RESIDUAL_ADD
    cmp     x15, #16
    b.lo    .Lneon_res_tail_\id
    ldp     q28, q29, [x9]
    add     x9, x9, x26
    b       .Lneon_res_add_\id
.Lneon_res_tail_\id:
    stp     xzr, xzr, [sp, #32]
    stp     xzr, xzr, [sp, #48]
    mov     x12, x9
    add     x13, sp, #32
    mov     x14, x15
.Lneon_res_copy_\id:
    ldrh    w17, [x12], #2
    strh    w17, [x13], #2
    subs    x14, x14, #1
    b.ne    .Lneon_res_copy_\id
    add     x9, x9, x26
    ldp     q28, q29, [sp, #32]
.Lneon_res_add_\id:
    fcvtl   v6.4s,  v28.4h
    fcvtl2  v7.4s,  v28.8h
    fcvtl   v30.4s, v29.4h
    fcvtl2  v31.4s, v29.8h
    fadd    v2.4s, v2.4s, v6.4s
    fadd    v3.4s, v3.4s, v7.4s
    fadd    v4.4s, v4.4s, v30.4s
    fadd    v5.4s, v5.4s, v31.4s
    b       .Lneon_repack_\id

.Lneon_epi_clamp_\id:
    // v28.4s = clamp_min, v29.4s = clamp_max (preloaded)
    fmin    v2.4s, v2.4s, v29.4s
    fmax    v2.4s, v2.4s, v28.4s
    fmin    v3.4s, v3.4s, v29.4s
    fmax    v3.4s, v3.4s, v28.4s
    fmin    v4.4s, v4.4s, v29.4s
    fmax    v4.4s, v4.4s, v28.4s
    fmin    v5.4s, v5.4s, v29.4s
    fmax    v5.4s, v5.4s, v28.4s
    b       .Lneon_repack_\id

.Lneon_epi_leaky_\id:
    // v28.4s = alpha (preloaded)
    fcmlt   v6.4s, v2.4s, #0.0
    fmul    v7.4s, v2.4s, v28.4s
    bsl     v6.16b, v7.16b, v2.16b
    mov     v2.16b, v6.16b

    fcmlt   v6.4s, v3.4s, #0.0
    fmul    v7.4s, v3.4s, v28.4s
    bsl     v6.16b, v7.16b, v3.16b
    mov     v3.16b, v6.16b

    fcmlt   v6.4s, v4.4s, #0.0
    fmul    v7.4s, v4.4s, v28.4s
    bsl     v6.16b, v7.16b, v4.16b
    mov     v4.16b, v6.16b

    fcmlt   v6.4s, v5.4s, #0.0
    fmul    v7.4s, v5.4s, v28.4s
    bsl     v6.16b, v7.16b, v5.16b
    mov     v5.16b, v6.16b

.Lneon_repack_\id:
    fcvtn   v0.4h, v2.4s
    fcvtn2  v0.8h, v3.4s
    fcvtn   v1.4h, v4.4s
    fcvtn2  v1.8h, v5.4s

.Lneon_store_row_\id:
    cmp     x15, #16
    b.lo    .Lneon_store_tail_\id
    stp     q0, q1, [x8]
    add     x8, x8, x26
    b       .Lneon_row_done_\id
.Lneon_store_tail_\id:
    stp     q0, q1, [sp]
    mov     x12, x8
    mov     x13, sp
    mov     x14, x15
.Lneon_store_copy_\id:
    ldrh    w17, [x13], #2
    strh    w17, [x12], #2
    subs    x14, x14, #1
    b.ne    .Lneon_store_copy_\id
    add     x8, x8, x26
.Lneon_row_done_\id:
.endm

    .text
    .align 4
    .globl MLVC_ASM_SYM(mlvc_arm64_gemm_neon_fp32_4x16)
#if !defined(__APPLE__)
    .type MLVC_ASM_SYM(mlvc_arm64_gemm_neon_fp32_4x16), %function
#endif
MLVC_ASM_SYM(mlvc_arm64_gemm_neon_fp32_4x16):
    ldr     x1, [x0, #ARG_M_ROWS]
    cbz     x1, .Lneon_early_exit
    ldr     x2, [x0, #ARG_N_COLS]
    cbz     x2, .Lneon_early_exit

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

    mov     x19, x1
    mov     x20, x2
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
    b.ne    .Lneon_nb_loop
    cbnz    x27, .Lneon_nb_loop
    mov     x28, #EPI_NONE

.Lneon_nb_loop:
    mov     x15, #16
    cmp     x20, #16
    csel    x15, x20, x15, lo

    mov     x6, x19
    mov     x16, x22
    mov     x8, x25
    mov     x9, x27

.Lneon_mi_loop:
    mov     x7, #4
    cmp     x6, #4
    csel    x7, x6, x7, lo

    mov     x0, x16
    add     x1, x0, x23
    add     x2, x1, x23
    add     x3, x2, x23
    cmp     x7, #4
    b.eq    .Lneon_rows_ready
    cmp     x7, #2
    csel    x1, x1, x0, hs
    cmp     x7, #3
    csel    x2, x2, x0, hs
    mov     x3, x0

.Lneon_rows_ready:
    ldp     q0, q2, [x24]
    add     x10, x24, #32
    fcvtl   v8.4s,  v0.4h
    fcvtl2  v9.4s,  v0.8h
    fcvtl   v10.4s, v2.4h
    fcvtl2  v11.4s, v2.8h

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
    cmp     x11, #4
    b.lo    .Lneon_k1_check

.Lneon_k4_loop:
    prfm    pldl1keep, [x10, #256]
    ldr     d24, [x0], #8
    ldr     d25, [x1], #8
    ldr     d26, [x2], #8
    ldr     d27, [x3], #8
    ldp     q0, q2, [x10], #32
    fcvtl   v24.4s, v24.4h
    fcvtl   v25.4s, v25.4h
    fcvtl   v26.4s, v26.4h
    fcvtl   v27.4s, v27.4h
    fcvtl2  v1.4s,  v0.8h
    fcvtl   v0.4s,  v0.4h
    fcvtl2  v3.4s,  v2.8h
    fcvtl   v2.4s,  v2.4h

    neon_step_a_prefetch_b 0
    neon_step_b_prefetch_a 1
    neon_step_a_prefetch_b 2
    neon_fmla4_b 3

    sub     x11, x11, #4
    cmp     x11, #4
    b.hs    .Lneon_k4_loop

.Lneon_k1_check:
    cbz     x11, .Lneon_k_done
.Lneon_k1_loop:
    ldr     h24, [x0], #2
    ldr     h25, [x1], #2
    ldr     h26, [x2], #2
    ldr     h27, [x3], #2
    ldp     q0, q2, [x10], #32
    fcvtl   v24.4s, v24.4h
    fcvtl   v25.4s, v25.4h
    fcvtl   v26.4s, v26.4h
    fcvtl   v27.4s, v27.4h
    fcvtl2  v1.4s,  v0.8h
    fcvtl   v0.4s,  v0.4h
    fcvtl2  v3.4s,  v2.8h
    fcvtl   v2.4s,  v2.4h

    neon_fmla4_a 0
    subs    x11, x11, #1
    b.ne    .Lneon_k1_loop

.Lneon_k_done:
    cmp     x28, #EPI_CLAMP
    b.ne    .Lneon_check_leaky_param
    ldr     s28, [sp, #68]
    ldr     s29, [sp, #72]
    dup     v28.4s, v28.s[0]
    dup     v29.4s, v29.s[0]
    b       .Lneon_store_rows

.Lneon_check_leaky_param:
    cmp     x28, #EPI_LEAKY_RELU
    b.ne    .Lneon_store_rows
    ldr     s28, [sp, #64]
    dup     v28.4s, v28.s[0]

.Lneon_store_rows:
    neon_epilogue_and_store_row v8,  v9,  v10, v11, r0
    cmp     x7, #1
    b.ls    .Lneon_mi_next
    neon_epilogue_and_store_row v12, v13, v14, v15, r1
    cmp     x7, #2
    b.ls    .Lneon_mi_next
    neon_epilogue_and_store_row v16, v17, v18, v19, r2
    cmp     x7, #3
    b.ls    .Lneon_mi_next
    neon_epilogue_and_store_row v20, v21, v22, v23, r3

.Lneon_mi_next:
    cmp     x6, #4
    b.ls    .Lneon_nb_next
    sub     x6, x6, #4
    add     x16, x16, x23, lsl #2   // + 4 * lhs_stride
    b       .Lneon_mi_loop

.Lneon_nb_next:
    add     x24, x24, x21, lsl #5
    add     x24, x24, #32
    add     x25, x25, #32
    add     x27, x27, #32
    subs    x20, x20, #16
    b.hi    .Lneon_nb_loop

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

.Lneon_early_exit:
    ret

#if !defined(__APPLE__)
    .size MLVC_ASM_SYM(mlvc_arm64_gemm_neon_fp32_4x16), .-MLVC_ASM_SYM(mlvc_arm64_gemm_neon_fp32_4x16)
#endif

#if defined(__linux__) || defined(__ANDROID__) || defined(__OHOS__)
    .section .note.GNU-stack,"",%progbits
#endif
