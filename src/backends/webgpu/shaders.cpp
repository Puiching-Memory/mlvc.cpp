#include "shaders.hpp"

#include <string_view>

namespace mlvc::webgpu_backend {

std::string wgsl_unary(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    count: u32,
    op_kind: u32, // 0=leaky_relu, 1=clip, 2=sigmoid, 3=reciprocal, 4=round
    pad0: u32,
    pad1: u32,
    alpha: f32,
    clamp_min: f32,
    clamp_max: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    let x = f32(src[i]);
    var y: f32 = x;
    if (p.op_kind == 0u) {
        y = select(x, x * p.alpha, x < 0.0);
    } else if (p.op_kind == 1u) {
        y = clamp(x, p.clamp_min, p.clamp_max);
    } else if (p.op_kind == 2u) {
        y = 1.0 / (1.0 + exp(-x));
    } else if (p.op_kind == 3u) {
        y = 1.0 / x;
    } else {
        y = round(x);
    }
    dst[i] = f16(y);
}
)";
    }
    return R"(struct Params {
    count: u32,
    op_kind: u32,
    pad0: u32,
    pad1: u32,
    alpha: f32,
    clamp_min: f32,
    clamp_max: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn apply_op(x: f32) -> f32 {
    if (p.op_kind == 0u) { return select(x, x * p.alpha, x < 0.0); }
    if (p.op_kind == 1u) { return clamp(x, p.clamp_min, p.clamp_max); }
    if (p.op_kind == 2u) { return 1.0 / (1.0 + exp(-x)); }
    if (p.op_kind == 3u) { return 1.0 / x; }
    return round(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if (pair_idx * 2u >= p.count) { return; }
    let v = unpack2x16float(src[pair_idx]);
    dst[pair_idx] = pack2x16float(vec2<f32>(apply_op(v.x), apply_op(v.y)));
}
)";
}

std::string wgsl_unary_vec4()
{
    // 8-byte aligned vec4<f16> unary kernel: eliminates D3D12 16-bit RMW contention!
    return R"(enable f16;
struct Params {
    count4: u32,
    op_kind: u32, // 0=leaky_relu, 1=clip, 2=sigmoid, 3=reciprocal, 4=round
    pad0: u32,
    pad1: u32,
    alpha: f32,
    clamp_min: f32,
    clamp_max: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count4) { return; }
    let x = vec4<f32>(src[i]);
    var y = x;
    if (p.op_kind == 0u) {
        y = select(x, x * p.alpha, x < vec4<f32>(0.0));
    } else if (p.op_kind == 1u) {
        y = clamp(x, vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max));
    } else if (p.op_kind == 2u) {
        y = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-x));
    } else if (p.op_kind == 3u) {
        y = vec4<f32>(1.0) / x;
    } else {
        y = round(x);
    }
    dst[i] = vec4<f16>(y);
}
)";
}

std::string wgsl_binary(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    total: u32,
    op_kind: u32, // 0=add, 1=sub, 2=mul
    mode: u32,    // 0=contig, 1=rhs_scalar, 2=lhs_scalar, 3=rhs_chan, 4=lhs_chan, 5=general4d
    channels: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    an: u32, ac: u32, ah: u32, aw: u32,
    bn: u32, bc: u32, bh: u32, bw: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> lhs: array<f16>;
@group(0) @binding(2) var<storage, read> rhs: array<f16>;
@group(0) @binding(3) var<storage, read_write> dst: array<f16>;

fn eval_bin(a: f32, b: f32) -> f16 {
    if (p.op_kind == 0u) { return f16(a + b); }
    if (p.op_kind == 1u) { return f16(a - b); }
    return f16(a * b);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.total) { return; }
    if (p.mode == 0u) {
        dst[i] = eval_bin(f32(lhs[i]), f32(rhs[i]));
        return;
    }
    if (p.mode == 1u) {
        dst[i] = eval_bin(f32(lhs[i]), f32(rhs[0]));
        return;
    }
    if (p.mode == 2u) {
        dst[i] = eval_bin(f32(lhs[0]), f32(rhs[i]));
        return;
    }
    if (p.mode == 3u) {
        let c = i % p.channels;
        dst[i] = eval_bin(f32(lhs[i]), f32(rhs[c]));
        return;
    }
    if (p.mode == 4u) {
        let c = i % p.channels;
        dst[i] = eval_bin(f32(lhs[c]), f32(rhs[i]));
        return;
    }
    // General 4D NHWC broadcast.
    let c = i % p.oc;
    var tmp = i / p.oc;
    let x = tmp % p.ow;
    tmp = tmp / p.ow;
    let y = tmp % p.oh;
    let b = tmp / p.oh;

    let ab = select(b, 0u, p.an == 1u);
    let ay = select(y, 0u, p.ah == 1u);
    let ax = select(x, 0u, p.aw == 1u);
    let ac = select(c, 0u, p.ac == 1u);
    let a_idx = (((ab * p.ah + ay) * p.aw + ax) * p.ac) + ac;

    let bb = select(b, 0u, p.bn == 1u);
    let by = select(y, 0u, p.bh == 1u);
    let bx = select(x, 0u, p.bw == 1u);
    let bc = select(c, 0u, p.bc == 1u);
    let b_idx = (((bb * p.bh + by) * p.bw + bx) * p.bc) + bc;

    dst[i] = eval_bin(f32(lhs[a_idx]), f32(rhs[b_idx]));
}
)";
    }
    return R"(struct Params {
    total: u32,
    op_kind: u32,
    mode: u32,
    channels: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    an: u32, ac: u32, ah: u32, aw: u32,
    bn: u32, bc: u32, bh: u32, bw: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> lhs: array<u32>;
@group(0) @binding(2) var<storage, read> rhs: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

fn rd_lhs(idx: u32) -> f32 {
    let pair = unpack2x16float(lhs[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn rd_rhs(idx: u32) -> f32 {
    let pair = unpack2x16float(rhs[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn eval_bin(a: f32, b: f32) -> f32 {
    if (p.op_kind == 0u) { return a + b; }
    if (p.op_kind == 1u) { return a - b; }
    return a * b;
}
fn compute_one(i: u32) -> f32 {
    if (i >= p.total) { return 0.0; }
    if (p.mode == 0u) { return eval_bin(rd_lhs(i), rd_rhs(i)); }
    if (p.mode == 1u) { return eval_bin(rd_lhs(i), rd_rhs(0u)); }
    if (p.mode == 2u) { return eval_bin(rd_lhs(0u), rd_rhs(i)); }
    if (p.mode == 3u) { return eval_bin(rd_lhs(i), rd_rhs(i % p.channels)); }
    if (p.mode == 4u) { return eval_bin(rd_lhs(i % p.channels), rd_rhs(i)); }
    let c = i % p.oc;
    var tmp = i / p.oc;
    let x = tmp % p.ow;
    tmp = tmp / p.ow;
    let y = tmp % p.oh;
    let b = tmp / p.oh;
    let a_idx = (((select(b, 0u, p.an == 1u) * p.ah + select(y, 0u, p.ah == 1u)) * p.aw + select(x, 0u, p.aw == 1u)) * p.ac) + select(c, 0u, p.ac == 1u);
    let b_idx = (((select(b, 0u, p.bn == 1u) * p.bh + select(y, 0u, p.bh == 1u)) * p.bw + select(x, 0u, p.bw == 1u)) * p.bc) + select(c, 0u, p.bc == 1u);
    return eval_bin(rd_lhs(a_idx), rd_rhs(b_idx));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let i0 = pair_idx * 2u;
    if (i0 >= p.total) { return; }
    dst[pair_idx] = pack2x16float(vec2<f32>(compute_one(i0), compute_one(i0 + 1u)));
}
)";
}

std::string wgsl_binary_vec4()
{
    // 8-byte aligned vec4<f16> binary kernel (modes 0=contig, 1=rhs_scalar,
    // 2=lhs_scalar, 3=rhs_chan4, 4=lhs_chan4): eliminates D3D12 16-bit RMW contention!
    return R"(enable f16;
struct Params {
    total4: u32,
    op_kind: u32, // 0=add, 1=sub, 2=mul
    mode: u32,
    channels4: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> rhs: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f16>>;

fn eval_v4(a: vec4<f32>, b: vec4<f32>) -> vec4<f16> {
    if (p.op_kind == 0u) { return vec4<f16>(a + b); }
    if (p.op_kind == 1u) { return vec4<f16>(a - b); }
    return vec4<f16>(a * b);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.total4) { return; }
    if (p.mode == 0u) {
        dst[i] = eval_v4(vec4<f32>(lhs[i]), vec4<f32>(rhs[i]));
        return;
    }
    if (p.mode == 1u) {
        let s = f32(rhs[0].x);
        dst[i] = eval_v4(vec4<f32>(lhs[i]), vec4<f32>(s));
        return;
    }
    if (p.mode == 2u) {
        let s = f32(lhs[0].x);
        dst[i] = eval_v4(vec4<f32>(s), vec4<f32>(rhs[i]));
        return;
    }
    let c4 = i % p.channels4;
    if (p.mode == 3u) {
        dst[i] = eval_v4(vec4<f32>(lhs[i]), vec4<f32>(rhs[c4]));
    } else {
        dst[i] = eval_v4(vec4<f32>(lhs[c4]), vec4<f32>(rhs[i]));
    }
}
)";
}

std::string wgsl_gather(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    outer: u32,
    axis_dim: u32,
    idx_count: u32,
    inner: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> data: array<f16>;
@group(0) @binding(2) var<storage, read> indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.outer * p.idx_count * p.inner;
    let tid = gid.x;
    if (tid >= total) { return; }
    let in_idx = tid % p.inner;
    let tmp = tid / p.inner;
    let ii = tmp % p.idx_count;
    let o = tmp / p.idx_count;
    var raw = indices[ii];
    if (raw < 0) { raw += i32(p.axis_dim); }
    let src_idx = (o * p.axis_dim + u32(raw)) * p.inner + in_idx;
    dst[tid] = data[src_idx];
}
)";
    }
    return R"(struct Params {
    outer: u32,
    axis_dim: u32,
    idx_count: u32,
    inner: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(data[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn fetch(tid: u32, total: u32) -> f32 {
    if (tid >= total) { return 0.0; }
    let in_idx = tid % p.inner;
    let tmp = tid / p.inner;
    let ii = tmp % p.idx_count;
    let o = tmp / p.idx_count;
    var raw = indices[ii];
    if (raw < 0) { raw += i32(p.axis_dim); }
    return rd((o * p.axis_dim + u32(raw)) * p.inner + in_idx);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.outer * p.idx_count * p.inner;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(fetch(i0, total), fetch(i0 + 1u, total)));
}
)";
}

std::string wgsl_slice_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    in_n: u32, in_c: u32, in_h: u32, in_w: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    sn: u32, sc: u32, sh: u32, sw: u32,
    stn: u32, stc: u32, sth: u32, stw: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.on * p.oh * p.ow * p.oc;
    let i = gid.x;
    if (i >= total) { return; }
    let c = i % p.oc;
    var t = i / p.oc;
    let x = t % p.ow;
    t = t / p.ow;
    let y = t % p.oh;
    let b = t / p.oh;
    let ib = p.sn + b * p.stn;
    let iy = p.sh + y * p.sth;
    let ix = p.sw + x * p.stw;
    let ic = p.sc + c * p.stc;
    let src_idx = (((ib * p.in_h + iy) * p.in_w + ix) * p.in_c) + ic;
    dst[i] = src[src_idx];
}
)";
    }
    return R"(struct Params {
    in_n: u32, in_c: u32, in_h: u32, in_w: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    sn: u32, sc: u32, sh: u32, sw: u32,
    stn: u32, stc: u32, sth: u32, stw: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(src[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn sample_one(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let c = i % p.oc;
    var t = i / p.oc;
    let x = t % p.ow;
    t = t / p.ow;
    let y = t % p.oh;
    let b = t / p.oh;
    let src_idx = ((((p.sn + b * p.stn) * p.in_h + (p.sh + y * p.sth)) * p.in_w + (p.sw + x * p.stw)) * p.in_c) + (p.sc + c * p.stc);
    return rd(src_idx);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.on * p.oh * p.ow * p.oc;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(sample_one(i0, total), sample_one(i0 + 1u, total)));
}
)";
}

std::string wgsl_concat_nhwc(bool use_f16)
{
    // Writes one source tensor into dst at offset (off_n, off_c, off_h, off_w).
    // Supports arbitrary number of concat inputs cleanly and stays well within
    // WebGPU binding limits.
    if (use_f16) {
        return R"(enable f16;
struct Params {
    in_n: u32, in_c: u32, in_h: u32, in_w: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    off_n: u32, off_c: u32, off_h: u32, off_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.in_n * p.in_h * p.in_w * p.in_c;
    let i = gid.x;
    if (i >= total) { return; }
    let c = i % p.in_c;
    var t = i / p.in_c;
    let x = t % p.in_w;
    t = t / p.in_w;
    let y = t % p.in_h;
    let b = t / p.in_h;
    let dst_idx = ((((b + p.off_n) * p.oh + (y + p.off_h)) * p.ow + (x + p.off_w)) * p.oc) + (c + p.off_c);
    dst[dst_idx] = src[i];
}
)";
    }
    // In fallback mode, each output pair in dst is written by reading from src
    // only when inside this input's slice? Wait: concat across multiple passes
    // requires channel-even offsets (all MLVC concats have even channels >= 16!).
    return R"(struct Params {
    in_n: u32, in_c: u32, in_h: u32, in_w: u32,
    on: u32, oc: u32, oh: u32, ow: u32,
    off_n: u32, off_c: u32, off_h: u32, off_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pairs = (p.in_n * p.in_h * p.in_w * p.in_c) >> 1u;
    let pi = gid.x;
    if (pi >= total_pairs) { return; }
    let in_c2 = p.in_c >> 1u;
    let c2 = pi % in_c2;
    var t = pi / in_c2;
    let x = t % p.in_w;
    t = t / p.in_w;
    let y = t % p.in_h;
    let b = t / p.in_h;
    let dst_pair = ((((b + p.off_n) * p.oh + (y + p.off_h)) * p.ow + (x + p.off_w)) * (p.oc >> 1u)) + ((p.off_c >> 1u) + c2);
    dst[dst_pair] = src[pi];
}
)";
}

std::string wgsl_depth_to_space_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    n: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    block: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.out_h * p.out_w * p.out_c;
    let i = gid.x;
    if (i >= total) { return; }
    let oc = i % p.out_c;
    var t = i / p.out_c;
    let ox = t % p.out_w;
    t = t / p.out_w;
    let oy = t % p.out_h;
    let b = t / p.out_h;
    let iy = oy / p.block;
    let dy = oy % p.block;
    let ix = ox / p.block;
    let dx = ox % p.block;
    let ic = (oc * p.block + dy) * p.block + dx;
    let src_idx = (((b * p.in_h + iy) * p.in_w + ix) * p.in_c) + ic;
    dst[i] = src[src_idx];
}
)";
    }
    return R"(struct Params {
    n: u32, in_c: u32, in_h: u32, in_w: u32,
    block: u32, out_c: u32, out_h: u32, out_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(src[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn sample_one(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let oc = i % p.out_c;
    var t = i / p.out_c;
    let ox = t % p.out_w;
    t = t / p.out_w;
    let oy = t % p.out_h;
    let b = t / p.out_h;
    let ic = (oc * p.block + (oy % p.block)) * p.block + (ox % p.block);
    return rd((((b * p.in_h + (oy / p.block)) * p.in_w + (ox / p.block)) * p.in_c) + ic);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.out_h * p.out_w * p.out_c;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(sample_one(i0, total), sample_one(i0 + 1u, total)));
}
)";
}

std::string wgsl_space_to_depth_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    n: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    block: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.out_h * p.out_w * p.out_c;
    let i = gid.x;
    if (i >= total) { return; }
    let oc = i % p.out_c;
    var t = i / p.out_c;
    let ox = t % p.out_w;
    t = t / p.out_w;
    let oy = t % p.out_h;
    let b = t / p.out_h;
    let b2 = p.block * p.block;
    let ic = oc / b2;
    let rem = oc % b2;
    let dy = rem / p.block;
    let dx = rem % p.block;
    let iy = oy * p.block + dy;
    let ix = ox * p.block + dx;
    let src_idx = (((b * p.in_h + iy) * p.in_w + ix) * p.in_c) + ic;
    dst[i] = src[src_idx];
}
)";
    }
    return R"(struct Params {
    n: u32, in_c: u32, in_h: u32, in_w: u32,
    block: u32, out_c: u32, out_h: u32, out_w: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(src[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn sample_one(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let oc = i % p.out_c;
    var t = i / p.out_c;
    let ox = t % p.out_w;
    t = t / p.out_w;
    let oy = t % p.out_h;
    let b = t / p.out_h;
    let b2 = p.block * p.block;
    let ic = oc / b2;
    let rem = oc % b2;
    let iy = oy * p.block + (rem / p.block);
    let ix = ox * p.block + (rem % p.block);
    return rd((((b * p.in_h + iy) * p.in_w + ix) * p.in_c) + ic);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.out_h * p.out_w * p.out_c;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(sample_one(i0, total), sample_one(i0 + 1u, total)));
}
)";
}

std::string wgsl_nchw_to_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    n: u32,
    c: u32,
    hw: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.hw;
    let i = gid.x;
    if (i >= total) { return; }
    let ch = i % p.c;
    let px = i / p.c;
    let b = px / p.hw;
    let s = px % p.hw;
    dst[i] = src[(b * p.c + ch) * p.hw + s];
}
)";
    }
    return R"(struct Params { n: u32, c: u32, hw: u32, pad: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(src[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn fetch(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let ch = i % p.c;
    let px = i / p.c;
    return rd(((px / p.hw) * p.c + ch) * p.hw + (px % p.hw));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.hw;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(fetch(i0, total), fetch(i0 + 1u, total)));
}
)";
}

std::string wgsl_nhwc_to_nchw(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    n: u32,
    c: u32,
    hw: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.hw;
    let i = gid.x;
    if (i >= total) { return; }
    let s = i % p.hw;
    let tmp = i / p.hw;
    let ch = tmp % p.c;
    let b = tmp / p.c;
    dst[i] = src[(b * p.hw + s) * p.c + ch];
}
)";
    }
    return R"(struct Params { n: u32, c: u32, hw: u32, pad: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn rd(idx: u32) -> f32 {
    let pair = unpack2x16float(src[idx >> 1u]);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
fn fetch(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let s = i % p.hw;
    let tmp = i / p.hw;
    let ch = tmp % p.c;
    let b = tmp / p.c;
    return rd((b * p.hw + s) * p.c + ch);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.hw;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    dst[gid.x] = pack2x16float(vec2<f32>(fetch(i0, total), fetch(i0 + 1u, total)));
}
)";
}

std::string wgsl_conv_generic_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    n: u32, ic: u32, ih: u32, iw: u32,
    oc: u32, oh: u32, ow: u32, groups: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    ph: u32, pw: u32, dh: u32, dw: u32,
    has_bias: u32, epilogue: u32, pad0: u32, pad1: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f16>; // OIHW
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read> residual: array<f16>;
@group(0) @binding(5) var<storage, read_write> outp: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.oh * p.ow * p.oc;
    let idx = gid.x;
    if (idx >= total) { return; }
    let oc = idx % p.oc;
    var t = idx / p.oc;
    let ox = t % p.ow;
    t = t / p.ow;
    let oy = t % p.oh;
    let b = t / p.oh;

    let ic_per_g = p.ic / p.groups;
    let oc_per_g = p.oc / p.groups;
    let g = oc / oc_per_g;
    let ic_start = g * ic_per_g;

    var acc: f32 = select(0.0, f32(bias[oc]), p.has_bias != 0u);
    let kplane = p.kh * p.kw;
    let w_oc_base = oc * ic_per_g * kplane;

    for (var ky: u32 = 0u; ky < p.kh; ky++) {
        let iy_s = i32(oy * p.sh + ky * p.dh) - i32(p.ph);
        if (iy_s < 0 || iy_s >= i32(p.ih)) { continue; }
        let iy = u32(iy_s);
        for (var kx: u32 = 0u; kx < p.kw; kx++) {
            let ix_s = i32(ox * p.sw + kx * p.dw) - i32(p.pw);
            if (ix_s < 0 || ix_s >= i32(p.iw)) { continue; }
            let ix = u32(ix_s);
            let in_base = ((b * p.ih + iy) * p.iw + ix) * p.ic + ic_start;
            let k_off = ky * p.kw + kx;
            for (var c: u32 = 0u; c < ic_per_g; c++) {
                acc = fma(f32(inp[in_base + c]),
                          f32(weight[w_oc_base + c * kplane + k_off]), acc);
            }
        }
    }
    var hv = f16(acc);
    if (p.epilogue == 1u) {
        let fv = f32(hv);
        hv = select(hv, f16(fv * p.alpha), fv < 0.0);
    } else if (p.epilogue == 2u) {
        hv = f16(clamp(f32(hv), p.clamp_min, p.clamp_max));
    } else if (p.epilogue == 3u) {
        hv = f16(f32(hv) + f32(residual[idx]));
    }
    outp[idx] = hv;
}
)";
    }
    return R"(struct Params {
    n: u32, ic: u32, ih: u32, iw: u32,
    oc: u32, oh: u32, ow: u32, groups: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    ph: u32, pw: u32, dh: u32, dw: u32,
    has_bias: u32, epilogue: u32, pad0: u32, pad1: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<u32>;
@group(0) @binding(4) var<storage, read> residual: array<u32>;
@group(0) @binding(5) var<storage, read_write> outp: array<u32>;

fn rd_in(idx: u32) -> f32 { let pr = unpack2x16float(inp[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }
fn rd_w(idx: u32) -> f32 { let pr = unpack2x16float(weight[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }
fn rd_b(idx: u32) -> f32 { let pr = unpack2x16float(bias[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }
fn rd_r(idx: u32) -> f32 { let pr = unpack2x16float(residual[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }
fn r16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }

fn eval_one(idx: u32, total: u32) -> f32 {
    if (idx >= total) { return 0.0; }
    let oc = idx % p.oc;
    var t = idx / p.oc;
    let ox = t % p.ow;
    t = t / p.ow;
    let oy = t % p.oh;
    let b = t / p.oh;
    let ic_per_g = p.ic / p.groups;
    let oc_per_g = p.oc / p.groups;
    let g = oc / oc_per_g;
    let ic_start = g * ic_per_g;
    var acc: f32 = select(0.0, rd_b(oc), p.has_bias != 0u);
    let kplane = p.kh * p.kw;
    let w_oc_base = oc * ic_per_g * kplane;
    for (var ky: u32 = 0u; ky < p.kh; ky++) {
        let iy_s = i32(oy * p.sh + ky * p.dh) - i32(p.ph);
        if (iy_s < 0 || iy_s >= i32(p.ih)) { continue; }
        let iy = u32(iy_s);
        for (var kx: u32 = 0u; kx < p.kw; kx++) {
            let ix_s = i32(ox * p.sw + kx * p.dw) - i32(p.pw);
            if (ix_s < 0 || ix_s >= i32(p.iw)) { continue; }
            let ix = u32(ix_s);
            let in_base = ((b * p.ih + iy) * p.iw + ix) * p.ic + ic_start;
            let k_off = ky * p.kw + kx;
            for (var c: u32 = 0u; c < ic_per_g; c++) {
                acc = fma(rd_in(in_base + c), rd_w(w_oc_base + c * kplane + k_off), acc);
            }
        }
    }
    var fv = r16(acc);
    if (p.epilogue == 1u) { fv = select(fv, r16(fv * p.alpha), fv < 0.0); }
    else if (p.epilogue == 2u) { fv = r16(clamp(fv, p.clamp_min, p.clamp_max)); }
    else if (p.epilogue == 3u) { fv = r16(fv + rd_r(idx)); }
    return fv;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.oh * p.ow * p.oc;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    outp[gid.x] = pack2x16float(vec2<f32>(eval_one(i0, total), eval_one(i0 + 1u, total)));
}
)";
}

std::string wgsl_pointwise_gemm(bool use_f16, int tile_variant, bool is_reglu)
{
    const bool small_m = (tile_variant == 0);
    const bool wide_n = (tile_variant == 2);
    if (!use_f16) {
        // Fallback path using u32-packed vec2<f32> tiled GEMM (exact FP16 rounding).
        // B is packed [K, N] in u32 pairs (N is multiple of 4; for ReGLU, N = 2*G
        // with interleaved [gate_v4, lin_v4] pairs of 2 u32s each).
        if (is_reglu) {
            return R"(struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read> B: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<u32>;
@group(0) @binding(4) var<storage, read> residual: array<u32>;
@group(0) @binding(5) var<storage, read_write> C: array<u32>;

fn r16v4(v: vec4<f32>) -> vec4<f32> {
    let p0 = unpack2x16float(pack2x16float(v.xy));
    let p1 = unpack2x16float(pack2x16float(v.zw));
    return vec4<f32>(p0.x, p0.y, p1.x, p1.y);
}
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g4 = gid.x;
    let row = gid.y;
    let G4 = p.n >> 3u; // p.n = 2 * G, G4 = G / 4
    if (g4 >= G4 || row >= p.m) { return; }
    let K2 = p.k >> 1u;
    let N2 = p.n >> 1u;
    var acc_g = vec4<f32>(0.0);
    var acc_l = vec4<f32>(0.0);
    if (p.has_bias != 0u) {
        let b_base = g4 * 4u;
        let bg0 = unpack2x16float(bias[b_base + 0u]);
        let bg1 = unpack2x16float(bias[b_base + 1u]);
        let bl0 = unpack2x16float(bias[b_base + 2u]);
        let bl1 = unpack2x16float(bias[b_base + 3u]);
        acc_g = vec4<f32>(bg0, bg1);
        acc_l = vec4<f32>(bl0, bl1);
    }
    let a_row = row * K2;
    let b_col_u32 = g4 * 4u;
    for (var k2: u32 = 0u; k2 < K2; k2++) {
        let av = unpack2x16float(A[a_row + k2]);
        let b0 = (k2 * 2u) * N2 + b_col_u32;
        let b1 = (k2 * 2u + 1u) * N2 + b_col_u32;
        let bg_0 = vec4<f32>(unpack2x16float(B[b0 + 0u]), unpack2x16float(B[b0 + 1u]));
        let bl_0 = vec4<f32>(unpack2x16float(B[b0 + 2u]), unpack2x16float(B[b0 + 3u]));
        let bg_1 = vec4<f32>(unpack2x16float(B[b1 + 0u]), unpack2x16float(B[b1 + 1u]));
        let bl_1 = vec4<f32>(unpack2x16float(B[b1 + 2u]), unpack2x16float(B[b1 + 3u]));
        acc_g = fma(vec4<f32>(av.x), bg_0, acc_g);
        acc_l = fma(vec4<f32>(av.x), bl_0, acc_l);
        acc_g = fma(vec4<f32>(av.y), bg_1, acc_g);
        acc_l = fma(vec4<f32>(av.y), bl_1, acc_l);
    }
    let g16 = r16v4(clamp(r16v4(acc_g), vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max)));
    let l16 = r16v4(acc_l);
    let outv = g16 * l16;
    let out_u32 = row * (G4 * 2u) + g4 * 2u;
    C[out_u32 + 0u] = pack2x16float(outv.xy);
    C[out_u32 + 1u] = pack2x16float(outv.zw);
}
)";
        }
        return R"(struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read> B: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<u32>;
@group(0) @binding(4) var<storage, read> residual: array<u32>;
@group(0) @binding(5) var<storage, read_write> C: array<u32>;

fn r16v4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v.zw)));
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col4 = gid.x;
    let row = gid.y;
    let N4 = p.n >> 2u;
    if (col4 >= N4 || row >= p.m) { return; }
    let K2 = p.k >> 1u;
    let N2 = p.n >> 1u;
    var acc = vec4<f32>(0.0);
    if (p.has_bias != 0u) {
        acc = vec4<f32>(unpack2x16float(bias[col4 * 2u]), unpack2x16float(bias[col4 * 2u + 1u]));
    }
    let a_row = row * K2;
    let b_col = col4 * 2u;
    for (var k2: u32 = 0u; k2 < K2; k2++) {
        let av = unpack2x16float(A[a_row + k2]);
        let b0 = (k2 * 2u) * N2 + b_col;
        let b1 = (k2 * 2u + 1u) * N2 + b_col;
        let bv0 = vec4<f32>(unpack2x16float(B[b0]), unpack2x16float(B[b0 + 1u]));
        let bv1 = vec4<f32>(unpack2x16float(B[b1]), unpack2x16float(B[b1 + 1u]));
        acc = fma(vec4<f32>(av.x), bv0, acc);
        acc = fma(vec4<f32>(av.y), bv1, acc);
    }
    var outv = r16v4(acc);
    let out_u32 = row * N2 + b_col;
    if (p.epilogue == 1u) {
        outv = select(outv, outv * p.alpha, outv < vec4<f32>(0.0));
    } else if (p.epilogue == 2u) {
        outv = clamp(outv, vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max));
    } else if (p.epilogue == 3u) {
        let rv = vec4<f32>(unpack2x16float(residual[out_u32]), unpack2x16float(residual[out_u32 + 1u]));
        outv = outv + rv;
    }
    C[out_u32 + 0u] = pack2x16float(outv.xy);
    C[out_u32 + 1u] = pack2x16float(outv.zw);
}
)";
    }

    // Native `enable f16;` high-throughput shared-memory Tiled GEMM!
    // Supports:
    //  - Wide-N   (BM=64, BN=128, BK=16), workgroup(16,16,1): 4 rows x 2 vec4 cols/thread (or 4 rows x (gate_v4 + lin_v4) for ReGLU)
    //  - Medium-N (BM=64, BN=64,  BK=16), workgroup(16,16,1): 4 rows x 1 vec4 col/thread (or 2 rows x (gate_v4 + lin_v4) for ReGLU)
    //  - Small-M  (BM=16, BN=64,  BK=16), workgroup(16,4,1):  4 rows x 1 vec4 col/thread (or 2 rows x (gate_v4 + lin_v4) for ReGLU)
    if (is_reglu) {
        if (wide_n) {
            // Wide 64x128 ReGLU tile (BG=64 output channels = 128 conv channels):
            // 256 threads (16x16), 4 rows x (gate_v4 + lin_v4) per thread, 100% coalesced 32-wide Bs loads!
            return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 256>;  // 64 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 512>;  // 16 k x 32 vec4 (16 gate + 16 lin)

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_g4 = wgid.x * 16u;
    let wg_m = wgid.y * 64u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u; // = 2 * G4
    let G4 = N4 >> 1u;

    let tx = lid.x; // 0..15 (gate vec4 index within tile)
    let ty = lid.y; // 0..15 (4 rows each -> 64 rows)
    let g4 = wg_g4 + tx;

    var bg_init = vec4<f32>(0.0);
    var bl_init = vec4<f32>(0.0);
    if (p.has_bias != 0u && g4 < G4) {
        bg_init = vec4<f32>(bias[g4 * 2u + 0u]);
        bl_init = vec4<f32>(bias[g4 * 2u + 1u]);
    }
    var acc_g0 = bg_init; var acc_l0 = bl_init;
    var acc_g1 = bg_init; var acc_l1 = bl_init;
    var acc_g2 = bg_init; var acc_l2 = bl_init;
    var acc_g3 = bg_init; var acc_l3 = bl_init;

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    let bk0 = tid >> 5u;          // 0..7
    let bk1 = bk0 + 8u;           // 8..15
    let bn4 = tid & 31u;          // 0..31 (32 consecutive vec4 = 256B coalesced per warp!)
    let gn4 = (wg_g4 << 1u) + bn4;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let k_base = kt4 << 2u;
        let gk0 = k_base + bk0;
        let gk1 = k_base + bk1;
        if (gk0 < p.k && gn4 < N4) {
            Bs[tid] = B[gk0 * N4 + gn4];
        } else {
            Bs[tid] = vec4<f16>(0.0h);
        }
        if (gk1 < p.k && gn4 < N4) {
            Bs[tid + 256u] = B[gk1 * N4 + gn4];
        } else {
            Bs[tid + 256u] = vec4<f16>(0.0h);
        }
        workgroupBarrier();

        let rbase = ty * 4u;
        let b_off_g = tx * 2u;
        let b_off_l = b_off_g + 1u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let a0 = vec4<f32>(As[(rbase + 0u) * 4u + kk4]);
            let a1 = vec4<f32>(As[(rbase + 1u) * 4u + kk4]);
            let a2 = vec4<f32>(As[(rbase + 2u) * 4u + kk4]);
            let a3 = vec4<f32>(As[(rbase + 3u) * 4u + kk4]);
            let kb = kk4 << 2u;

            let bg0 = vec4<f32>(Bs[(kb + 0u) * 32u + b_off_g]);
            let bl0 = vec4<f32>(Bs[(kb + 0u) * 32u + b_off_l]);
            acc_g0 = fma(vec4<f32>(a0.x), bg0, acc_g0); acc_l0 = fma(vec4<f32>(a0.x), bl0, acc_l0);
            acc_g1 = fma(vec4<f32>(a1.x), bg0, acc_g1); acc_l1 = fma(vec4<f32>(a1.x), bl0, acc_l1);
            acc_g2 = fma(vec4<f32>(a2.x), bg0, acc_g2); acc_l2 = fma(vec4<f32>(a2.x), bl0, acc_l2);
            acc_g3 = fma(vec4<f32>(a3.x), bg0, acc_g3); acc_l3 = fma(vec4<f32>(a3.x), bl0, acc_l3);

            let bg1 = vec4<f32>(Bs[(kb + 1u) * 32u + b_off_g]);
            let bl1 = vec4<f32>(Bs[(kb + 1u) * 32u + b_off_l]);
            acc_g0 = fma(vec4<f32>(a0.y), bg1, acc_g0); acc_l0 = fma(vec4<f32>(a0.y), bl1, acc_l0);
            acc_g1 = fma(vec4<f32>(a1.y), bg1, acc_g1); acc_l1 = fma(vec4<f32>(a1.y), bl1, acc_l1);
            acc_g2 = fma(vec4<f32>(a2.y), bg1, acc_g2); acc_l2 = fma(vec4<f32>(a2.y), bl1, acc_l2);
            acc_g3 = fma(vec4<f32>(a3.y), bg1, acc_g3); acc_l3 = fma(vec4<f32>(a3.y), bl1, acc_l3);

            let bg2 = vec4<f32>(Bs[(kb + 2u) * 32u + b_off_g]);
            let bl2 = vec4<f32>(Bs[(kb + 2u) * 32u + b_off_l]);
            acc_g0 = fma(vec4<f32>(a0.z), bg2, acc_g0); acc_l0 = fma(vec4<f32>(a0.z), bl2, acc_l0);
            acc_g1 = fma(vec4<f32>(a1.z), bg2, acc_g1); acc_l1 = fma(vec4<f32>(a1.z), bl2, acc_l1);
            acc_g2 = fma(vec4<f32>(a2.z), bg2, acc_g2); acc_l2 = fma(vec4<f32>(a2.z), bl2, acc_l2);
            acc_g3 = fma(vec4<f32>(a3.z), bg2, acc_g3); acc_l3 = fma(vec4<f32>(a3.z), bl2, acc_l3);

            let bg3 = vec4<f32>(Bs[(kb + 3u) * 32u + b_off_g]);
            let bl3 = vec4<f32>(Bs[(kb + 3u) * 32u + b_off_l]);
            acc_g0 = fma(vec4<f32>(a0.w), bg3, acc_g0); acc_l0 = fma(vec4<f32>(a0.w), bl3, acc_l0);
            acc_g1 = fma(vec4<f32>(a1.w), bg3, acc_g1); acc_l1 = fma(vec4<f32>(a1.w), bl3, acc_l1);
            acc_g2 = fma(vec4<f32>(a2.w), bg3, acc_g2); acc_l2 = fma(vec4<f32>(a2.w), bl3, acc_l2);
            acc_g3 = fma(vec4<f32>(a3.w), bg3, acc_g3); acc_l3 = fma(vec4<f32>(a3.w), bl3, acc_l3);
        }
        workgroupBarrier();
    }

    if (g4 < G4) {
        let cmin = vec4<f32>(p.clamp_min);
        let cmax = vec4<f32>(p.clamp_max);
        let r0 = wg_m + ty * 4u;
        if (r0 + 0u < p.m) {
            let g0 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g0)), cmin, cmax)));
            C[(r0 + 0u) * G4 + g4] = vec4<f16>(g0 * vec4<f32>(vec4<f16>(acc_l0)));
        }
        if (r0 + 1u < p.m) {
            let g1 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g1)), cmin, cmax)));
            C[(r0 + 1u) * G4 + g4] = vec4<f16>(g1 * vec4<f32>(vec4<f16>(acc_l1)));
        }
        if (r0 + 2u < p.m) {
            let g2 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g2)), cmin, cmax)));
            C[(r0 + 2u) * G4 + g4] = vec4<f16>(g2 * vec4<f32>(vec4<f16>(acc_l2)));
        }
        if (r0 + 3u < p.m) {
            let g3 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g3)), cmin, cmax)));
            C[(r0 + 3u) * G4 + g4] = vec4<f16>(g3 * vec4<f32>(vec4<f16>(acc_l3)));
        }
    }
}
)";
        }
        if (small_m) {
            return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 64>;   // 16 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 256>;  // 16 k x 16 vec4 (8 gate + 8 lin)

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_g4 = wgid.x * 8u;
    let wg_m = wgid.y * 16u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u; // = 2 * G4
    let G4 = N4 >> 1u;

    let tx = lid.x; // 0..7 (gate vec4 index within tile)
    let ty = lid.y; // 0..7 (2 rows each -> 16 rows)
    let g4 = wg_g4 + tx;
    let row0 = wg_m + ty * 2u;
    let row1 = row0 + 1u;

    var acc_g0 = vec4<f32>(0.0);
    var acc_l0 = vec4<f32>(0.0);
    var acc_g1 = vec4<f32>(0.0);
    var acc_l1 = vec4<f32>(0.0);
    if (p.has_bias != 0u && g4 < G4) {
        let bg = vec4<f32>(bias[g4 * 2u + 0u]);
        let bl = vec4<f32>(bias[g4 * 2u + 1u]);
        acc_g0 = bg; acc_l0 = bl;
        acc_g1 = bg; acc_l1 = bl;
    }

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        // 64 threads load 256 vec4s into Bs (4 each).
        let k_base = kt4 << 2u;
        for (var u: u32 = 0u; u < 4u; u++) {
            let b_slot = tid + u * 64u;
            let bk = b_slot >> 4u;
            let bn4 = b_slot & 15u;
            let gk = k_base + bk;
            let gn4 = (wg_g4 << 1u) + bn4;
            if (gk < p.k && gn4 < N4) {
                Bs[b_slot] = B[gk * N4 + gn4];
            } else {
                Bs[b_slot] = vec4<f16>(0.0h);
            }
        }
        workgroupBarrier();

        let b_off_g = tx * 2u;
        let b_off_l = b_off_g + 1u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let av0 = vec4<f32>(As[(ty * 2u) * 4u + kk4]);
            let av1 = vec4<f32>(As[(ty * 2u + 1u) * 4u + kk4]);
            let kb = kk4 << 2u;

            let bg0 = vec4<f32>(Bs[(kb + 0u) * 16u + b_off_g]);
            let bl0 = vec4<f32>(Bs[(kb + 0u) * 16u + b_off_l]);
            let bg1 = vec4<f32>(Bs[(kb + 1u) * 16u + b_off_g]);
            let bl1 = vec4<f32>(Bs[(kb + 1u) * 16u + b_off_l]);
            let bg2 = vec4<f32>(Bs[(kb + 2u) * 16u + b_off_g]);
            let bl2 = vec4<f32>(Bs[(kb + 2u) * 16u + b_off_l]);
            let bg3 = vec4<f32>(Bs[(kb + 3u) * 16u + b_off_g]);
            let bl3 = vec4<f32>(Bs[(kb + 3u) * 16u + b_off_l]);

            acc_g0 = fma(vec4<f32>(av0.x), bg0, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.x), bl0, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.y), bg1, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.y), bl1, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.z), bg2, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.z), bl2, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.w), bg3, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.w), bl3, acc_l0);

            acc_g1 = fma(vec4<f32>(av1.x), bg0, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.x), bl0, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.y), bg1, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.y), bl1, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.z), bg2, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.z), bl2, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.w), bg3, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.w), bl3, acc_l1);
        }
        workgroupBarrier();
    }

    if (g4 < G4) {
        let cmin = vec4<f32>(p.clamp_min);
        let cmax = vec4<f32>(p.clamp_max);
        if (row0 < p.m) {
            let g0 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g0)), cmin, cmax)));
            let l0 = vec4<f32>(vec4<f16>(acc_l0));
            C[row0 * G4 + g4] = vec4<f16>(g0 * l0);
        }
        if (row1 < p.m) {
            let g1 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g1)), cmin, cmax)));
            let l1 = vec4<f32>(vec4<f16>(acc_l1));
            C[row1 * G4 + g4] = vec4<f16>(g1 * l1);
        }
    }
}
)";
        }
        // Full 64x64 ReGLU tile: 256 threads (8 x 32), 1 vec4 load each for As & Bs!
        return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 256>;  // 64 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 256>;  // 16 k x 16 vec4 (8 gate + 8 lin)

@compute @workgroup_size(8, 32, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_g4 = wgid.x * 8u;
    let wg_m = wgid.y * 64u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u; // = 2 * G4
    let G4 = N4 >> 1u;

    let tx = lid.x; // 0..7
    let ty = lid.y; // 0..31 (2 rows each -> 64 rows)
    let g4 = wg_g4 + tx;
    let row0 = wg_m + ty * 2u;
    let row1 = row0 + 1u;

    var acc_g0 = vec4<f32>(0.0);
    var acc_l0 = vec4<f32>(0.0);
    var acc_g1 = vec4<f32>(0.0);
    var acc_l1 = vec4<f32>(0.0);
    if (p.has_bias != 0u && g4 < G4) {
        let bg = vec4<f32>(bias[g4 * 2u + 0u]);
        let bl = vec4<f32>(bias[g4 * 2u + 1u]);
        acc_g0 = bg; acc_l0 = bl;
        acc_g1 = bg; acc_l1 = bl;
    }

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    let bk = tid >> 4u;
    let bn4 = tid & 15u;
    let gn4 = (wg_g4 << 1u) + bn4;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let gk = (kt4 << 2u) + bk;
        if (gk < p.k && gn4 < N4) {
            Bs[tid] = B[gk * N4 + gn4];
        } else {
            Bs[tid] = vec4<f16>(0.0h);
        }
        workgroupBarrier();

        let b_off_g = tx * 2u;
        let b_off_l = b_off_g + 1u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let av0 = vec4<f32>(As[(ty * 2u) * 4u + kk4]);
            let av1 = vec4<f32>(As[(ty * 2u + 1u) * 4u + kk4]);
            let kb = kk4 << 2u;

            let bg0 = vec4<f32>(Bs[(kb + 0u) * 16u + b_off_g]);
            let bl0 = vec4<f32>(Bs[(kb + 0u) * 16u + b_off_l]);
            let bg1 = vec4<f32>(Bs[(kb + 1u) * 16u + b_off_g]);
            let bl1 = vec4<f32>(Bs[(kb + 1u) * 16u + b_off_l]);
            let bg2 = vec4<f32>(Bs[(kb + 2u) * 16u + b_off_g]);
            let bl2 = vec4<f32>(Bs[(kb + 2u) * 16u + b_off_l]);
            let bg3 = vec4<f32>(Bs[(kb + 3u) * 16u + b_off_g]);
            let bl3 = vec4<f32>(Bs[(kb + 3u) * 16u + b_off_l]);

            acc_g0 = fma(vec4<f32>(av0.x), bg0, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.x), bl0, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.y), bg1, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.y), bl1, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.z), bg2, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.z), bl2, acc_l0);
            acc_g0 = fma(vec4<f32>(av0.w), bg3, acc_g0);
            acc_l0 = fma(vec4<f32>(av0.w), bl3, acc_l0);

            acc_g1 = fma(vec4<f32>(av1.x), bg0, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.x), bl0, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.y), bg1, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.y), bl1, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.z), bg2, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.z), bl2, acc_l1);
            acc_g1 = fma(vec4<f32>(av1.w), bg3, acc_g1);
            acc_l1 = fma(vec4<f32>(av1.w), bl3, acc_l1);
        }
        workgroupBarrier();
    }

    if (g4 < G4) {
        let cmin = vec4<f32>(p.clamp_min);
        let cmax = vec4<f32>(p.clamp_max);
        if (row0 < p.m) {
            let g0 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g0)), cmin, cmax)));
            let l0 = vec4<f32>(vec4<f16>(acc_l0));
            C[row0 * G4 + g4] = vec4<f16>(g0 * l0);
        }
        if (row1 < p.m) {
            let g1 = vec4<f32>(vec4<f16>(clamp(vec4<f32>(vec4<f16>(acc_g1)), cmin, cmax)));
            let l1 = vec4<f32>(vec4<f16>(acc_l1));
            C[row1 * G4 + g4] = vec4<f16>(g1 * l1);
        }
    }
}
)";
    }

    if (wide_n) {
        // Wide 64x128 tile: workgroup(16, 16, 1) -> 256 threads, 4 rows x 2 vec4 cols/thread!
        // Halves global A bandwidth and boosts SMEM arithmetic intensity to 2.67 FMAs/byte.
        return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 256>;  // 64 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 512>;  // 16 k x 32 vec4 (BN=128)

fn finalize_v4(acc: vec4<f32>, out_idx: u32) -> vec4<f16> {
    var hv = vec4<f16>(acc);
    if (p.epilogue == 1u) {
        let fv = vec4<f32>(hv);
        hv = vec4<f16>(select(fv, fv * p.alpha, fv < vec4<f32>(0.0)));
    } else if (p.epilogue == 2u) {
        hv = vec4<f16>(clamp(vec4<f32>(hv), vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max)));
    } else if (p.epilogue == 3u) {
        hv = vec4<f16>(vec4<f32>(hv) + vec4<f32>(residual[out_idx]));
    }
    return hv;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_n4 = wgid.x * 32u;
    let wg_m = wgid.y * 64u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u;
    let tx = lid.x;
    let ty = lid.y;
    let col4_a = wg_n4 + tx;
    let col4_b = wg_n4 + tx + 16u;

    var ba = vec4<f32>(0.0);
    var bb = vec4<f32>(0.0);
    if (p.has_bias != 0u) {
        if (col4_a < N4) { ba = vec4<f32>(bias[col4_a]); }
        if (col4_b < N4) { bb = vec4<f32>(bias[col4_b]); }
    }
    var acc0a = ba; var acc1a = ba; var acc2a = ba; var acc3a = ba;
    var acc0b = bb; var acc1b = bb; var acc2b = bb; var acc3b = bb;

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    let bk0 = tid >> 5u;
    let bk1 = bk0 + 8u;
    let bn4 = tid & 31u;
    let gn4 = wg_n4 + bn4;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let k_base = kt4 << 2u;
        let gk0 = k_base + bk0;
        let gk1 = k_base + bk1;
        if (gk0 < p.k && gn4 < N4) {
            Bs[tid] = B[gk0 * N4 + gn4];
        } else {
            Bs[tid] = vec4<f16>(0.0h);
        }
        if (gk1 < p.k && gn4 < N4) {
            Bs[tid + 256u] = B[gk1 * N4 + gn4];
        } else {
            Bs[tid + 256u] = vec4<f16>(0.0h);
        }
        workgroupBarrier();

        let rbase = ty * 4u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let a0 = vec4<f32>(As[(rbase + 0u) * 4u + kk4]);
            let a1 = vec4<f32>(As[(rbase + 1u) * 4u + kk4]);
            let a2 = vec4<f32>(As[(rbase + 2u) * 4u + kk4]);
            let a3 = vec4<f32>(As[(rbase + 3u) * 4u + kk4]);
            let kb = kk4 << 2u;

            let b0a = vec4<f32>(Bs[(kb + 0u) * 32u + tx]);
            let b0b = vec4<f32>(Bs[(kb + 0u) * 32u + tx + 16u]);
            acc0a = fma(vec4<f32>(a0.x), b0a, acc0a); acc0b = fma(vec4<f32>(a0.x), b0b, acc0b);
            acc1a = fma(vec4<f32>(a1.x), b0a, acc1a); acc1b = fma(vec4<f32>(a1.x), b0b, acc1b);
            acc2a = fma(vec4<f32>(a2.x), b0a, acc2a); acc2b = fma(vec4<f32>(a2.x), b0b, acc2b);
            acc3a = fma(vec4<f32>(a3.x), b0a, acc3a); acc3b = fma(vec4<f32>(a3.x), b0b, acc3b);

            let b1a = vec4<f32>(Bs[(kb + 1u) * 32u + tx]);
            let b1b = vec4<f32>(Bs[(kb + 1u) * 32u + tx + 16u]);
            acc0a = fma(vec4<f32>(a0.y), b1a, acc0a); acc0b = fma(vec4<f32>(a0.y), b1b, acc0b);
            acc1a = fma(vec4<f32>(a1.y), b1a, acc1a); acc1b = fma(vec4<f32>(a1.y), b1b, acc1b);
            acc2a = fma(vec4<f32>(a2.y), b1a, acc2a); acc2b = fma(vec4<f32>(a2.y), b1b, acc2b);
            acc3a = fma(vec4<f32>(a3.y), b1a, acc3a); acc3b = fma(vec4<f32>(a3.y), b1b, acc3b);

            let b2a = vec4<f32>(Bs[(kb + 2u) * 32u + tx]);
            let b2b = vec4<f32>(Bs[(kb + 2u) * 32u + tx + 16u]);
            acc0a = fma(vec4<f32>(a0.z), b2a, acc0a); acc0b = fma(vec4<f32>(a0.z), b2b, acc0b);
            acc1a = fma(vec4<f32>(a1.z), b2a, acc1a); acc1b = fma(vec4<f32>(a1.z), b2b, acc1b);
            acc2a = fma(vec4<f32>(a2.z), b2a, acc2a); acc2b = fma(vec4<f32>(a2.z), b2b, acc2b);
            acc3a = fma(vec4<f32>(a3.z), b2a, acc3a); acc3b = fma(vec4<f32>(a3.z), b2b, acc3b);

            let b3a = vec4<f32>(Bs[(kb + 3u) * 32u + tx]);
            let b3b = vec4<f32>(Bs[(kb + 3u) * 32u + tx + 16u]);
            acc0a = fma(vec4<f32>(a0.w), b3a, acc0a); acc0b = fma(vec4<f32>(a0.w), b3b, acc0b);
            acc1a = fma(vec4<f32>(a1.w), b3a, acc1a); acc1b = fma(vec4<f32>(a1.w), b3b, acc1b);
            acc2a = fma(vec4<f32>(a2.w), b3a, acc2a); acc2b = fma(vec4<f32>(a2.w), b3b, acc2b);
            acc3a = fma(vec4<f32>(a3.w), b3a, acc3a); acc3b = fma(vec4<f32>(a3.w), b3b, acc3b);
        }
        workgroupBarrier();
    }

    let r0 = wg_m + ty * 4u;
    if (col4_a < N4) {
        if (r0 + 0u < p.m) { let o = (r0 + 0u) * N4 + col4_a; C[o] = finalize_v4(acc0a, o); }
        if (r0 + 1u < p.m) { let o = (r0 + 1u) * N4 + col4_a; C[o] = finalize_v4(acc1a, o); }
        if (r0 + 2u < p.m) { let o = (r0 + 2u) * N4 + col4_a; C[o] = finalize_v4(acc2a, o); }
        if (r0 + 3u < p.m) { let o = (r0 + 3u) * N4 + col4_a; C[o] = finalize_v4(acc3a, o); }
    }
    if (col4_b < N4) {
        if (r0 + 0u < p.m) { let o = (r0 + 0u) * N4 + col4_b; C[o] = finalize_v4(acc0b, o); }
        if (r0 + 1u < p.m) { let o = (r0 + 1u) * N4 + col4_b; C[o] = finalize_v4(acc1b, o); }
        if (r0 + 2u < p.m) { let o = (r0 + 2u) * N4 + col4_b; C[o] = finalize_v4(acc2b, o); }
        if (r0 + 3u < p.m) { let o = (r0 + 3u) * N4 + col4_b; C[o] = finalize_v4(acc3b, o); }
    }
}
)";
    }

    if (small_m) {
        return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 64>;   // 16 rows x 4 vec4
var<workgroup> Bs: array<vec4<f16>, 256>;  // 16 k x 16 vec4

fn finalize_v4(acc: vec4<f32>, out_idx: u32) -> vec4<f16> {
    var hv = vec4<f16>(acc);
    if (p.epilogue == 1u) {
        let fv = vec4<f32>(hv);
        hv = vec4<f16>(select(fv, fv * p.alpha, fv < vec4<f32>(0.0)));
    } else if (p.epilogue == 2u) {
        hv = vec4<f16>(clamp(vec4<f32>(hv), vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max)));
    } else if (p.epilogue == 3u) {
        hv = vec4<f16>(vec4<f32>(hv) + vec4<f32>(residual[out_idx]));
    }
    return hv;
}

@compute @workgroup_size(16, 4, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_n4 = wgid.x * 16u;
    let wg_m = wgid.y * 16u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u;
    let tx = lid.x;
    let ty = lid.y;
    let col4 = wg_n4 + tx;

    var bval = vec4<f32>(0.0);
    if (p.has_bias != 0u && col4 < N4) { bval = vec4<f32>(bias[col4]); }
    var acc0 = bval; var acc1 = bval; var acc2 = bval; var acc3 = bval;

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let k_base = kt4 << 2u;
        for (var u: u32 = 0u; u < 4u; u++) {
            let bs = tid + u * 64u;
            let bk = bs >> 4u;
            let bn4 = bs & 15u;
            let gk = k_base + bk;
            let gn4 = wg_n4 + bn4;
            if (gk < p.k && gn4 < N4) {
                Bs[bs] = B[gk * N4 + gn4];
            } else {
                Bs[bs] = vec4<f16>(0.0h);
            }
        }
        workgroupBarrier();

        let rbase = ty * 4u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let a0 = vec4<f32>(As[(rbase + 0u) * 4u + kk4]);
            let a1 = vec4<f32>(As[(rbase + 1u) * 4u + kk4]);
            let a2 = vec4<f32>(As[(rbase + 2u) * 4u + kk4]);
            let a3 = vec4<f32>(As[(rbase + 3u) * 4u + kk4]);
            let kb = kk4 << 2u;
            let b0 = vec4<f32>(Bs[(kb + 0u) * 16u + tx]);
            let b1 = vec4<f32>(Bs[(kb + 1u) * 16u + tx]);
            let b2 = vec4<f32>(Bs[(kb + 2u) * 16u + tx]);
            let b3 = vec4<f32>(Bs[(kb + 3u) * 16u + tx]);

            acc0 = fma(vec4<f32>(a0.x), b0, fma(vec4<f32>(a0.y), b1, fma(vec4<f32>(a0.z), b2, fma(vec4<f32>(a0.w), b3, acc0))));
            acc1 = fma(vec4<f32>(a1.x), b0, fma(vec4<f32>(a1.y), b1, fma(vec4<f32>(a1.z), b2, fma(vec4<f32>(a1.w), b3, acc1))));
            acc2 = fma(vec4<f32>(a2.x), b0, fma(vec4<f32>(a2.y), b1, fma(vec4<f32>(a2.z), b2, fma(vec4<f32>(a2.w), b3, acc2))));
            acc3 = fma(vec4<f32>(a3.x), b0, fma(vec4<f32>(a3.y), b1, fma(vec4<f32>(a3.z), b2, fma(vec4<f32>(a3.w), b3, acc3))));
        }
        workgroupBarrier();
    }

    if (col4 < N4) {
        let r0 = wg_m + ty * 4u;
        if (r0 + 0u < p.m) { let o = (r0 + 0u) * N4 + col4; C[o] = finalize_v4(acc0, o); }
        if (r0 + 1u < p.m) { let o = (r0 + 1u) * N4 + col4; C[o] = finalize_v4(acc1, o); }
        if (r0 + 2u < p.m) { let o = (r0 + 2u) * N4 + col4; C[o] = finalize_v4(acc2, o); }
        if (r0 + 3u < p.m) { let o = (r0 + 3u) * N4 + col4; C[o] = finalize_v4(acc3, o); }
    }
}
)";
    }

    // Standard full 64x64 tile: workgroup(16, 16, 1) -> 256 threads!
    return R"(enable f16;
struct Params {
    m: u32, k: u32, n: u32, has_bias: u32,
    epilogue: u32, pad0: u32, pad1: u32, pad2: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> C: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 256>;  // 64 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 256>;  // 16 k x 16 vec4 (BN=64)

fn finalize_v4(acc: vec4<f32>, out_idx: u32) -> vec4<f16> {
    var hv = vec4<f16>(acc);
    if (p.epilogue == 1u) {
        let fv = vec4<f32>(hv);
        hv = vec4<f16>(select(fv, fv * p.alpha, fv < vec4<f32>(0.0)));
    } else if (p.epilogue == 2u) {
        hv = vec4<f16>(clamp(vec4<f32>(hv), vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max)));
    } else if (p.epilogue == 3u) {
        hv = vec4<f16>(vec4<f32>(hv) + vec4<f32>(residual[out_idx]));
    }
    return hv;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let wg_n4 = wgid.x * 16u;
    let wg_m = wgid.y * 64u;
    let K4 = p.k >> 2u;
    let N4 = p.n >> 2u;
    let tx = lid.x;
    let ty = lid.y;
    let col4 = wg_n4 + tx;

    var bval = vec4<f32>(0.0);
    if (p.has_bias != 0u && col4 < N4) { bval = vec4<f32>(bias[col4]); }
    var acc0 = bval; var acc1 = bval; var acc2 = bval; var acc3 = bval;

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;

    let bk = tid >> 4u;
    let bn4 = tid & 15u;
    let gn4 = wg_n4 + bn4;

    for (var kt4: u32 = 0u; kt4 < K4; kt4 += 4u) {
        let gk4 = kt4 + a_k4;
        if (a_gr < p.m && gk4 < K4) {
            As[tid] = A[a_gr * K4 + gk4];
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let gk = (kt4 << 2u) + bk;
        if (gk < p.k && gn4 < N4) {
            Bs[tid] = B[gk * N4 + gn4];
        } else {
            Bs[tid] = vec4<f16>(0.0h);
        }
        workgroupBarrier();

        let rbase = ty * 4u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let a0 = vec4<f32>(As[(rbase + 0u) * 4u + kk4]);
            let a1 = vec4<f32>(As[(rbase + 1u) * 4u + kk4]);
            let a2 = vec4<f32>(As[(rbase + 2u) * 4u + kk4]);
            let a3 = vec4<f32>(As[(rbase + 3u) * 4u + kk4]);
            let kb = kk4 << 2u;
            let b0 = vec4<f32>(Bs[(kb + 0u) * 16u + tx]);
            let b1 = vec4<f32>(Bs[(kb + 1u) * 16u + tx]);
            let b2 = vec4<f32>(Bs[(kb + 2u) * 16u + tx]);
            let b3 = vec4<f32>(Bs[(kb + 3u) * 16u + tx]);

            acc0 = fma(vec4<f32>(a0.x), b0, fma(vec4<f32>(a0.y), b1, fma(vec4<f32>(a0.z), b2, fma(vec4<f32>(a0.w), b3, acc0))));
            acc1 = fma(vec4<f32>(a1.x), b0, fma(vec4<f32>(a1.y), b1, fma(vec4<f32>(a1.z), b2, fma(vec4<f32>(a1.w), b3, acc1))));
            acc2 = fma(vec4<f32>(a2.x), b0, fma(vec4<f32>(a2.y), b1, fma(vec4<f32>(a2.z), b2, fma(vec4<f32>(a2.w), b3, acc2))));
            acc3 = fma(vec4<f32>(a3.x), b0, fma(vec4<f32>(a3.y), b1, fma(vec4<f32>(a3.z), b2, fma(vec4<f32>(a3.w), b3, acc3))));
        }
        workgroupBarrier();
    }

    if (col4 < N4) {
        let r0 = wg_m + ty * 4u;
        if (r0 + 0u < p.m) { let o = (r0 + 0u) * N4 + col4; C[o] = finalize_v4(acc0, o); }
        if (r0 + 1u < p.m) { let o = (r0 + 1u) * N4 + col4; C[o] = finalize_v4(acc1, o); }
        if (r0 + 2u < p.m) { let o = (r0 + 2u) * N4 + col4; C[o] = finalize_v4(acc2, o); }
        if (r0 + 3u < p.m) { let o = (r0 + 3u) * N4 + col4; C[o] = finalize_v4(acc3, o); }
    }
}
)";
}

std::string wgsl_spatial_conv_gemm(bool use_f16)
{
    // Implicit-GEMM Tiled Spatial Conv (groups == 1, Cout % 4 == 0, arbitrary Cin/kh/kw/s/p/d).
    // B is pre-packed [kh * kw * ic, oc] in vec4<f16> (row-major along oc).
    if (!use_f16) {
        return wgsl_conv_generic_nhwc(false);
    }
    return R"(enable f16;
struct Params {
    n: u32, ic: u32, ih: u32, iw: u32,
    oc: u32, oh: u32, ow: u32, k_total: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    ph: u32, pw: u32, dh: u32, dw: u32,
    has_bias: u32, epilogue: u32, pad0: u32, pad1: u32,
    alpha: f32, clamp_min: f32, clamp_max: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> outp: array<vec4<f16>>;

var<workgroup> As: array<vec4<f16>, 256>;  // 64 rows x 4 vec4 (BK=16)
var<workgroup> Bs: array<vec4<f16>, 256>;  // 16 k x 16 vec4 (BN=64)

fn load_im2col_pos(b: u32, oy: u32, ox: u32, k: u32) -> f16 {
    if (k >= p.k_total) { return 0.0h; }
    let c = k % p.ic;
    let ksp = k / p.ic;
    let ky = ksp / p.kw;
    let kx = ksp % p.kw;

    let iy_s = i32(oy * p.sh + ky * p.dh) - i32(p.ph);
    let ix_s = i32(ox * p.sw + kx * p.dw) - i32(p.pw);
    if (iy_s < 0 || iy_s >= i32(p.ih) || ix_s < 0 || ix_s >= i32(p.iw)) {
        return 0.0h;
    }
    let idx = (((b * p.ih + u32(iy_s)) * p.iw + u32(ix_s)) * p.ic) + c;
    return inp[idx];
}

fn finalize_v4(acc: vec4<f32>, out_idx: u32) -> vec4<f16> {
    var hv = vec4<f16>(acc);
    if (p.epilogue == 1u) {
        let fv = vec4<f32>(hv);
        hv = vec4<f16>(select(fv, fv * p.alpha, fv < vec4<f32>(0.0)));
    } else if (p.epilogue == 2u) {
        hv = vec4<f16>(clamp(vec4<f32>(hv), vec4<f32>(p.clamp_min), vec4<f32>(p.clamp_max)));
    } else if (p.epilogue == 3u) {
        hv = vec4<f16>(vec4<f32>(hv) + vec4<f32>(residual[out_idx]));
    }
    return hv;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let m_total = p.n * p.oh * p.ow;
    let wg_n4 = wgid.x * 16u;
    let wg_m = wgid.y * 64u;
    let N4 = p.oc >> 2u;
    let tx = lid.x;
    let ty = lid.y;
    let col4 = wg_n4 + tx;

    var bval = vec4<f32>(0.0);
    if (p.has_bias != 0u && col4 < N4) { bval = vec4<f32>(bias[col4]); }
    var acc0 = bval; var acc1 = bval; var acc2 = bval; var acc3 = bval;

    let a_r = tid >> 2u;
    let a_k4 = tid & 3u;
    let a_gr = wg_m + a_r;
    let a_valid = a_gr < m_total;
    let ow_oh = p.oh * p.ow;
    let a_b = select(0u, a_gr / ow_oh, a_valid);
    let a_rem = select(0u, a_gr % ow_oh, a_valid);
    let a_oy = a_rem / p.ow;
    let a_ox = a_rem % p.ow;

    let bk = tid >> 4u;
    let bn4 = tid & 15u;
    let gn4 = wg_n4 + bn4;

    for (var kt: u32 = 0u; kt < p.k_total; kt += 16u) {
        let k0 = kt + (a_k4 << 2u);
        if (a_valid) {
            As[tid] = vec4<f16>(
                load_im2col_pos(a_b, a_oy, a_ox, k0 + 0u),
                load_im2col_pos(a_b, a_oy, a_ox, k0 + 1u),
                load_im2col_pos(a_b, a_oy, a_ox, k0 + 2u),
                load_im2col_pos(a_b, a_oy, a_ox, k0 + 3u)
            );
        } else {
            As[tid] = vec4<f16>(0.0h);
        }
        let gk = kt + bk;
        if (gk < p.k_total && gn4 < N4) {
            Bs[tid] = B[gk * N4 + gn4];
        } else {
            Bs[tid] = vec4<f16>(0.0h);
        }
        workgroupBarrier();

        let rbase = ty * 4u;
        for (var kk4: u32 = 0u; kk4 < 4u; kk4++) {
            let a0 = vec4<f32>(As[(rbase + 0u) * 4u + kk4]);
            let a1 = vec4<f32>(As[(rbase + 1u) * 4u + kk4]);
            let a2 = vec4<f32>(As[(rbase + 2u) * 4u + kk4]);
            let a3 = vec4<f32>(As[(rbase + 3u) * 4u + kk4]);
            let kb = kk4 << 2u;
            let b0 = vec4<f32>(Bs[(kb + 0u) * 16u + tx]);
            let b1 = vec4<f32>(Bs[(kb + 1u) * 16u + tx]);
            let b2 = vec4<f32>(Bs[(kb + 2u) * 16u + tx]);
            let b3 = vec4<f32>(Bs[(kb + 3u) * 16u + tx]);

            acc0 = fma(vec4<f32>(a0.x), b0, fma(vec4<f32>(a0.y), b1, fma(vec4<f32>(a0.z), b2, fma(vec4<f32>(a0.w), b3, acc0))));
            acc1 = fma(vec4<f32>(a1.x), b0, fma(vec4<f32>(a1.y), b1, fma(vec4<f32>(a1.z), b2, fma(vec4<f32>(a1.w), b3, acc1))));
            acc2 = fma(vec4<f32>(a2.x), b0, fma(vec4<f32>(a2.y), b1, fma(vec4<f32>(a2.z), b2, fma(vec4<f32>(a2.w), b3, acc2))));
            acc3 = fma(vec4<f32>(a3.x), b0, fma(vec4<f32>(a3.y), b1, fma(vec4<f32>(a3.z), b2, fma(vec4<f32>(a3.w), b3, acc3))));
        }
        workgroupBarrier();
    }

    if (col4 < N4) {
        let r0 = wg_m + ty * 4u;
        if (r0 + 0u < m_total) { let o = (r0 + 0u) * N4 + col4; outp[o] = finalize_v4(acc0, o); }
        if (r0 + 1u < m_total) { let o = (r0 + 1u) * N4 + col4; outp[o] = finalize_v4(acc1, o); }
        if (r0 + 2u < m_total) { let o = (r0 + 2u) * N4 + col4; outp[o] = finalize_v4(acc2, o); }
        if (r0 + 3u < m_total) { let o = (r0 + 3u) * N4 + col4; outp[o] = finalize_v4(acc3, o); }
    }
}
)";
}

std::string wgsl_depthwise_3x3_s1p1(bool use_f16)
{
    if (!use_f16) {
        return wgsl_conv_generic_nhwc(false);
    }
    // Coalesced NHWC 2x4 Register Sliding-Window 3x3 Depthwise Conv (zero SMEM, zero barriers!).
    // lid.x (0..15) walks contiguous vec4<f16> channels = 128B full cache-line coalescing!
    // lid.y (0..3) walks 4 horizontal strips of width 4 (each thread outputs 2 rows x 4 cols).
    // Packed weights layout: [9, C4] of vec4<f16>, bias: [C4] of vec4<f16>.
    return R"(enable f16;
struct Params {
    n: u32,
    h: u32,
    w: u32,
    c4: u32,
    has_bias: u32,
    c_chunks: u32,
    pad0: u32,
    pad1: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> outp: array<vec4<f16>>;

@compute @workgroup_size(16, 4, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wgid.z / p.c_chunks;
    let c_chunk = wgid.z % p.c_chunks;
    let c4_idx = c_chunk * 16u + lid.x;
    let x0 = (wgid.x * 4u + lid.y) * 4u;
    let y0 = wgid.y * 2u;
    if (b >= p.n || c4_idx >= p.c4 || x0 >= p.w || y0 >= p.h) { return; }

    let w0 = vec4<f32>(weights[0u * p.c4 + c4_idx]);
    let w1 = vec4<f32>(weights[1u * p.c4 + c4_idx]);
    let w2 = vec4<f32>(weights[2u * p.c4 + c4_idx]);
    let w3 = vec4<f32>(weights[3u * p.c4 + c4_idx]);
    let w4 = vec4<f32>(weights[4u * p.c4 + c4_idx]);
    let w5 = vec4<f32>(weights[5u * p.c4 + c4_idx]);
    let w6 = vec4<f32>(weights[6u * p.c4 + c4_idx]);
    let w7 = vec4<f32>(weights[7u * p.c4 + c4_idx]);
    let w8 = vec4<f32>(weights[8u * p.c4 + c4_idx]);

    var bval = vec4<f32>(0.0);
    if (p.has_bias != 0u) {
        bval = vec4<f32>(bias[c4_idx]);
    }

    let has_y1 = (y0 + 1u) < p.h;
    let valid0 = y0 > 0u;
    let valid2 = (y0 + 1u) < p.h;
    let valid3 = has_y1 && ((y0 + 2u) < p.h);

    let row_stride = p.w * p.c4;
    let batch_base = b * p.h * row_stride + c4_idx;
    let r0_base = batch_base + select(0u, y0 - 1u, valid0) * row_stride;
    let r1_base = batch_base + y0 * row_stride;
    let r2_base = batch_base + min(y0 + 1u, p.h - 1u) * row_stride;
    let r3_base = batch_base + min(y0 + 2u, p.h - 1u) * row_stride;

    let z4 = vec4<f16>(0.0h);
    var l0 = z4; var l1 = z4; var l2 = z4; var l3 = z4;
    if (x0 > 0u) {
        let xl_off = (x0 - 1u) * p.c4;
        if (valid0) { l0 = inp[r0_base + xl_off]; }
        l1 = inp[r1_base + xl_off];
        if (valid2) { l2 = inp[r2_base + xl_off]; }
        if (valid3) { l3 = inp[r3_base + xl_off]; }
    }
    let xm_off = x0 * p.c4;
    var m0 = z4; if (valid0) { m0 = inp[r0_base + xm_off]; }
    var m1 = inp[r1_base + xm_off];
    var m2 = z4; if (valid2) { m2 = inp[r2_base + xm_off]; }
    var m3 = z4; if (valid3) { m3 = inp[r3_base + xm_off]; }

    let x_end = min(x0 + 4u, p.w);
    for (var x: u32 = x0; x < x_end; x++) {
        var r0 = z4; var r1 = z4; var r2 = z4; var r3 = z4;
        if (x + 1u < p.w) {
            let xr_off = (x + 1u) * p.c4;
            if (valid0) { r0 = inp[r0_base + xr_off]; }
            r1 = inp[r1_base + xr_off];
            if (valid2) { r2 = inp[r2_base + xr_off]; }
            if (valid3) { r3 = inp[r3_base + xr_off]; }
        }

        var acc0 = bval;
        acc0 = fma(vec4<f32>(l0), w0, acc0);
        acc0 = fma(vec4<f32>(m0), w1, acc0);
        acc0 = fma(vec4<f32>(r0), w2, acc0);
        acc0 = fma(vec4<f32>(l1), w3, acc0);
        acc0 = fma(vec4<f32>(m1), w4, acc0);
        acc0 = fma(vec4<f32>(r1), w5, acc0);
        acc0 = fma(vec4<f32>(l2), w6, acc0);
        acc0 = fma(vec4<f32>(m2), w7, acc0);
        acc0 = fma(vec4<f32>(r2), w8, acc0);
        outp[r1_base + x * p.c4] = vec4<f16>(acc0);

        if (has_y1) {
            var acc1 = bval;
            acc1 = fma(vec4<f32>(l1), w0, acc1);
            acc1 = fma(vec4<f32>(m1), w1, acc1);
            acc1 = fma(vec4<f32>(r1), w2, acc1);
            acc1 = fma(vec4<f32>(l2), w3, acc1);
            acc1 = fma(vec4<f32>(m2), w4, acc1);
            acc1 = fma(vec4<f32>(r2), w5, acc1);
            acc1 = fma(vec4<f32>(l3), w6, acc1);
            acc1 = fma(vec4<f32>(m3), w7, acc1);
            acc1 = fma(vec4<f32>(r3), w8, acc1);
            outp[r2_base + x * p.c4] = vec4<f16>(acc1);
        }

        l0 = m0; m0 = r0;
        l1 = m1; m1 = r1;
        l2 = m2; m2 = r2;
        l3 = m3; m3 = r3;
    }
}
)";
}

std::string wgsl_reglu_nhwc(bool use_f16)
{
    if (use_f16) {
        return R"(enable f16;
struct Params {
    spatial: u32,
    gate_c: u32,
    pad0: u32,
    pad1: u32,
    clamp_min: f32,
    clamp_max: f32,
    pad2: f32,
    pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.gate_c;
    let i = gid.x;
    if (i >= total) { return; }
    let s = i / p.gate_c;
    let c = i % p.gate_c;
    let base = s * (p.gate_c * 2u) + c;
    let g = f32(f16(clamp(f32(inp[base]), p.clamp_min, p.clamp_max)));
    let l = f32(inp[base + p.gate_c]);
    outp[i] = f16(g * l);
}
)";
    }
    return R"(struct Params {
    spatial: u32, gate_c: u32, pad0: u32, pad1: u32,
    clamp_min: f32, clamp_max: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
fn rd(idx: u32) -> f32 { let pr = unpack2x16float(inp[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }
fn r16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }
fn eval_one(i: u32, total: u32) -> f32 {
    if (i >= total) { return 0.0; }
    let s = i / p.gate_c;
    let c = i % p.gate_c;
    let base = s * (p.gate_c * 2u) + c;
    return r16(r16(clamp(rd(base), p.clamp_min, p.clamp_max)) * rd(base + p.gate_c));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.gate_c;
    let i0 = gid.x * 2u;
    if (i0 >= total) { return; }
    outp[gid.x] = pack2x16float(vec2<f32>(eval_one(i0, total), eval_one(i0 + 1u, total)));
}
)";
}

std::string wgsl_feature_update_nhwc(bool use_f16, bool emit_side_outputs)
{
    if (!use_f16) {
        if (emit_side_outputs) {
            return R"(struct Params { spatial: u32, channels: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read> history: array<u32>;
@group(0) @binding(3) var<storage, read> scale: array<u32>;
@group(0) @binding(4) var<storage, read_write> scaled_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> blended_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> feature_out: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch2 = p.channels >> 1u;
    let total = p.spatial * ch2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / ch2;
    let c2 = idx % ch2;
    let in_base = s * (ch2 * 3u) + c2;
    let a = unpack2x16float(inp[in_base]);
    let b = unpack2x16float(inp[in_base + ch2]);
    let cg = unpack2x16float(inp[in_base + ch2 * 2u]);
    let hist = unpack2x16float(history[idx]);
    let sc = unpack2x16float(scale[c2]);
    let sig_b = r16v(vec2<f32>(1.0) / (vec2<f32>(1.0) + exp(-b)));
    let sig_c = r16v(vec2<f32>(1.0) / (vec2<f32>(1.0) + exp(-cg)));
    let wb = r16v(sig_b * hist);
    let inv_g = r16v(vec2<f32>(1.0) - sig_b);
    let wa = r16v(inv_g * a);
    let blended = r16v(wb + wa);
    let gated = r16v(sig_c * blended);
    let scaled = r16v(gated * sc);
    let ps = pack2x16float(scaled);
    let pb = pack2x16float(blended);
    scaled_out[idx] = ps;
    blended_out[idx] = pb;
    let fbase = s * (ch2 * 2u) + c2;
    feature_out[fbase] = ps;
    feature_out[fbase + ch2] = pb;
}
)";
        }
        return R"(struct Params { spatial: u32, channels: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read> history: array<u32>;
@group(0) @binding(3) var<storage, read> scale: array<u32>;
@group(0) @binding(4) var<storage, read_write> feature_out: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch2 = p.channels >> 1u;
    let total = p.spatial * ch2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / ch2;
    let c2 = idx % ch2;
    let in_base = s * (ch2 * 3u) + c2;
    let a = unpack2x16float(inp[in_base]);
    let b = unpack2x16float(inp[in_base + ch2]);
    let cg = unpack2x16float(inp[in_base + ch2 * 2u]);
    let hist = unpack2x16float(history[idx]);
    let sc = unpack2x16float(scale[c2]);
    let sig_b = r16v(vec2<f32>(1.0) / (vec2<f32>(1.0) + exp(-b)));
    let sig_c = r16v(vec2<f32>(1.0) / (vec2<f32>(1.0) + exp(-cg)));
    let wb = r16v(sig_b * hist);
    let inv_g = r16v(vec2<f32>(1.0) - sig_b);
    let wa = r16v(inv_g * a);
    let blended = r16v(wb + wa);
    let gated = r16v(sig_c * blended);
    let scaled = r16v(gated * sc);
    let fbase = s * (ch2 * 2u) + c2;
    feature_out[fbase] = pack2x16float(scaled);
    feature_out[fbase + ch2] = pack2x16float(blended);
}
)";
    }

    if (emit_side_outputs) {
        return R"(enable f16;
struct Params { spatial: u32, channels: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<storage, read> history: array<f16>;
@group(0) @binding(3) var<storage, read> scale: array<f16>;
@group(0) @binding(4) var<storage, read_write> scaled_out: array<f16>;
@group(0) @binding(5) var<storage, read_write> blended_out: array<f16>;
@group(0) @binding(6) var<storage, read_write> feature_out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.channels;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.channels;
    let c = idx % p.channels;
    let in_base = s * (p.channels * 3u) + c;
    let a = f32(inp[in_base]);
    let b = f32(inp[in_base + p.channels]);
    let cg = f32(inp[in_base + p.channels * 2u]);
    let hist = f32(history[idx]);
    let sc = f32(scale[c]);

    let sig_b = f32(f16(1.0 / (1.0 + exp(-b))));
    let sig_c = f32(f16(1.0 / (1.0 + exp(-cg))));
    let wb = f32(f16(sig_b * hist));
    let inv_g = f32(f16(1.0 - sig_b));
    let wa = f32(f16(inv_g * a));
    let blended = f16(wb + wa);
    let gated = f32(f16(sig_c * f32(blended)));
    let scaled = f16(gated * sc);

    scaled_out[idx] = scaled;
    blended_out[idx] = blended;
    let fbase = s * (p.channels * 2u) + c;
    feature_out[fbase] = scaled;
    feature_out[fbase + p.channels] = blended;
}
)";
    }
    return R"(enable f16;
struct Params { spatial: u32, channels: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<storage, read> history: array<f16>;
@group(0) @binding(3) var<storage, read> scale: array<f16>;
@group(0) @binding(4) var<storage, read_write> feature_out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.channels;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.channels;
    let c = idx % p.channels;
    let in_base = s * (p.channels * 3u) + c;
    let a = f32(inp[in_base]);
    let b = f32(inp[in_base + p.channels]);
    let cg = f32(inp[in_base + p.channels * 2u]);
    let hist = f32(history[idx]);
    let sc = f32(scale[c]);

    let sig_b = f32(f16(1.0 / (1.0 + exp(-b))));
    let sig_c = f32(f16(1.0 / (1.0 + exp(-cg))));
    let wb = f32(f16(sig_b * hist));
    let inv_g = f32(f16(1.0 - sig_b));
    let wa = f32(f16(inv_g * a));
    let blended = f16(wb + wa);
    let gated = f32(f16(sig_c * f32(blended)));
    let scaled = f16(gated * sc);

    let fbase = s * (p.channels * 2u) + c;
    feature_out[fbase] = scaled;
    feature_out[fbase + p.channels] = blended;
}
)";
}

std::string wgsl_y0_tail_nhwc(bool use_f16)
{
    if (!use_f16) {
        return R"(struct Params {
    spatial: u32, q: u32, pad0: u32, pad1: u32,
    clip_min: f32, clip_max: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> latent: array<u32>;
@group(0) @binding(2) var<storage, read> prior: array<u32>;
@group(0) @binding(3) var<storage, read> quant_scale: array<u32>;
@group(0) @binding(4) var<storage, read> us0: array<u32>;
@group(0) @binding(5) var<storage, read> us1: array<u32>;
@group(0) @binding(6) var<storage, read_write> normalized: array<u32>;
@group(0) @binding(7) var<storage, read_write> clipped: array<u32>;
@group(0) @binding(8) var<storage, read_write> y_raw: array<u32>;
@group(0) @binding(9) var<storage, read_write> concat2: array<u32>;
@group(0) @binding(10) var<storage, read_write> concat3: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q2 = p.q >> 1u;
    let total = p.spatial * q2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / q2;
    let c2 = idx % q2;
    let norm_off = s * (q2 * 2u) + c2;
    let prior_off = s * (q2 * 4u) + c2;
    let c3_off = s * (q2 * 6u) + c2;

    let pr0 = unpack2x16float(prior[prior_off]);
    let pr1 = unpack2x16float(prior[prior_off + q2]);
    let pt0 = unpack2x16float(prior[prior_off + q2 * 2u]);
    let pt1 = unpack2x16float(prior[prior_off + q2 * 3u]);
    let l0 = unpack2x16float(latent[norm_off]);
    let l1 = unpack2x16float(latent[norm_off + q2]);
    let qs0 = unpack2x16float(quant_scale[norm_off]);
    let qs1 = unpack2x16float(quant_scale[norm_off + q2]);
    let u0 = unpack2x16float(us0[idx]);
    let u1 = unpack2x16float(us1[idx]);

    let cl0 = r16v(clamp(pr0, vec2<f32>(p.clip_min), vec2<f32>(p.clip_max)));
    let cl1 = r16v(clamp(pr1, vec2<f32>(p.clip_min), vec2<f32>(p.clip_max)));
    let inv0 = r16v(vec2<f32>(1.0) / cl0);
    let inv1 = r16v(vec2<f32>(1.0) / cl1);
    let n0 = r16v(l0 * inv0);
    let n1 = r16v(l1 * inv1);
    normalized[norm_off] = pack2x16float(n0);
    normalized[norm_off + q2] = pack2x16float(n1);
    clipped[norm_off] = pack2x16float(cl0);
    clipped[norm_off + q2] = pack2x16float(cl1);

    let sc0 = r16v(r16v(n0 - pt0) * qs0);
    let sc1 = r16v(r16v(n1 - pt1) * qs1);
    let rnd = r16v(round(r16v(sc0 + sc1)));
    y_raw[idx] = pack2x16float(rnd);

    let upd0 = r16v(r16v(rnd + pt0) * u0);
    let upd1 = r16v(r16v(rnd + pt1) * u1);
    let pu0 = pack2x16float(upd0);
    let pu1 = pack2x16float(upd1);
    concat2[norm_off] = pu0;
    concat2[norm_off + q2] = pu1;
    concat3[c3_off + q2 * 0u] = pu0;
    concat3[c3_off + q2 * 1u] = pu1;
    concat3[c3_off + q2 * 2u] = prior[prior_off + q2 * 0u];
    concat3[c3_off + q2 * 3u] = prior[prior_off + q2 * 1u];
    concat3[c3_off + q2 * 4u] = prior[prior_off + q2 * 2u];
    concat3[c3_off + q2 * 5u] = prior[prior_off + q2 * 3u];
}
)";
    }
    return R"(enable f16;
struct Params {
    spatial: u32, q: u32, pad0: u32, pad1: u32,
    clip_min: f32, clip_max: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> latent: array<f16>;
@group(0) @binding(2) var<storage, read> prior: array<f16>;
@group(0) @binding(3) var<storage, read> quant_scale: array<f16>;
@group(0) @binding(4) var<storage, read> us0: array<f16>;
@group(0) @binding(5) var<storage, read> us1: array<f16>;
@group(0) @binding(6) var<storage, read_write> normalized: array<f16>;
@group(0) @binding(7) var<storage, read_write> clipped: array<f16>;
@group(0) @binding(8) var<storage, read_write> y_raw: array<f16>;
@group(0) @binding(9) var<storage, read_write> concat2: array<f16>;
@group(0) @binding(10) var<storage, read_write> concat3: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.q;
    let c = idx % p.q;
    let norm_off = s * (p.q * 2u) + c;
    let prior_off = s * (p.q * 4u) + c;
    let c3_off = s * (p.q * 6u) + c;

    let pr0 = prior[prior_off];
    let pr1 = prior[prior_off + p.q];
    let pt0 = prior[prior_off + p.q * 2u];
    let pt1 = prior[prior_off + p.q * 3u];
    let l0 = latent[norm_off];
    let l1 = latent[norm_off + p.q];

    let cl0 = f16(clamp(f32(pr0), p.clip_min, p.clip_max));
    let cl1 = f16(clamp(f32(pr1), p.clip_min, p.clip_max));
    let inv0 = f16(1.0 / f32(cl0));
    let inv1 = f16(1.0 / f32(cl1));
    let n0 = f16(f32(l0) * f32(inv0));
    let n1 = f16(f32(l1) * f32(inv1));
    normalized[norm_off] = n0;
    normalized[norm_off + p.q] = n1;
    clipped[norm_off] = cl0;
    clipped[norm_off + p.q] = cl1;

    let d0 = f16(f32(n0) - f32(pt0));
    let d1 = f16(f32(n1) - f32(pt1));
    let sc0 = f16(f32(d0) * f32(quant_scale[norm_off]));
    let sc1 = f16(f32(d1) * f32(quant_scale[norm_off + p.q]));
    let rnd = f16(round(f32(f16(f32(sc0) + f32(sc1)))));
    y_raw[idx] = rnd;

    let upd0 = f16(f32(f16(f32(rnd) + f32(pt0))) * f32(us0[idx]));
    let upd1 = f16(f32(f16(f32(rnd) + f32(pt1))) * f32(us1[idx]));
    concat2[norm_off] = upd0;
    concat2[norm_off + p.q] = upd1;
    concat3[c3_off + p.q * 0u] = upd0;
    concat3[c3_off + p.q * 1u] = upd1;
    concat3[c3_off + p.q * 2u] = pr0;
    concat3[c3_off + p.q * 3u] = pr1;
    concat3[c3_off + p.q * 4u] = pt0;
    concat3[c3_off + p.q * 5u] = pt1;
}
)";
}

std::string wgsl_y1_tail_nhwc(bool use_f16)
{
    if (!use_f16) {
        return R"(struct Params { spatial: u32, q: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> normalized: array<u32>;
@group(0) @binding(2) var<storage, read> prior: array<u32>;
@group(0) @binding(3) var<storage, read> quant_scale: array<u32>;
@group(0) @binding(4) var<storage, read> us0: array<u32>;
@group(0) @binding(5) var<storage, read> us1: array<u32>;
@group(0) @binding(6) var<storage, read> concat2: array<u32>;
@group(0) @binding(7) var<storage, read> clipped: array<u32>;
@group(0) @binding(8) var<storage, read_write> y_raw: array<u32>;
@group(0) @binding(9) var<storage, read_write> output: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q2 = p.q >> 1u;
    let total = p.spatial * q2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / q2;
    let c2 = idx % q2;
    let base = s * (q2 * 2u) + c2;

    let n0 = unpack2x16float(normalized[base]);
    let n1 = unpack2x16float(normalized[base + q2]);
    let p0 = unpack2x16float(prior[base]);
    let p1 = unpack2x16float(prior[base + q2]);
    let qs0 = unpack2x16float(quant_scale[base]);
    let qs1 = unpack2x16float(quant_scale[base + q2]);
    let u0 = unpack2x16float(us0[idx]);
    let u1 = unpack2x16float(us1[idx]);
    let c2_0 = unpack2x16float(concat2[base]);
    let c2_1 = unpack2x16float(concat2[base + q2]);
    let cl0 = unpack2x16float(clipped[base]);
    let cl1 = unpack2x16float(clipped[base + q2]);

    let sc0 = r16v(r16v(n0 - p0) * qs0);
    let sc1 = r16v(r16v(n1 - p1) * qs1);
    let rnd = r16v(round(r16v(sc0 + sc1)));
    y_raw[idx] = pack2x16float(rnd);

    let upd0 = r16v(r16v(rnd + p0) * u0);
    let upd1 = r16v(r16v(rnd + p1) * u1);
    output[base] = pack2x16float(r16v(r16v(c2_0 + upd0) * cl0));
    output[base + q2] = pack2x16float(r16v(r16v(c2_1 + upd1) * cl1));
}
)";
    }
    return R"(enable f16;
struct Params { spatial: u32, q: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> normalized: array<f16>;
@group(0) @binding(2) var<storage, read> prior: array<f16>;
@group(0) @binding(3) var<storage, read> quant_scale: array<f16>;
@group(0) @binding(4) var<storage, read> us0: array<f16>;
@group(0) @binding(5) var<storage, read> us1: array<f16>;
@group(0) @binding(6) var<storage, read> concat2: array<f16>;
@group(0) @binding(7) var<storage, read> clipped: array<f16>;
@group(0) @binding(8) var<storage, read_write> y_raw: array<f16>;
@group(0) @binding(9) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.q;
    let c = idx % p.q;
    let base = s * (p.q * 2u) + c;

    let n0 = normalized[base];
    let n1 = normalized[base + p.q];
    let p0 = prior[base];
    let p1 = prior[base + p.q];
    let sc0 = f16(f32(f16(f32(n0) - f32(p0))) * f32(quant_scale[base]));
    let sc1 = f16(f32(f16(f32(n1) - f32(p1))) * f32(quant_scale[base + p.q]));
    let rnd = f16(round(f32(f16(f32(sc0) + f32(sc1)))));
    y_raw[idx] = rnd;

    let upd0 = f16(f32(f16(f32(rnd) + f32(p0))) * f32(us0[idx]));
    let upd1 = f16(f32(f16(f32(rnd) + f32(p1))) * f32(us1[idx]));
    let add0 = f16(f32(concat2[base]) + f32(upd0));
    let add1 = f16(f32(concat2[base + p.q]) + f32(upd1));
    output[base] = f16(f32(add0) * f32(clipped[base]));
    output[base + p.q] = f16(f32(add1) * f32(clipped[base + p.q]));
}
)";
}

std::string wgsl_dec_y0_tail_nhwc(bool use_f16)
{
    if (!use_f16) {
        return R"(struct Params {
    spatial: u32, q: u32, pad0: u32, pad1: u32,
    clip_min: f32, clip_max: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prior: array<u32>;
@group(0) @binding(2) var<storage, read> y_raw: array<u32>;
@group(0) @binding(3) var<storage, read> us0: array<u32>;
@group(0) @binding(4) var<storage, read> us1: array<u32>;
@group(0) @binding(5) var<storage, read_write> clipped: array<u32>;
@group(0) @binding(6) var<storage, read_write> concat2: array<u32>;
@group(0) @binding(7) var<storage, read_write> concat3: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q2 = p.q >> 1u;
    let total = p.spatial * q2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / q2;
    let c2 = idx % q2;
    let norm_off = s * (q2 * 2u) + c2;
    let prior_off = s * (q2 * 4u) + c2;
    let c3_off = s * (q2 * 6u) + c2;

    let pr0 = unpack2x16float(prior[prior_off]);
    let pr1 = unpack2x16float(prior[prior_off + q2]);
    let pt0 = unpack2x16float(prior[prior_off + q2 * 2u]);
    let pt1 = unpack2x16float(prior[prior_off + q2 * 3u]);
    let y = unpack2x16float(y_raw[idx]);
    let u0 = unpack2x16float(us0[idx]);
    let u1 = unpack2x16float(us1[idx]);

    clipped[norm_off] = pack2x16float(r16v(clamp(pr0, vec2<f32>(p.clip_min), vec2<f32>(p.clip_max))));
    clipped[norm_off + q2] = pack2x16float(r16v(clamp(pr1, vec2<f32>(p.clip_min), vec2<f32>(p.clip_max))));

    let pu0 = pack2x16float(r16v(r16v(y + pt0) * u0));
    let pu1 = pack2x16float(r16v(r16v(y + pt1) * u1));
    concat2[norm_off] = pu0;
    concat2[norm_off + q2] = pu1;
    concat3[c3_off + q2 * 0u] = pu0;
    concat3[c3_off + q2 * 1u] = pu1;
    concat3[c3_off + q2 * 2u] = prior[prior_off + q2 * 0u];
    concat3[c3_off + q2 * 3u] = prior[prior_off + q2 * 1u];
    concat3[c3_off + q2 * 4u] = prior[prior_off + q2 * 2u];
    concat3[c3_off + q2 * 5u] = prior[prior_off + q2 * 3u];
}
)";
    }
    return R"(enable f16;
struct Params {
    spatial: u32, q: u32, pad0: u32, pad1: u32,
    clip_min: f32, clip_max: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prior: array<f16>;
@group(0) @binding(2) var<storage, read> y_raw: array<f16>;
@group(0) @binding(3) var<storage, read> us0: array<f16>;
@group(0) @binding(4) var<storage, read> us1: array<f16>;
@group(0) @binding(5) var<storage, read_write> clipped: array<f16>;
@group(0) @binding(6) var<storage, read_write> concat2: array<f16>;
@group(0) @binding(7) var<storage, read_write> concat3: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.q;
    let c = idx % p.q;
    let norm_off = s * (p.q * 2u) + c;
    let prior_off = s * (p.q * 4u) + c;
    let c3_off = s * (p.q * 6u) + c;

    let pr0 = prior[prior_off];
    let pr1 = prior[prior_off + p.q];
    let pt0 = prior[prior_off + p.q * 2u];
    let pt1 = prior[prior_off + p.q * 3u];
    let y = y_raw[idx];

    clipped[norm_off] = f16(clamp(f32(pr0), p.clip_min, p.clip_max));
    clipped[norm_off + p.q] = f16(clamp(f32(pr1), p.clip_min, p.clip_max));

    let upd0 = f16(f32(f16(f32(y) + f32(pt0))) * f32(us0[idx]));
    let upd1 = f16(f32(f16(f32(y) + f32(pt1))) * f32(us1[idx]));
    concat2[norm_off] = upd0;
    concat2[norm_off + p.q] = upd1;
    concat3[c3_off + p.q * 0u] = upd0;
    concat3[c3_off + p.q * 1u] = upd1;
    concat3[c3_off + p.q * 2u] = pr0;
    concat3[c3_off + p.q * 3u] = pr1;
    concat3[c3_off + p.q * 4u] = pt0;
    concat3[c3_off + p.q * 5u] = pt1;
}
)";
}

std::string wgsl_dec_y1_tail_nhwc(bool use_f16)
{
    if (!use_f16) {
        return R"(struct Params { spatial: u32, q: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prior: array<u32>;
@group(0) @binding(2) var<storage, read> y_raw: array<u32>;
@group(0) @binding(3) var<storage, read> us0: array<u32>;
@group(0) @binding(4) var<storage, read> us1: array<u32>;
@group(0) @binding(5) var<storage, read> concat2: array<u32>;
@group(0) @binding(6) var<storage, read> clipped: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<u32>;

fn r16v(v: vec2<f32>) -> vec2<f32> { return unpack2x16float(pack2x16float(v)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q2 = p.q >> 1u;
    let total = p.spatial * q2;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / q2;
    let c2 = idx % q2;
    let base = s * (q2 * 2u) + c2;

    let p0 = unpack2x16float(prior[base]);
    let p1 = unpack2x16float(prior[base + q2]);
    let y = unpack2x16float(y_raw[idx]);
    let u0 = unpack2x16float(us0[idx]);
    let u1 = unpack2x16float(us1[idx]);
    let c2_0 = unpack2x16float(concat2[base]);
    let c2_1 = unpack2x16float(concat2[base + q2]);
    let cl0 = unpack2x16float(clipped[base]);
    let cl1 = unpack2x16float(clipped[base + q2]);

    let upd0 = r16v(r16v(y + p0) * u0);
    let upd1 = r16v(r16v(y + p1) * u1);
    output[base] = pack2x16float(r16v(r16v(c2_0 + upd0) * cl0));
    output[base + q2] = pack2x16float(r16v(r16v(c2_1 + upd1) * cl1));
}
)";
    }
    return R"(enable f16;
struct Params { spatial: u32, q: u32, pad0: u32, pad1: u32, };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prior: array<f16>;
@group(0) @binding(2) var<storage, read> y_raw: array<f16>;
@group(0) @binding(3) var<storage, read> us0: array<f16>;
@group(0) @binding(4) var<storage, read> us1: array<f16>;
@group(0) @binding(5) var<storage, read> concat2: array<f16>;
@group(0) @binding(6) var<storage, read> clipped: array<f16>;
@group(0) @binding(7) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.spatial * p.q;
    let idx = gid.x;
    if (idx >= total) { return; }
    let s = idx / p.q;
    let c = idx % p.q;
    let base = s * (p.q * 2u) + c;

    let y = y_raw[idx];
    let upd0 = f16(f32(f16(f32(y) + f32(prior[base]))) * f32(us0[idx]));
    let upd1 = f16(f32(f16(f32(y) + f32(prior[base + p.q]))) * f32(us1[idx]));
    let add0 = f16(f32(concat2[base]) + f32(upd0));
    let add1 = f16(f32(concat2[base + p.q]) + f32(upd1));
    output[base] = f16(f32(add0) * f32(clipped[base]));
    output[base + p.q] = f16(f32(add1) * f32(clipped[base + p.q]));
}
)";
}

std::string wgsl_yuv420_to_nhwc(bool use_f16)
{
    if (use_f16) {
        // Writes 2 pixels = 6 f16 = 3 vec2<f16> (4-byte aligned, zero 16-bit RMW!).
        return R"(enable f16;
struct Params {
    width: u32, height: u32, model_width: u32, model_height: u32,
    rotated: u32, pad0: u32, pad1: u32, pad2: u32,
    scale: f32, pad3: f32, pad4: f32, pad5: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> yuv_bytes: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f16>>;

fn read_byte(byte_off: u32) -> u32 {
    return (yuv_bytes[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xFFu;
}
fn sample_px(x: u32, y: u32) -> vec3<f16> {
    let active_w = select(p.width, p.height, p.rotated != 0u);
    let active_h = select(p.height, p.width, p.rotated != 0u);
    let edge_y = min(y, active_h - 1u);
    let edge_x = min(x, active_w - 1u);
    let src_y = select(edge_y, edge_x, p.rotated != 0u);
    let src_x = select(edge_x, edge_y, p.rotated != 0u);
    let y_size = p.width * p.height;
    let uv_size = y_size >> 2u;
    let uv_off = (src_y >> 1u) * (p.width >> 1u) + (src_x >> 1u);
    return vec3<f16>(
        f16(f32(read_byte(src_y * p.width + src_x)) * p.scale),
        f16(f32(read_byte(y_size + uv_off)) * p.scale),
        f16(f32(read_byte(y_size + uv_size + uv_off)) * p.scale)
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x2 = gid.x;
    let y = gid.y;
    let mw2 = p.model_width >> 1u;
    if (x2 >= mw2 || y >= p.model_height) { return; }
    let p0 = sample_px(x2 * 2u, y);
    let p1 = sample_px(x2 * 2u + 1u, y);
    let out_v2 = (y * mw2 + x2) * 3u;
    dst[out_v2 + 0u] = vec2<f16>(p0.x, p0.y);
    dst[out_v2 + 1u] = vec2<f16>(p0.z, p1.x);
    dst[out_v2 + 2u] = vec2<f16>(p1.y, p1.z);
}
)";
    }
    // Fallback writes 2 pixels (6 f16 = 3 u32) per thread.
    return R"(struct Params {
    width: u32, height: u32, model_width: u32, model_height: u32,
    rotated: u32, pad0: u32, pad1: u32, pad2: u32,
    scale: f32, pad3: f32, pad4: f32, pad5: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> yuv_bytes: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn read_byte(byte_off: u32) -> u32 {
    return (yuv_bytes[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xFFu;
}
fn sample_px(x: u32, y: u32) -> vec3<f32> {
    let active_w = select(p.width, p.height, p.rotated != 0u);
    let active_h = select(p.height, p.width, p.rotated != 0u);
    let edge_y = min(y, active_h - 1u);
    let edge_x = min(x, active_w - 1u);
    let src_y = select(edge_y, edge_x, p.rotated != 0u);
    let src_x = select(edge_x, edge_y, p.rotated != 0u);
    let y_size = p.width * p.height;
    let uv_size = y_size >> 2u;
    let uv_off = (src_y >> 1u) * (p.width >> 1u) + (src_x >> 1u);
    return vec3<f32>(
        f32(read_byte(src_y * p.width + src_x)) * p.scale,
        f32(read_byte(y_size + uv_off)) * p.scale,
        f32(read_byte(y_size + uv_size + uv_off)) * p.scale
    );
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x2 = gid.x;
    let y = gid.y;
    let mw2 = p.model_width >> 1u;
    if (x2 >= mw2 || y >= p.model_height) { return; }
    let p0 = sample_px(x2 * 2u, y);
    let p1 = sample_px(x2 * 2u + 1u, y);
    let out_u32 = (y * mw2 + x2) * 3u;
    dst[out_u32 + 0u] = pack2x16float(vec2<f32>(p0.x, p0.y));
    dst[out_u32 + 1u] = pack2x16float(vec2<f32>(p0.z, p1.x));
    dst[out_u32 + 2u] = pack2x16float(vec2<f32>(p1.y, p1.z));
}
)";
}

std::string wgsl_nhwc_to_yuv420(bool use_f16, bool fused_d2s8)
{
    // Writes flat packed YUV420 byte array as aligned u32 words (4 bytes/thread).
    // Works for any even width & height without byte-level write races!
    std::string header = use_f16
        ? "enable f16;\n@group(0) @binding(1) var<storage, read> src: array<f16>;\n"
          "fn rd(idx: u32) -> f32 { return f32(src[idx]); }\n"
        : "@group(0) @binding(1) var<storage, read> src: array<u32>;\n"
          "fn rd(idx: u32) -> f32 { let pr = unpack2x16float(src[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }\n";

    std::string sampler = fused_d2s8
        ? R"(fn model_sample(ch: u32, y: u32, x: u32) -> f32 {
    let my = select(y, x, p.rotated != 0u) + p.pad_top;
    let mx = select(x, y, p.rotated != 0u) + p.pad_left;
    let iy = my >> 3u;
    let dy = my & 7u;
    let ix = mx >> 3u;
    let dx = mx & 7u;
    let iw = p.model_width >> 3u;
    let ic = (ch * 64u) + dy * 8u + dx;
    return clamp(rd((iy * iw + ix) * 192u + ic) * p.inv_pixel_range, 0.0, 1.0);
}
)"
        : R"(fn model_sample(ch: u32, y: u32, x: u32) -> f32 {
    let my = select(y, x, p.rotated != 0u) + p.pad_top;
    let mx = select(x, y, p.rotated != 0u) + p.pad_left;
    return rd((my * p.model_width + mx) * 3u + ch) * p.inv_pixel_range;
}
)";

    return header + R"(struct Params {
    width: u32, height: u32, model_width: u32, model_height: u32,
    pad_left: u32, pad_top: u32, rotated: u32, total_bytes: u32,
    inv_pixel_range: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> out_bytes: array<u32>;

fn quantize_u8(v: f32) -> u32 {
    let clamped = clamp(v, 0.0, 1.0);
    return u32(round(clamped * 255.0));
}
)" + sampler + R"(
fn compute_byte(byte_idx: u32) -> u32 {
    if (byte_idx >= p.total_bytes) { return 0u; }
    let y_count = p.width * p.height;
    let uv_count = y_count >> 2u;
    if (byte_idx < y_count) {
        let y = byte_idx / p.width;
        let x = byte_idx % p.width;
        return quantize_u8(model_sample(0u, y, x));
    }
    let uv_w = p.width >> 1u;
    let ch = select(1u, 2u, byte_idx >= y_count + uv_count);
    let rel = select(byte_idx - y_count, byte_idx - y_count - uv_count, ch == 2u);
    let uy = rel / uv_w;
    let ux = rel % uv_w;
    let y0 = uy << 1u;
    let x0 = ux << 1u;
    let sum = model_sample(ch, y0, x0) + model_sample(ch, y0, x0 + 1u) +
              model_sample(ch, y0 + 1u, x0) + model_sample(ch, y0 + 1u, x0 + 1u);
    return quantize_u8(sum * 0.25);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    let b0 = word_idx * 4u;
    if (b0 >= p.total_bytes) { return; }
    let v0 = compute_byte(b0 + 0u);
    let v1 = compute_byte(b0 + 1u);
    let v2 = compute_byte(b0 + 2u);
    let v3 = compute_byte(b0 + 3u);
    out_bytes[word_idx] = v0 | (v1 << 8u) | (v2 << 16u) | (v3 << 24u);
}
)";
}

std::string wgsl_render_yuv444_to_rgba(bool use_f16, bool fused_d2s8)
{
    // Zero-readback full-screen triangle render pass:
    // Reads YUV444 directly from x_hat (or pre-DepthToSpace(8) C=192 feature map!),
    // applies crop/un-transpose + BT.709/BT.601 YUV->RGB conversion directly to Canvas!
    std::string header = use_f16
        ? "enable f16;\n@group(0) @binding(1) var<storage, read> src: array<f16>;\n"
          "fn rd(idx: u32) -> f32 { return f32(src[idx]); }\n"
        : "@group(0) @binding(1) var<storage, read> src: array<u32>;\n"
          "fn rd(idx: u32) -> f32 { let pr = unpack2x16float(src[idx >> 1u]); return select(pr.x, pr.y, (idx & 1u) != 0u); }\n";

    std::string sample_yuv = fused_d2s8
        ? R"(fn read_yuv(x: u32, y: u32) -> vec3<f32> {
    let my = select(y, x, p.rotated != 0u) + p.pad_top;
    let mx = select(x, y, p.rotated != 0u) + p.pad_left;
    let iy = my >> 3u;
    let dy = my & 7u;
    let ix = mx >> 3u;
    let dx = mx & 7u;
    let iw = p.model_width >> 3u;
    let base = (iy * iw + ix) * 192u + dy * 8u + dx;
    return vec3<f32>(rd(base), rd(base + 64u), rd(base + 128u)) * p.inv_pixel_range;
}
)"
        : R"(fn read_yuv(x: u32, y: u32) -> vec3<f32> {
    let my = select(y, x, p.rotated != 0u) + p.pad_top;
    let mx = select(x, y, p.rotated != 0u) + p.pad_left;
    let base = (my * p.model_width + mx) * 3u;
    return vec3<f32>(rd(base), rd(base + 1u), rd(base + 2u)) * p.inv_pixel_range;
}
)";

    return header + R"(struct RenderParams {
    width: u32, height: u32, model_width: u32, model_height: u32,
    pad_left: u32, pad_top: u32, rotated: u32, color_matrix: u32, // 0=BT.709 full, 1=BT.601 full, 2=BT.709 limited
    inv_pixel_range: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: RenderParams;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0)
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}
)" + sample_yuv + R"(
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let px = min(u32(frag_coord.x), p.width - 1u);
    let py = min(u32(frag_coord.y), p.height - 1u);
    let yuv = clamp(read_yuv(px, py), vec3<f32>(0.0), vec3<f32>(1.0));
    var r: f32; var g: f32; var b: f32;
    if (p.color_matrix == 1u) {
        // BT.601 full-range
        let Y = yuv.x;
        let U = yuv.y - 0.5;
        let V = yuv.z - 0.5;
        r = Y + 1.402 * V;
        g = Y - 0.344136 * U - 0.714136 * V;
        b = Y + 1.772 * U;
    } else if (p.color_matrix == 2u) {
        // BT.709 limited-range (16..235 / 16..240)
        let Y = (yuv.x - 16.0 / 255.0) * (255.0 / 219.0);
        let U = (yuv.y - 128.0 / 255.0) * (255.0 / 224.0);
        let V = (yuv.z - 128.0 / 255.0) * (255.0 / 224.0);
        r = Y + 1.5748 * V;
        g = Y - 0.187324 * U - 0.468124 * V;
        b = Y + 1.8556 * U;
    } else {
        // BT.709 full-range (default for MLVC RGB<->YUV)
        let Y = yuv.x;
        let U = yuv.y - 0.5;
        let V = yuv.z - 0.5;
        r = Y + 1.5748 * V;
        g = Y - 0.187324 * U - 0.468124 * V;
        b = Y + 1.8556 * U;
    }
    return vec4<f32>(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
)";
}

}  // namespace mlvc::webgpu_backend
