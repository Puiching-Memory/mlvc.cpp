#include "backends/webgpu/webgpu_surface.hpp"
#include "backends/webgpu/wgpu_context.hpp"
#include "core/assets.hpp"
#include "core/bitstream.hpp"
#include "core/entropy.hpp"
#include "core/model.hpp"
#include "core/tensor.hpp"
#include "core/yuv.hpp"
#include "runtime/backend.hpp"

#include <emscripten.h>
#include <emscripten/em_js.h>
#include <webgpu/webgpu.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mlvc;
using namespace mlvc::webgpu_backend;

class BrowserMemoryAssetReader final : public ModelAssetReader {
public:
    void clear() { files_.clear(); }

    void put(std::string relative_path, const std::uint8_t* data, std::size_t size)
    {
        std::vector<std::byte> buf(size);
        if (size > 0U && data)
            std::memcpy(buf.data(), data, size);
        files_[std::move(relative_path)] = std::move(buf);
    }

    std::string location_label() const override { return "browser-memory"; }

    bool exists(const std::filesystem::path& relative_path) const override
    {
        return files_.find(relative_path.generic_string()) != files_.end();
    }

    std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const override
    {
        const auto it = files_.find(relative_path.generic_string());
        if (it == files_.end()) {
            throw std::runtime_error(
                "browser asset not loaded: " + relative_path.generic_string());
        }
        return it->second;
    }

    AssetView read_view(const std::filesystem::path& relative_path) const override
    {
        const auto it = files_.find(relative_path.generic_string());
        if (it == files_.end()) {
            throw std::runtime_error(
                "browser asset not loaded: " + relative_path.generic_string());
        }
        return AssetView::borrowed(it->second);
    }

private:
    std::map<std::string, std::vector<std::byte>> files_;
};

struct WebDecoderState {
    ModelConfig config;
    std::unique_ptr<EntropyCodec> entropy;
    BitstreamDecoder bitstream_decoder;
    std::unique_ptr<InferenceBackend> backend;
    WebGpuSurfaceBackend* surface_backend = nullptr;

    UniqueSurface surface;
    WGPUTextureFormat surface_format = WGPUTextureFormat_BGRA8Unorm;
    std::uint32_t color_matrix = 0U;
    bool surface_configured = false;
    int configured_width = 0;
    int configured_height = 0;
    FramePadding padding{};

    bool has_active_ref = false;
    int last_frame_idx = -1;

    // Reusable host buffers for rANS decoded FP16 latents.
    Tensor z_tensor;
    Tensor y0_tensor;
    Tensor y1_tensor;
};

struct DecodeMetrics {
    std::int32_t width;
    std::int32_t height;
    std::int32_t frame_idx;
    std::int32_t q_index;
    std::int32_t frame_type;
    std::int32_t fused_d2s8;
    std::int32_t uses_f16;
    std::int32_t dispatches;
    double rans_ms;
    double submit_ms;
};

WGPUInstance g_instance = nullptr;
WGPUAdapter g_adapter = nullptr;
WGPUDevice g_device = nullptr;
WGPUQueue g_queue = nullptr;
std::string g_last_error;
std::string g_gpu_info_json = "{}";
auto g_assets = std::make_shared<BrowserMemoryAssetReader>();
std::unique_ptr<WebDecoderState> g_decoder;

void set_error(const std::string& msg)
{
    g_last_error = msg;
}

EM_JS(void, js_notify_gpu_ready, (int ok, const char* msg_ptr), {
    const msg = UTF8ToString(msg_ptr);
    if (typeof globalThis.__mlvcOnGpuReady === 'function') {
        globalThis.__mlvcOnGpuReady(ok !== 0, msg);
    }
});

EM_JS(void, js_notify_gpu_flushed, (int ok), {
    if (typeof globalThis.__mlvcOnGpuFlush === 'function') {
        const fn = globalThis.__mlvcOnGpuFlush;
        globalThis.__mlvcOnGpuFlush = null;
        fn(ok !== 0);
    }
});

EM_JS(void, js_resize_canvas, (const char* selector_ptr, int w, int h), {
    const sel = UTF8ToString(selector_ptr);
    if (typeof globalThis.__mlvcResizeCanvas === 'function') {
        globalThis.__mlvcResizeCanvas(sel, w, h);
    } else if (typeof document !== 'undefined') {
        const el = document.querySelector(sel);
        if (el) {
            el.width = w;
            el.height = h;
        }
    }
});

std::string g_canvas_selector;

void configure_surface_for_frame(WebDecoderState& st, const FrameData& fd)
{
    const int dw = fd.display_width();
    const int dh = fd.display_height();
    if (st.surface_configured && st.configured_width == dw && st.configured_height == dh)
        return;

    js_resize_canvas(g_canvas_selector.c_str(), dw, dh);

    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(st.surface.get(), g_adapter, &caps);
    WGPUTextureFormat fmt = WGPUTextureFormat_BGRA8Unorm;
    if (caps.formatCount > 0U && caps.formats) {
        fmt = caps.formats[0];
        for (std::size_t i = 0; i < caps.formatCount; ++i) {
            if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm ||
                caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
                fmt = caps.formats[i];
                break;
            }
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    st.surface_format = fmt;

    WGPUSurfaceConfiguration sc = WGPU_SURFACE_CONFIGURATION_INIT;
    sc.device = g_device;
    sc.format = fmt;
    sc.usage = WGPUTextureUsage_RenderAttachment;
    sc.width = static_cast<std::uint32_t>(dw);
    sc.height = static_cast<std::uint32_t>(dh);
    sc.alphaMode = WGPUCompositeAlphaMode_Opaque;
    sc.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(st.surface.get(), &sc);

    st.padding = frame_padding(dw, dh, st.config.model_width, st.config.model_height);

    SurfaceRenderConfig rc;
    rc.width = dw;
    rc.height = dh;
    rc.model_width = st.config.model_width;
    rc.model_height = st.config.model_height;
    rc.pixel_range = st.config.pixel_range;
    rc.padding = st.padding;
    rc.target_format = fmt;
    rc.color_matrix = st.color_matrix;
    st.surface_backend->configure_surface_output(rc);

    st.configured_width = dw;
    st.configured_height = dh;
    st.surface_configured = true;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
const char* mlvc_web_last_error()
{
    return g_last_error.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char* mlvc_web_gpu_info()
{
    return g_gpu_info_json.c_str();
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_init_gpu()
{
    g_last_error.clear();
    if (!g_instance)
        g_instance = wgpuCreateInstance(nullptr);
    if (!g_instance) {
        set_error("wgpuCreateInstance failed");
        js_notify_gpu_ready(0, g_last_error.c_str());
        return;
    }

    WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                      WGPUStringView message, void*, void*) {
        if (status != WGPURequestAdapterStatus_Success || !adapter) {
            set_error("requestAdapter failed: " + from_wgpu_string(message));
            js_notify_gpu_ready(0, g_last_error.c_str());
            return;
        }
        g_adapter = adapter;

        std::vector<WGPUFeatureName> feats;
        if (wgpuAdapterHasFeature(adapter, WGPUFeatureName_ShaderF16) == WGPU_TRUE)
            feats.push_back(WGPUFeatureName_ShaderF16);
        if (wgpuAdapterHasFeature(adapter, WGPUFeatureName_Subgroups) == WGPU_TRUE)
            feats.push_back(WGPUFeatureName_Subgroups);

        WGPULimits al = WGPU_LIMITS_INIT;
        WGPULimits rl = WGPU_LIMITS_INIT;
        if (wgpuAdapterGetLimits(adapter, &al) == WGPUStatus_Success) {
            rl.maxStorageBufferBindingSize = al.maxStorageBufferBindingSize;
            rl.maxBufferSize = al.maxBufferSize;
            rl.maxStorageBuffersPerShaderStage = al.maxStorageBuffersPerShaderStage;
            rl.maxComputeWorkgroupStorageSize = al.maxComputeWorkgroupStorageSize;
            rl.maxComputeInvocationsPerWorkgroup = al.maxComputeInvocationsPerWorkgroup;
            rl.maxComputeWorkgroupsPerDimension = al.maxComputeWorkgroupsPerDimension;
        }

        WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
        ddesc.label = to_wgpu_string("mlvc.browser.device");
        ddesc.requiredFeatureCount = feats.size();
        ddesc.requiredFeatures = feats.empty() ? nullptr : feats.data();
        ddesc.requiredLimits = &rl;
        ddesc.uncapturedErrorCallbackInfo.callback = [](
            WGPUDevice const*, WGPUErrorType type, WGPUStringView msg, void*, void*) {
            set_error("WebGPU uncaptured error (" + std::to_string(static_cast<int>(type)) +
                      "): " + from_wgpu_string(msg));
        };

        WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        dcb.mode = WGPUCallbackMode_AllowSpontaneous;
        dcb.callback = [](WGPURequestDeviceStatus dstatus, WGPUDevice device,
                          WGPUStringView dmsg, void*, void*) {
            if (dstatus != WGPURequestDeviceStatus_Success || !device) {
                set_error("requestDevice failed: " + from_wgpu_string(dmsg));
                js_notify_gpu_ready(0, g_last_error.c_str());
                return;
            }
            g_device = device;
            g_queue = wgpuDeviceGetQueue(device);
            auto ctx = WgpuContext::from_external(g_instance, g_adapter, g_device, g_queue);
            const auto& c = ctx->caps();
            std::ostringstream oss;
            oss << "{\"adapter\":\"" << c.adapter_name
                << "\",\"backend\":\"" << c.backend_name
                << "\",\"f16\":" << (c.has_shader_f16 ? "true" : "false")
                << ",\"subgroups\":" << (c.has_subgroups ? "true" : "false")
                << ",\"maxStorageBuffers\":" << c.max_storage_buffers_per_stage
                << "}";
            g_gpu_info_json = oss.str();
            WgpuContext::set_external_context(std::move(ctx));
            js_notify_gpu_ready(1, g_gpu_info_json.c_str());
        };
        wgpuAdapterRequestDevice(adapter, &ddesc, dcb);
    };
    wgpuInstanceRequestAdapter(g_instance, &opts, acb);
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_flush_gpu()
{
    if (!g_queue) {
        js_notify_gpu_flushed(0);
        return;
    }
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void*, void*) {
        js_notify_gpu_flushed(status == WGPUQueueWorkDoneStatus_Success ? 1 : 0);
    };
    wgpuQueueOnSubmittedWorkDone(g_queue, cb);
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_asset_clear()
{
    g_assets->clear();
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_asset_put(const char* rel_path, const std::uint8_t* data, std::uint32_t size)
{
    if (!rel_path)
        return;
    g_assets->put(std::string(rel_path), data, static_cast<std::size_t>(size));
}

EMSCRIPTEN_KEEPALIVE
int mlvc_web_decoder_create(const char* canvas_selector, std::uint32_t color_matrix)
{
    try {
        g_last_error.clear();
        g_decoder.reset();
        if (!g_instance || !g_device)
            throw std::runtime_error("WebGPU device is not initialized");
        g_canvas_selector = canvas_selector ? std::string(canvas_selector) : "#mlvc-canvas";

        auto st = std::make_unique<WebDecoderState>();
        st->color_matrix = color_matrix;
        st->config = load_model_config(*g_assets, "MLVCDecoder");
        st->entropy = std::make_unique<EntropyCodec>(g_assets, st->config);

        BackendOptions bopts;
        bopts.backend_name = "webgpu";
        bopts.asset_reader = g_assets;
        st->backend = create_backend(bopts);

        ModelExecutionConfig exec_cfg;
        exec_cfg.state_bindings.push_back({
            3U, 1U, TensorDataType::kFloat16,
            {1, st->config.feature_channels,
             st->config.feature_height(), st->config.feature_width()},
        });
        st->backend->load("MLVCDecoder", exec_cfg);

        st->surface_backend = dynamic_cast<WebGpuSurfaceBackend*>(st->backend.get());
        if (!st->surface_backend)
            throw std::runtime_error("backend does not support WebGpuSurfaceBackend");

        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_src =
            WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
        canvas_src.selector = to_wgpu_string(g_canvas_selector);
        WGPUSurfaceDescriptor sdesc = WGPU_SURFACE_DESCRIPTOR_INIT;
        sdesc.nextInChain = &canvas_src.chain;
        st->surface.reset(wgpuInstanceCreateSurface(g_instance, &sdesc));
        if (!st->surface)
            throw std::runtime_error("wgpuInstanceCreateSurface failed for " + g_canvas_selector);

        st->z_tensor = Tensor{
            "z_raw",
            {1, st->config.hyperprior_channels,
             st->config.hyperprior_height(), st->config.hyperprior_width()},
            std::vector<Float16Storage>(
                static_cast<std::size_t>(st->config.hyperprior_channels) *
                st->config.hyperprior_height() * st->config.hyperprior_width(), 0)};
        const std::vector<std::int64_t> y_shape{
            1, st->config.latent_channels / 2,
            st->config.latent_height(), st->config.latent_width()};
        const std::size_t y_elems =
            static_cast<std::size_t>(st->config.latent_channels / 2) *
            st->config.latent_height() * st->config.latent_width();
        st->y0_tensor = Tensor{"y_raw_0", y_shape, std::vector<Float16Storage>(y_elems, 0)};
        st->y1_tensor = Tensor{"y_raw_1", y_shape, std::vector<Float16Storage>(y_elems, 0)};

        g_decoder = std::move(st);
        return 0;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_decoder_reset()
{
    if (!g_decoder)
        return;
    g_decoder->bitstream_decoder.reset();
    g_decoder->backend->reset_state();
    g_decoder->has_active_ref = false;
    g_decoder->last_frame_idx = -1;
}

EMSCRIPTEN_KEEPALIVE
void mlvc_web_decoder_destroy()
{
    g_decoder.reset();
}

EMSCRIPTEN_KEEPALIVE
int mlvc_web_decode_au(const std::uint8_t* au_data, std::uint32_t au_size,
                       DecodeMetrics* out_metrics)
{
    try {
        if (!g_decoder)
            throw std::runtime_error("decoder not created");
        auto& st = *g_decoder;

        const auto t0 = std::chrono::steady_clock::now();
        const std::span<const std::byte> au_span(
            reinterpret_cast<const std::byte*>(au_data), au_size);
        const FrameData fd = st.bitstream_decoder.decode(au_span);
        if (fd.model_width != st.config.model_width ||
            fd.model_height != st.config.model_height) {
            throw std::runtime_error("bitstream model canvas mismatch");
        }

        configure_surface_for_frame(st, fd);

        const bool resets_ref = (fd.frame_type == FrameType::kIFrame || fd.feature_reset_flag);
        if (!resets_ref) {
            if (!st.has_active_ref || fd.ref_frame_idx != st.last_frame_idx) {
                throw std::runtime_error("missing reference frame (IDR required)");
            }
        }
        const int shifted_q = fd.qp + st.config.q_index_shift(fd.cur_frame_idx);

        st.entropy->decode_into(
            fd.payload, fd.qp,
            mutable_tensor_view(st.z_tensor),
            mutable_tensor_view(st.y0_tensor),
            mutable_tensor_view(st.y1_tensor));
        const auto t1 = std::chrono::steady_clock::now();

        WGPUSurfaceTexture surf_tex = WGPU_SURFACE_TEXTURE_INIT;
        wgpuSurfaceGetCurrentTexture(st.surface.get(), &surf_tex);
        if (surf_tex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            surf_tex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            throw std::runtime_error("wgpuSurfaceGetCurrentTexture failed");
        }
        UniqueTexture frame_tex(surf_tex.texture);
        UniqueTextureView frame_view(wgpuTextureCreateView(frame_tex.get(), nullptr));

        const std::vector<TensorView> latents{
            tensor_view(st.z_tensor),
            tensor_view(st.y0_tensor),
            tensor_view(st.y1_tensor),
        };
        st.surface_backend->submit_decoder_to_texture(
            latents, shifted_q, resets_ref, frame_view.get());
        const auto t2 = std::chrono::steady_clock::now();

        st.has_active_ref = true;
        st.last_frame_idx = fd.cur_frame_idx;

        if (out_metrics) {
            out_metrics->width = fd.display_width();
            out_metrics->height = fd.display_height();
            out_metrics->frame_idx = fd.cur_frame_idx;
            out_metrics->q_index = fd.qp;
            out_metrics->frame_type = static_cast<std::int32_t>(fd.frame_type);
            out_metrics->fused_d2s8 = st.surface_backend->fused_d2s8_active() ? 1 : 0;
            out_metrics->uses_f16 = st.surface_backend->uses_f16() ? 1 : 0;
            out_metrics->dispatches = static_cast<std::int32_t>(st.surface_backend->step_count());
            out_metrics->rans_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            out_metrics->submit_ms =
                std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        return 0;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return -1;
    }
}

}  // extern "C"
