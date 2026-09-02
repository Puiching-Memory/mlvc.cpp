#include "mlvc/driver/driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" const unsigned char mlvc_driver_kernels_fatbin[];
extern "C" const std::size_t mlvc_driver_kernels_fatbin_size;

namespace {

struct Options {
    int device_id = 0;
    std::size_t iterations = 1000;
};

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s [--device-id N] [--iterations N]\n",
        argv0);
}

template <typename T>
T parse_unsigned(const char* text, std::string_view option, bool allow_zero)
{
    try {
        std::size_t parsed = 0;
        const unsigned long long value = std::stoull(text, &parsed);
        if (text[parsed] == '\0' && (allow_zero || value > 0) &&
            value <= static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
            return static_cast<T>(value);
        }
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(option) + " has an invalid value");
}

Options parse_cli(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(argument) + " requires a value");
            return argv[++i];
        };
        if (argument == "--device-id") {
            options.device_id = parse_unsigned<int>(value(), argument, true);
        } else if (argument == "--iterations") {
            options.iterations = parse_unsigned<std::size_t>(value(), argument, false);
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    return options;
}

std::vector<std::uint16_t> make_fp16_bits(std::size_t count)
{
    constexpr std::uint16_t seeds[] = {
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00,
        0xbc00, 0x3555, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
    };
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = i < std::size(seeds)
            ? seeds[i]
            : static_cast<std::uint16_t>((i * 40503U + 17U) & 0xffffU);
    }
    return values;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_cli(argc, argv);
        constexpr std::uint32_t element_count = 1U << 20;
        constexpr unsigned int block_size = 256;
        const std::size_t bytes =
            static_cast<std::size_t>(element_count) * sizeof(std::uint16_t);

        mlvc::driver::Driver driver(options.device_id);
        const auto image = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(mlvc_driver_kernels_fatbin),
            mlvc_driver_kernels_fatbin_size);
        mlvc::driver::Module module = driver.load_module(image);
        const auto kernel = module.function("mlvc_copy_fp16_bits");
        mlvc::driver::DeviceBuffer input_device = driver.allocate(bytes);
        mlvc::driver::DeviceBuffer output_device = driver.allocate(bytes);

        const std::vector<std::uint16_t> input = make_fp16_bits(element_count);
        std::vector<std::uint16_t> output(element_count);
        driver.upload_async(input_device, input.data(), bytes);

        const mlvc::driver::abi::DeviceAddress input_address = input_device.address();
        const mlvc::driver::abi::DeviceAddress output_address = output_device.address();
        std::uint32_t count_parameter = element_count;
        void* parameters[] = {
            const_cast<mlvc::driver::abi::DeviceAddress*>(&input_address),
            const_cast<mlvc::driver::abi::DeviceAddress*>(&output_address),
            &count_parameter,
        };
        const mlvc::driver::Dim3 grid{
            (element_count + block_size - 1) / block_size, 1, 1};
        const mlvc::driver::Dim3 block{block_size, 1, 1};

        for (int i = 0; i < 20; ++i)
            driver.launch(kernel, grid, block, 0, parameters);
        driver.synchronize();

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < options.iterations; ++i)
            driver.launch(kernel, grid, block, 0, parameters);
        driver.synchronize();
        const auto stop = std::chrono::steady_clock::now();

        driver.download_async(output.data(), output_device, bytes);
        driver.synchronize();

        const auto mismatch = std::mismatch(input.begin(), input.end(), output.begin());
        if (mismatch.first != input.end()) {
            const std::size_t index =
                static_cast<std::size_t>(mismatch.first - input.begin());
            throw std::runtime_error(
                "FP16 bit mismatch at element " + std::to_string(index));
        }

        const double elapsed_us =
            std::chrono::duration<double, std::micro>(stop - start).count();
        const double mean_us = elapsed_us / static_cast<double>(options.iterations);
        const mlvc::driver::DeviceInfo& info = driver.device_info();
        std::printf("driver-cubin probe passed\n");
        std::printf("device: %s (sm_%d%d, ordinal %d)\n",
                    info.name.c_str(), info.compute_major, info.compute_minor,
                    info.ordinal);
        std::printf("driver API version: %d\n", info.driver_version);
        std::printf("embedded fatbin: %zu bytes\n", mlvc_driver_kernels_fatbin_size);
        std::printf("FP16 payload: %zu bytes, bit mismatches: 0\n", bytes);
        std::printf("kernel launches: %zu, mean launch+execution: %.3f us\n",
                    options.iterations, mean_us);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
