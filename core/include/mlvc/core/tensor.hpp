#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace mlvc {

enum class TensorDataType { kFloat16, kInt32 };

// IEEE-754 binary16 is stored as raw 16-bit data at every backend boundary.
using Float16Storage = std::uint16_t;
using TensorStorage = std::variant<
    std::vector<Float16Storage>, std::vector<std::int32_t>>;

inline std::size_t tensor_data_type_size(TensorDataType type)
{
    switch (type) {
    case TensorDataType::kFloat16:
        return sizeof(Float16Storage);
    case TensorDataType::kInt32:
        return sizeof(std::int32_t);
    }
    throw std::runtime_error("unsupported tensor data type");
}

inline std::size_t checked_tensor_element_count(
    std::span<const std::int64_t> shape)
{
    std::size_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0)
            throw std::runtime_error("tensor dimensions must be positive");
        const auto extent = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / extent)
            throw std::runtime_error("tensor element count overflows size_t");
        count *= extent;
    }
    return count;
}

inline std::size_t checked_tensor_byte_size(
    std::span<const std::int64_t> shape, TensorDataType type)
{
    const std::size_t count = checked_tensor_element_count(shape);
    const std::size_t element_size = tensor_data_type_size(type);
    if (count > std::numeric_limits<std::size_t>::max() / element_size)
        throw std::runtime_error("tensor byte size overflows size_t");
    return count * element_size;
}

struct Tensor {
    std::string name;
    std::vector<std::int64_t> shape;
    TensorStorage data;

    TensorDataType data_type() const noexcept
    {
        return std::holds_alternative<std::vector<Float16Storage>>(data)
            ? TensorDataType::kFloat16 : TensorDataType::kInt32;
    }

    std::size_t element_count() const noexcept
    {
        return std::visit([](const auto& values) { return values.size(); }, data);
    }

    std::size_t byte_size() const
    {
        return std::visit([](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            return values.size() * sizeof(Value);
        }, data);
    }

    const void* raw_data() const noexcept
    {
        return std::visit([](const auto& values) -> const void* {
            return values.data();
        }, data);
    }

    void* raw_data() noexcept
    {
        return std::visit([](auto& values) -> void* {
            return values.data();
        }, data);
    }

    void validate() const
    {
        if (checked_tensor_element_count(shape) != element_count())
            throw std::runtime_error("tensor shape does not match its storage");
    }
};

class TensorView {
public:
    TensorView(std::string_view name, std::span<const std::int64_t> shape,
               std::span<const Float16Storage> data)
        : TensorView(name, shape, TensorDataType::kFloat16,
                     data.data(), data.size())
    {
    }

    TensorView(std::string_view name, std::span<const std::int64_t> shape,
               std::span<const std::int32_t> data)
        : TensorView(name, shape, TensorDataType::kInt32,
                     data.data(), data.size())
    {
    }

    static TensorView from_raw(std::string_view name,
                               std::span<const std::int64_t> shape,
                               TensorDataType type, const void* data,
                               std::size_t bytes)
    {
        const std::size_t expected = checked_tensor_byte_size(shape, type);
        if (bytes != expected || data == nullptr)
            throw std::runtime_error(
                "tensor view shape does not match its storage");
        return TensorView(name, shape, type, data,
                          expected / tensor_data_type_size(type));
    }

    std::string_view name() const noexcept { return name_; }
    std::span<const std::int64_t> shape() const noexcept { return shape_; }
    TensorDataType data_type() const noexcept { return data_type_; }
    const void* raw_data() const noexcept { return data_; }
    std::size_t element_count() const noexcept { return element_count_; }
    std::size_t byte_size() const noexcept
    {
        return element_count_ * tensor_data_type_size(data_type_);
    }

    std::span<const Float16Storage> fp16_data() const
    {
        if (data_type_ != TensorDataType::kFloat16)
            throw std::runtime_error("tensor view is not FP16");
        return {static_cast<const Float16Storage*>(data_), element_count_};
    }

    std::span<const std::int32_t> int32_data() const
    {
        if (data_type_ != TensorDataType::kInt32)
            throw std::runtime_error("tensor view is not int32");
        return {static_cast<const std::int32_t*>(data_), element_count_};
    }

private:
    TensorView(std::string_view name, std::span<const std::int64_t> shape,
               TensorDataType type, const void* data, std::size_t element_count)
        : name_(name), shape_(shape), data_type_(type), data_(data),
          element_count_(element_count)
    {
        if (checked_tensor_element_count(shape_) != element_count_ ||
            data_ == nullptr) {
            throw std::runtime_error(
                "tensor view shape does not match its storage");
        }
    }

    std::string_view name_;
    std::span<const std::int64_t> shape_;
    TensorDataType data_type_;
    const void* data_;
    std::size_t element_count_;
};

class MutableTensorView {
public:
    MutableTensorView(std::string_view name,
                      std::span<const std::int64_t> shape,
                      std::span<Float16Storage> data)
        : MutableTensorView(name, shape, TensorDataType::kFloat16,
                            data.data(), data.size())
    {
    }

    MutableTensorView(std::string_view name,
                      std::span<const std::int64_t> shape,
                      std::span<std::int32_t> data)
        : MutableTensorView(name, shape, TensorDataType::kInt32,
                            data.data(), data.size())
    {
    }

    static MutableTensorView from_raw(std::string_view name,
                                      std::span<const std::int64_t> shape,
                                      TensorDataType type, void* data,
                                      std::size_t bytes)
    {
        const std::size_t expected = checked_tensor_byte_size(shape, type);
        if (bytes != expected || data == nullptr)
            throw std::runtime_error(
                "mutable tensor view shape does not match its storage");
        return MutableTensorView(name, shape, type, data,
                                 expected / tensor_data_type_size(type));
    }

    std::string_view name() const noexcept { return name_; }
    std::span<const std::int64_t> shape() const noexcept { return shape_; }
    TensorDataType data_type() const noexcept { return data_type_; }
    void* raw_data() const noexcept { return data_; }
    std::size_t element_count() const noexcept { return element_count_; }
    std::size_t byte_size() const
    {
        return element_count_ * tensor_data_type_size(data_type_);
    }

    std::span<Float16Storage> fp16_data() const
    {
        if (data_type_ != TensorDataType::kFloat16)
            throw std::runtime_error("mutable tensor view is not FP16");
        return {static_cast<Float16Storage*>(data_), element_count_};
    }

    std::span<std::int32_t> int32_data() const
    {
        if (data_type_ != TensorDataType::kInt32)
            throw std::runtime_error("mutable tensor view is not int32");
        return {static_cast<std::int32_t*>(data_), element_count_};
    }

    operator TensorView() const
    {
        return TensorView::from_raw(name_, shape_, data_type_, data_, byte_size());
    }

private:
    MutableTensorView(std::string_view name,
                      std::span<const std::int64_t> shape,
                      TensorDataType type, void* data,
                      std::size_t element_count)
        : name_(name), shape_(shape), data_type_(type), data_(data),
          element_count_(element_count)
    {
        if (checked_tensor_element_count(shape_) != element_count_ ||
            data_ == nullptr) {
            throw std::runtime_error(
                "mutable tensor view shape does not match its storage");
        }
    }

    std::string_view name_;
    std::span<const std::int64_t> shape_;
    TensorDataType data_type_;
    void* data_;
    std::size_t element_count_;
};

inline TensorView tensor_view(const Tensor& tensor)
{
    tensor.validate();
    if (tensor.data_type() == TensorDataType::kFloat16) {
        return TensorView(tensor.name, tensor.shape,
                          std::get<std::vector<Float16Storage>>(tensor.data));
    }
    return TensorView(tensor.name, tensor.shape,
                      std::get<std::vector<std::int32_t>>(tensor.data));
}

inline MutableTensorView mutable_tensor_view(Tensor& tensor)
{
    tensor.validate();
    if (tensor.data_type() == TensorDataType::kFloat16) {
        return MutableTensorView(
            tensor.name, tensor.shape,
            std::get<std::vector<Float16Storage>>(tensor.data));
    }
    return MutableTensorView(
        tensor.name, tensor.shape,
        std::get<std::vector<std::int32_t>>(tensor.data));
}

}  // namespace mlvc
