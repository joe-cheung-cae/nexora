#pragma once
#include <cstddef>
#include <memory>
#include <nexora/core/ArrayView.hpp>
namespace nexora {
// CPU storage; views remain valid across moves but are invalidated when their storage is destroyed
// or replaced.
class Buffer {
  public:
    explicit Buffer(std::size_t size = 0)
        : data_(size ? new std::byte[size] : nullptr), size_(size) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
    std::size_t size() const noexcept {
        return size_;
    }
    ArrayView<std::byte> view() & {
        return {data_.get(), size_};
    }
    ArrayView<const std::byte> view() const& {
        return {data_.get(), size_};
    }
    ArrayView<std::byte> view() && = delete;
    ArrayView<const std::byte> view() const&& = delete;

  private:
    std::unique_ptr<std::byte[]> data_;
    std::size_t size_{};
};
} // namespace nexora
