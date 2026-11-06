#pragma once
#include <cstddef>
#include <nexora/core/ArrayView.hpp>
#include <nexora/core/Types.hpp>
#include <stdexcept>
namespace nexora {
// A contiguous borrowed host view. Its owner must preserve storage and layout.
template <class T> class FieldView {
  public:
    FieldView(ArrayView<T> values, std::size_t components, LayoutVersion layout_version = {})
        : values_(values), components_(components), layout_version_(layout_version) {
        if (!components || values.size() % components != 0)
            throw std::invalid_argument("field view shape is invalid");
    }
    ArrayView<T> values() const noexcept {
        return values_;
    }
    std::size_t components() const noexcept {
        return components_;
    }
    std::size_t entities() const noexcept {
        return values_.size() / components_;
    }
    LayoutVersion layout_version() const noexcept {
        return layout_version_;
    }

  private:
    ArrayView<T> values_;
    std::size_t components_;
    LayoutVersion layout_version_;
};
template <class T> struct KernelView {
    T* data{};
    std::size_t size{};
    std::size_t stride{1};
};
} // namespace nexora
