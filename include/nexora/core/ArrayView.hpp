#pragma once
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace nexora {
// Borrowed: its owner must outlive all uses and must not invalidate storage.
template <class T> class ArrayView {
  public:
    using value_type = T;
    constexpr ArrayView() noexcept = default;
    ArrayView(T* data, std::size_t size) : data_(data), size_(size) {
        if (!data && size)
            throw std::invalid_argument("null nonempty view");
    }
    template <class U, class A,
              typename std::enable_if<std::is_convertible<U (*)[], T (*)[]>::value, int>::type = 0>
    ArrayView(std::vector<U, A>& v) : ArrayView(v.data(), v.size()) {}
    template <
        class U, class A,
        typename std::enable_if<std::is_convertible<const U (*)[], T (*)[]>::value, int>::type = 0>
    ArrayView(const std::vector<U, A>& v) : ArrayView(v.data(), v.size()) {}
    template <class U, class A> ArrayView(std::vector<U, A>&&) = delete;
    template <class U, class A> ArrayView(const std::vector<U, A>&&) = delete;
    T* data() const noexcept {
        return data_;
    }
    std::size_t size() const noexcept {
        return size_;
    }
    bool empty() const noexcept {
        return size_ == 0;
    }
    T& operator[](std::size_t i) const noexcept {
        return data_[i];
    }
    T& at(std::size_t i) const {
        if (i >= size_)
            throw std::out_of_range("view index");
        return data_[i];
    }
    T* begin() const noexcept {
        return data_;
    }
    T* end() const noexcept {
        return size_ ? data_ + size_ : data_;
    }

  private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};
} // namespace nexora
