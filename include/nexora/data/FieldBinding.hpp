#pragma once
#include <Kokkos_Core.hpp>
#include <array>
#include <nexora/data/FieldAccess.hpp>
#include <nexora/data/FieldDescriptor.hpp>
#include <nexora/data/FieldStamp.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nexora {
namespace detail {
struct TransportBuilder;
struct TransportReceiver;
}
// Keep the host binding alive through the operation. Capture only the native
// View in Kokkos kernels. A copied View does not extend the access grant.
template <class View> class ReadField {
    static_assert(Kokkos::is_view<View>::value);

  public:
    using view_type = typename View::const_type;
    ReadField(ReadField&&) noexcept = default;
    ReadField& operator=(ReadField&&) noexcept = default;
    ReadField(const ReadField&) = delete;
    ReadField& operator=(const ReadField&) = delete;

    view_type view() const & {
        if (!lease_)
            throw std::logic_error("read field has no active lease");
        return view_;
    }
    view_type view() const && = delete;

    const FieldDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    const FieldStamp& stamp() const noexcept {
        return stamp_;
    }

  private:
    friend class detail::FieldIssuer;
    friend struct detail::TransportBuilder;
    friend struct detail::TransportReceiver;
    ReadField(FieldDescriptor descriptor, FieldStamp stamp, view_type view, ReadLease lease,
              std::array<bool, View::rank()> component_axes)
        : descriptor_(std::move(descriptor)), stamp_(stamp), view_(std::move(view)),
          lease_(std::move(lease)), component_axes_(component_axes) {}

    FieldDescriptor descriptor_;
    FieldStamp stamp_;
    view_type view_;
    ReadLease lease_;
    std::array<bool, View::rank()> component_axes_;
};

template <class View> class WriteField {
    static_assert(Kokkos::is_view<View>::value);
    static_assert(!std::is_const_v<typename View::value_type>);

  public:
    using view_type = View;
    WriteField(WriteField&&) noexcept = default;
    WriteField& operator=(WriteField&&) noexcept = default;
    WriteField(const WriteField&) = delete;
    WriteField& operator=(const WriteField&) = delete;

    View view() const & {
        if (!lease_)
            throw std::logic_error("write field has no active lease");
        return view_;
    }
    view_type view() const && = delete;

    const FieldDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    const FieldStamp& stamp() const noexcept {
        return stamp_;
    }

  private:
    friend class detail::FieldIssuer;
    friend struct detail::TransportReceiver;
    WriteField(FieldDescriptor descriptor, FieldStamp stamp, View view, WriteLease lease,
               std::array<bool, View::rank()> component_axes)
        : descriptor_(std::move(descriptor)), stamp_(stamp), view_(std::move(view)),
          lease_(std::move(lease)), component_axes_(component_axes) {}

    FieldDescriptor descriptor_;
    FieldStamp stamp_;
    View view_;
    WriteLease lease_;
    std::array<bool, View::rank()> component_axes_;
};
} // namespace nexora
