#pragma once
#include <array>
#include <memory>
#include <nexora/core/Result.hpp>
#include <nexora/data/FieldBinding.hpp>
#include <span>
#include <string_view>

namespace nexora::detail {
template <class T> class ExternalField;

struct FieldAxis {
    std::size_t extent;
    bool component = false;
};

struct FieldBindingSpec {
    Association association;
    std::string_view units;
    std::span<const FieldAxis> axes;
};

// Trusted participant entry for synchronous Serial fields. One issuer claims
// an allocation for one stamp. Close stops new grants. Existing grants keep the
// claim and storage alive through their final use.
class FieldIssuer {
  public:
    template <class View>
    static Result<FieldIssuer> create(const View& allocation, FieldStamp stamp) {
        if (auto valid = check_accessibility<View>(); !valid.ok())
            return valid;
        return claim(allocation.impl_track(), stamp, !std::is_const_v<typename View::value_type>);
    }

    FieldIssuer(FieldIssuer&&) noexcept = default;
    FieldIssuer& operator=(FieldIssuer&&) = delete;
    FieldIssuer(const FieldIssuer&) = delete;
    FieldIssuer& operator=(const FieldIssuer&) = delete;
    ~FieldIssuer();

    const FieldStamp& stamp() const noexcept {
        return stamp_;
    }
    void close() noexcept;

    template <class View>
    Result<ReadField<View>> read(const View& view, FieldDescriptor descriptor,
                                 const FieldStamp& expected, const FieldBindingSpec& spec) const {
        if (auto valid = validate_binding(view, descriptor, expected, spec); !valid.ok())
            return valid;
        if (auto granted = acquire(FieldAccess::read); !granted.ok())
            return granted;
        ReadLease lease(authority_);
        return ReadField<View>{std::move(descriptor), stamp_, typename View::const_type(view),
                               std::move(lease), component_axes<View>(spec)};
    }

    template <class View>
        requires (!std::is_const_v<typename View::value_type>)
    Result<WriteField<View>> write(const View& view, FieldDescriptor descriptor,
                                   const FieldStamp& expected, const FieldBindingSpec& spec,
                                   FieldAccess access) const {
        if ((access != FieldAccess::overwrite && access != FieldAccess::accumulate) ||
            descriptor.access != access)
            return Status{StatusCode::invalid_argument, "field does not declare requested write mode"};
        if (auto valid = validate_binding(view, descriptor, expected, spec); !valid.ok())
            return valid;
        if (auto granted = acquire(access); !granted.ok())
            return granted;
        WriteLease lease(authority_);
        return WriteField<View>{std::move(descriptor), stamp_, view, std::move(lease),
                                component_axes<View>(spec)};
    }

  private:
    template <class T> friend class ExternalField;
    struct StorageRange {
        const void* base;
        std::size_t bytes;
    };
    using AllocationRecord = Kokkos::Impl::SharedAllocationRecord<void, void>;

    template <class View> static auto component_axes(const FieldBindingSpec& spec) {
        std::array<bool, View::rank()> axes{};
        for (std::size_t axis = 0; axis < View::rank(); ++axis)
            axes[axis] = spec.axes[axis].component;
        return axes;
    }

    FieldIssuer(std::shared_ptr<FieldAuthority> authority, FieldStamp stamp) noexcept;
    static Result<FieldIssuer> claim(const Kokkos::Impl::SharedAllocationTracker&, FieldStamp,
                                     bool mutable_storage);
    static Result<FieldIssuer> claim_external(StorageRange range, std::shared_ptr<const void> owner,
                                              FieldStamp stamp, bool mutable_storage);
    static Result<FieldIssuer> register_authority(std::shared_ptr<FieldAuthority> authority,
                                                  FieldStamp stamp);
    Status acquire(FieldAccess access) const;

    // This is the private 5.2.2 adapter for recognizing managed subview aliases.
    // Public fields use only the native View interface.
    template <class View> static Status check_accessibility() {
        static_assert(Kokkos::is_view<View>::value);
        if (!Kokkos::is_initialized() || Kokkos::is_finalized())
            return {StatusCode::invalid_state, "field access requires an active Kokkos runtime"};
        if constexpr (!Kokkos::SpaceAccessibility<Kokkos::Serial,
                                                  typename View::memory_space>::accessible)
            return {StatusCode::unsupported, "field storage is inaccessible to Kokkos Serial"};
        return {};
    }

    template <class T> static bool matches_scalar(ScalarType type) {
        using Scalar = std::remove_cv_t<T>;
        switch (type) {
        case ScalarType::float32:
            return std::is_same_v<Scalar, float>;
        case ScalarType::float64:
            return std::is_same_v<Scalar, double>;
        case ScalarType::int32:
            return std::is_same_v<Scalar, std::int32_t>;
        case ScalarType::int64:
            return std::is_same_v<Scalar, std::int64_t>;
        case ScalarType::uint32:
            return std::is_same_v<Scalar, std::uint32_t>;
        case ScalarType::uint64:
            return std::is_same_v<Scalar, std::uint64_t>;
        }
        return false;
    }

    Result<StorageRange> validate_stamp_and_storage(const FieldStamp& expected,
                                                    const AllocationRecord* record, bool unmanaged,
                                                    FieldOwnership ownership) const;
    static Status validate_shape(std::span<const std::size_t> strides,
                                 std::span<const FieldAxis> shape, std::size_t components,
                                 std::size_t scalar_bytes, std::size_t alignment, const void* view_data,
                                 const void* allocation_data, std::size_t allocation_bytes);

    template <class View>
    Status validate_binding(const View& view, const FieldDescriptor& descriptor,
                            const FieldStamp& expected, const FieldBindingSpec& spec) const {
        if (auto valid = descriptor.validate(); !valid.ok())
            return valid;
        if (descriptor.association != spec.association || descriptor.units != spec.units ||
            !matches_scalar<typename View::value_type>(descriptor.scalar_type))
            return {StatusCode::invalid_argument, "field association, units or scalar type mismatch"};
        if (auto valid = check_accessibility<View>(); !valid.ok())
            return valid;
        auto storage = validate_stamp_and_storage(expected, view.impl_track().template get_record<void>(),
                                                  View::memory_traits::is_unmanaged, descriptor.ownership);
        if (!storage.ok())
            return storage.status();

        if (spec.axes.size() != View::rank())
            return {StatusCode::invalid_argument, "field rank mismatch"};
        static_assert(View::rank() <= Kokkos::ARRAY_LAYOUT_MAX_RANK);
        std::array<std::size_t, View::rank()> strides{};
        for (std::size_t axis = 0; axis < View::rank(); ++axis) {
            if (view.extent(axis) != spec.axes[axis].extent)
                return {StatusCode::invalid_argument, "field extent mismatch"};
            strides[axis] = view.stride(axis);
        }
        return validate_shape(strides, spec.axes, descriptor.components, sizeof(typename View::value_type),
                              alignof(typename View::value_type), view.data(), storage.value().base,
                              storage.value().bytes);
    }

    std::shared_ptr<FieldAuthority> authority_;
    FieldStamp stamp_;
};
} // namespace nexora::detail
