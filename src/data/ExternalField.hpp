#pragma once
#include "FieldIssuer.hpp"

#include <limits>
#include <new>

namespace nexora::detail {
// The adapter supplies the real allocation base and capacity. It must finish
// producer work before import and prevent resize, free, or out-of-band writes
// until the issuer and all its grants are released.
template <class T> class ExternalField {
  public:
    using view_type = Kokkos::View<T*, Kokkos::LayoutStride, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static Result<ExternalField> create(T* data, std::size_t entities, std::size_t stride,
                                        std::size_t accessible_bytes,
                                        std::shared_ptr<const void> owner, FieldStamp stamp) {
        if (auto valid = FieldIssuer::check_accessibility<view_type>(); !valid.ok())
            return valid;
        if (!data || owner.get() != data || owner.use_count() == 0)
            return Status{StatusCode::invalid_argument, "external owner must retain the allocation base"};

        const auto address = reinterpret_cast<std::uintptr_t>(data);
        if (address % alignof(T) != 0 ||
            accessible_bytes > std::numeric_limits<std::uintptr_t>::max() - address)
            return Status{StatusCode::invalid_argument, "external allocation range is invalid"};
        if (stride == 0 ||
            (entities != 0 && stride > std::numeric_limits<std::size_t>::max() / entities))
            return Status{StatusCode::invalid_argument, "external field stride is invalid"};

        const std::array axes{FieldAxis{entities, false}};
        const std::array strides{stride};
        if (auto valid = FieldIssuer::validate_shape(strides, axes, 1,
                                                     sizeof(T), alignof(T), data, data,
                                                     accessible_bytes);
            !valid.ok())
            return valid;

        try {
            auto issuer = FieldIssuer::claim_external({data, accessible_bytes}, std::move(owner),
                                                      stamp, !std::is_const_v<T>);
            if (!issuer.ok())
                return issuer.status();
            view_type view(data, Kokkos::LayoutStride(entities, stride));
            return ExternalField(std::move(issuer).value(), std::move(view));
        } catch (const std::bad_alloc&) {
            return Status{StatusCode::resource_exhausted, "cannot register external allocation"};
        }
    }

    const FieldStamp& stamp() const noexcept {
        return issuer_.stamp();
    }
    void close() noexcept {
        issuer_.close();
    }

    Result<ReadField<view_type>> read(FieldDescriptor descriptor, const FieldStamp& expected,
                                    const FieldBindingSpec& spec) const {
        return issuer_.read(view_, std::move(descriptor), expected, spec);
    }

    Result<WriteField<view_type>> write(FieldDescriptor descriptor, const FieldStamp& expected,
                                      const FieldBindingSpec& spec, FieldAccess access) const
        requires (!std::is_const_v<T>)
    {
        return issuer_.write(view_, std::move(descriptor), expected, spec, access);
    }

  private:
    ExternalField(FieldIssuer issuer, view_type view)
        : issuer_(std::move(issuer)), view_(std::move(view)) {}

    FieldIssuer issuer_;
    view_type view_;
};
} // namespace nexora::detail
