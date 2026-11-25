#include "FieldIssuer.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <vector>

namespace nexora::detail {
namespace {
using AllocationTracker = Kokkos::Impl::SharedAllocationTracker;
using AllocationRecord = Kokkos::Impl::SharedAllocationRecord<void, void>;

struct AllocationClaims {
    std::mutex mutex;
    std::map<std::uintptr_t, std::size_t> active;
    std::set<std::weak_ptr<const void>, std::owner_less<>> owners;
    std::uint64_t last_incarnation{};
};

AllocationClaims& claims() {
    static AllocationClaims registry;
    return registry;
}
} // namespace

struct FieldAuthority {
    FieldAuthority(const AllocationTracker& tracker, const AllocationRecord& record, bool writable)
        : allocation(tracker), base(record.data()), bytes(record.size()), writable(writable) {}

    FieldAuthority(const void* base, std::size_t bytes, std::shared_ptr<const void> owner,
                   bool writable)
        : owner(std::move(owner)), base(base), bytes(bytes), writable(writable) {}

    std::uintptr_t address() const noexcept {
        return reinterpret_cast<std::uintptr_t>(base);
    }

    ~FieldAuthority() {
        if (registered) {
            auto& registry = claims();
            const std::lock_guard lock(registry.mutex);
            registry.active.erase(address());
            if (owner)
                registry.owners.erase(std::weak_ptr<const void>(owner));
        }
    }

    AllocationTracker allocation;
    const std::shared_ptr<const void> owner;
    const void* const base;
    const std::size_t bytes;
    const bool writable;
    bool registered{};
    std::uint64_t incarnation{};
    std::mutex mutex;
    std::size_t readers{};
    bool writing{};
    bool closed{};
};

// One record per allocation in a granted group. Every returned binding for
// that allocation shares the record, so any binding can outlive its siblings.
struct FieldGrant {
    FieldGrant(std::shared_ptr<FieldAuthority> authority, FieldAccess access) noexcept
        : authority(std::move(authority)), access(access) {}

    ~FieldGrant() {
        if (!active)
            return;
        const std::lock_guard lock(authority->mutex);
        if (access == FieldAccess::read)
            --authority->readers;
        else
            authority->writing = false;
    }

    // The caller holds this authority's mutex during both checks and updates.
    Status validate() const {
        if (authority->closed)
            return {StatusCode::invalid_state, "field issuer is closed"};
        if (access == FieldAccess::read) {
            if (authority->writing)
                return {StatusCode::invalid_state, "allocation has an active writer"};
            if (authority->readers == std::numeric_limits<std::size_t>::max())
                return {StatusCode::resource_exhausted, "field reader count exhausted"};
        } else {
            if (!authority->writable)
                return {StatusCode::invalid_state, "field allocation is read-only"};
            if (authority->writing || authority->readers != 0)
                return {StatusCode::invalid_state, "allocation has active readers or a writer"};
        }
        return {};
    }

    void activate() noexcept {
        if (access == FieldAccess::read)
            ++authority->readers;
        else
            authority->writing = true;
        active = true;
    }

    const std::shared_ptr<FieldAuthority> authority;
    const FieldAccess access;
    bool active{};
};

FieldIssuer::FieldIssuer(std::shared_ptr<FieldAuthority> authority, FieldStamp stamp) noexcept
    : authority_(std::move(authority)), stamp_(stamp) {}

Result<FieldIssuer> FieldIssuer::claim(const AllocationTracker& tracker, FieldStamp stamp,
                                       bool mutable_storage) {
    const auto* record = tracker.get_record<void>();
    if (!record)
        return Status{StatusCode::unsupported, "field access requires a tracked Kokkos allocation"};
    auto authority = std::make_shared<FieldAuthority>(tracker, *record,
                                                      mutable_storage && !stamp.accepted);
    if (!authority->allocation.get_record<void>())
        return Status{StatusCode::unsupported, "Kokkos allocation tracking is disabled"};
    return register_authority(std::move(authority), stamp);
}

Result<FieldIssuer> FieldIssuer::claim_external(StorageRange range,
                                                std::shared_ptr<const void> owner, FieldStamp stamp,
                                                bool mutable_storage) {
    if (auto valid = stamp.validate(); !valid.ok())
        return valid;
    auto authority = std::make_shared<FieldAuthority>(range.base, range.bytes, std::move(owner),
                                                      mutable_storage && !stamp.accepted);
    return register_authority(std::move(authority), stamp);
}

Result<FieldIssuer> FieldIssuer::register_authority(std::shared_ptr<FieldAuthority> authority,
                                                    FieldStamp stamp) {
    if (stamp.storage_incarnation != 0)
        return Status{StatusCode::invalid_argument, "new field stamp already has an incarnation"};

    auto& registry = claims();
    {
        const std::lock_guard lock(registry.mutex);
        if (registry.last_incarnation == std::numeric_limits<std::uint64_t>::max())
            return Status{StatusCode::resource_exhausted, "field incarnation exhausted"};
        const auto address = authority->address();
        const auto next = registry.active.lower_bound(address);
        const bool overlaps_next = next != registry.active.end() &&
                                   (next->first == address || next->first - address < authority->bytes);
        bool overlaps_previous = false;
        if (next != registry.active.begin()) {
            const auto previous = std::prev(next);
            overlaps_previous = address - previous->first < previous->second;
        }
        if (overlaps_next || overlaps_previous ||
            (authority->owner && registry.owners.contains(authority->owner)))
            return Status{StatusCode::already_exists, "allocation already has a field issuer"};

        const auto entry = registry.active.emplace_hint(next, address, authority->bytes);
        try {
            if (authority->owner)
                registry.owners.emplace(authority->owner);
        } catch (...) {
            registry.active.erase(entry);
            throw;
        }
        authority->registered = true;
        authority->incarnation = ++registry.last_incarnation;
        stamp.storage_incarnation = authority->incarnation;
    }
    return FieldIssuer(std::move(authority), stamp);
}

FieldIssuer::~FieldIssuer() {
    close();
}

void FieldIssuer::close() noexcept {
    if (authority_) {
        const std::lock_guard lock(authority_->mutex);
        authority_->closed = true;
    }
}

Status FieldIssuer::check_runtime() {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
        return {StatusCode::invalid_state, "field access requires an active Kokkos runtime"};
    return {};
}

Status FieldIssuer::acquire_group(std::span<GroupMember> members,
                                  std::span<std::shared_ptr<FieldGrant>> grants) {
    if (auto valid = check_runtime(); !valid.ok())
        return valid;
    for (std::size_t index = 0; index < members.size(); ++index)
        members[index].index = index;
    std::ranges::sort(members, {}, [](const GroupMember& member) {
        return member.authority->incarnation;
    });
    for (std::size_t index = 1; index < members.size(); ++index) {
        const auto& previous = members[index - 1];
        const auto& current = members[index];
        if (previous.authority == current.authority && previous.access != current.access)
            return {StatusCode::invalid_argument, "one allocation has conflicting group access modes"};
    }

    try {
        std::vector<std::shared_ptr<FieldGrant>> distinct;
        distinct.reserve(members.size());
        for (const auto& member : members) {
            if (distinct.empty() || distinct.back()->authority != member.authority)
                distinct.push_back(std::make_shared<FieldGrant>(member.authority, member.access));
            grants[member.index] = distinct.back();
        }

        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(distinct.size());
        for (const auto& grant : distinct)
            locks.emplace_back(grant->authority->mutex);
        for (const auto& grant : distinct) {
            if (auto valid = grant->validate(); !valid.ok())
                return valid;
        }
        for (const auto& grant : distinct)
            grant->activate();
        return {};
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot prepare field grants"};
    }
}

Result<FieldIssuer::StorageRange> FieldIssuer::validate_stamp_and_storage(
    const FieldStamp& expected, const AllocationRecord* record, bool unmanaged,
    FieldOwnership ownership) const {
    if (!authority_)
        return Status{StatusCode::invalid_state, "field issuer has no allocation"};
    // This equality also authenticates time against the validated owned stamp.
    if (expected != stamp_)
        return Status{StatusCode::invalid_state, "field stamp is stale or does not match the binding"};

    const bool external = static_cast<bool>(authority_->owner);
    if (external) {
        if (ownership != FieldOwnership::external)
            return Status{StatusCode::invalid_argument, "field ownership does not match its allocation"};
        if (!unmanaged)
            return Status{StatusCode::invalid_argument, "external field requires an unmanaged View"};
    } else {
        if (ownership == FieldOwnership::external || !record)
            return Status{StatusCode::unsupported, "untracked storage requires an external owner"};
        if (record != authority_->allocation.get_record<void>())
            return Status{StatusCode::invalid_argument, "view belongs to a different allocation"};
    }
    return StorageRange{authority_->base, authority_->bytes};
}

Status FieldIssuer::validate_shape(std::span<const std::size_t> strides,
                                   std::span<const FieldAxis> shape, std::size_t components,
                                   std::size_t scalar_bytes, std::size_t alignment, const void* view_data,
                                   const void* allocation_data, std::size_t allocation_bytes) {
    constexpr auto max_size = std::numeric_limits<std::size_t>::max();
    constexpr auto max_entities = std::numeric_limits<LocalIndex>::max();
    const bool empty = std::ranges::any_of(shape, [](const FieldAxis& axis) {
        return !axis.component && axis.extent == 0;
    });
    std::size_t component_count = 1;
    std::size_t entity_count = 1;
    for (const auto& axis : shape) {
        if (axis.component) {
            if (axis.extent == 0 || component_count > max_size / axis.extent)
                return {StatusCode::invalid_argument, "invalid component extent"};
            component_count *= axis.extent;
        } else if (!empty) {
            if (entity_count > max_entities / axis.extent)
                return {StatusCode::invalid_argument, "field entity count exceeds LocalIndex capacity"};
            entity_count *= axis.extent;
        }
    }
    if (component_count != components)
        return {StatusCode::invalid_argument, "field component count mismatch"};
    if (empty)
        return {};

    // Native strides describe the mapping. Verify a non-overlapping span, from
    // the innermost stride out, without requiring contiguous storage.
    std::array<std::pair<std::size_t, std::size_t>, Kokkos::ARRAY_LAYOUT_MAX_RANK> axes{};
    std::size_t active_axes = 0;
    for (std::size_t axis = 0; axis < strides.size(); ++axis) {
        if (shape[axis].extent > 1)
            axes[active_axes++] = {strides[axis], shape[axis].extent - 1};
    }
    auto varying_axes = std::span(axes).first(active_axes);
    std::ranges::sort(varying_axes);
    std::size_t span = 1;
    for (const auto& [stride, steps] : varying_axes) {
        if (stride < span || steps > (max_size - span) / stride)
            return {StatusCode::invalid_argument, "field strides overlap or exceed capacity"};
        span += steps * stride;
    }
    const auto start = reinterpret_cast<std::uintptr_t>(view_data);
    const auto base = reinterpret_cast<std::uintptr_t>(allocation_data);
    if (!view_data || start % alignment != 0 || start < base || start - base > allocation_bytes ||
        span > (allocation_bytes - (start - base)) / scalar_bytes)
        return {StatusCode::invalid_argument, "field view exceeds its allocation"};
    return {};
}
} // namespace nexora::detail

namespace nexora {
ReadLease::ReadLease(std::shared_ptr<detail::FieldGrant> grant) noexcept
    : grant_(std::move(grant)) {}

ReadLease& ReadLease::operator=(ReadLease&&) noexcept = default;
ReadLease::~ReadLease() = default;

WriteLease::WriteLease(std::shared_ptr<detail::FieldGrant> grant) noexcept
    : grant_(std::move(grant)) {}

WriteLease& WriteLease::operator=(WriteLease&&) noexcept = default;
WriteLease::~WriteLease() = default;
} // namespace nexora
