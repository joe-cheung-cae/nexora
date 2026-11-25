#pragma once
#include <memory>

namespace nexora {
namespace detail {
struct FieldGrant;
class FieldIssuer;
} // namespace detail

/** @brief Move-only host lease for synchronous read access to a field allocation.
 *
 * A @ref ReadField holds the lease for its binding. Multiple read leases can
 * coexist for an allocation. Finish all reads, including work using native
 * Views copied from the binding, before releasing the lease.
 */
class ReadLease {
  public:
    ReadLease(ReadLease&&) noexcept = default;
    ReadLease& operator=(ReadLease&&) noexcept;
    ReadLease(const ReadLease&) = delete;
    ReadLease& operator=(const ReadLease&) = delete;
    ~ReadLease();

    explicit operator bool() const noexcept {
        return static_cast<bool>(grant_);
    }

  private:
    friend class detail::FieldIssuer;
    explicit ReadLease(std::shared_ptr<detail::FieldGrant> grant) noexcept;
    std::shared_ptr<detail::FieldGrant> grant_;
};

/** @brief Move-only host lease for exclusive synchronous write access to a field allocation.
 *
 * A @ref WriteField holds the lease for its binding. Finish all work that uses
 * the writable View before releasing the lease.
 */
class WriteLease {
  public:
    WriteLease(WriteLease&&) noexcept = default;
    WriteLease& operator=(WriteLease&&) noexcept;
    WriteLease(const WriteLease&) = delete;
    WriteLease& operator=(const WriteLease&) = delete;
    ~WriteLease();

    explicit operator bool() const noexcept {
        return static_cast<bool>(grant_);
    }

  private:
    friend class detail::FieldIssuer;
    explicit WriteLease(std::shared_ptr<detail::FieldGrant> grant) noexcept;
    std::shared_ptr<detail::FieldGrant> grant_;
};
} // namespace nexora
