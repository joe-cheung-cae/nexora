#pragma once
#include <memory>

namespace nexora {
namespace detail {
struct FieldGrant;
class FieldIssuer;
} // namespace detail

// Host-only grants for synchronous access. The operation must finish before its
// grant is released, including all uses of native Views copied from the binding.
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
