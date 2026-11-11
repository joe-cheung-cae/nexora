#pragma once
#include <memory>

namespace nexora {
namespace detail {
struct FieldAuthority;
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
        return static_cast<bool>(authority_);
    }

  private:
    friend class detail::FieldIssuer;
    explicit ReadLease(std::shared_ptr<detail::FieldAuthority> authority) noexcept;
    void release() noexcept;
    std::shared_ptr<detail::FieldAuthority> authority_;
};

class WriteLease {
  public:
    WriteLease(WriteLease&&) noexcept = default;
    WriteLease& operator=(WriteLease&&) noexcept;
    WriteLease(const WriteLease&) = delete;
    WriteLease& operator=(const WriteLease&) = delete;
    ~WriteLease();

    explicit operator bool() const noexcept {
        return static_cast<bool>(authority_);
    }

  private:
    friend class detail::FieldIssuer;
    explicit WriteLease(std::shared_ptr<detail::FieldAuthority> authority) noexcept;
    void release() noexcept;
    std::shared_ptr<detail::FieldAuthority> authority_;
};
} // namespace nexora
