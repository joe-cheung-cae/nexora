#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <nexora/core/Result.hpp>
#include <utility>

namespace nexora::detail {

class ResourceBudget {
  public:
    struct Limits {
        std::size_t snapshot_records = 64;
        std::size_t controlled_bytes = 256 * 1024 * 1024;
    };

    struct Usage {
        std::size_t snapshot_records = 0;
        std::size_t controlled_bytes = 0;
        // These synchronous participants retain no external owners.
        std::size_t known_external_owner_bytes = 0;
    };

    // Hold the ticket with the storage or operation record. Share the same
    // ticket among holders of one allocation or one issued snapshot record.
    using Ticket = std::shared_ptr<const void>;

    ResourceBudget() : ResourceBudget(Limits{}) {}
    explicit ResourceBudget(Limits limits) : account_(std::make_shared<Account>(limits)) {}
    ResourceBudget(const ResourceBudget&) = delete;
    ResourceBudget& operator=(const ResourceBudget&) = delete;

    Result<Ticket> reserve(std::size_t controlled_bytes, std::size_t snapshot_records = 0) const {
        std::shared_ptr<Reservation> ticket;
        {
            const std::lock_guard lock(account_->mutex);
            auto& usage = account_->usage;
            const auto& limits = account_->limits;
            if (snapshot_records > limits.snapshot_records - usage.snapshot_records ||
                controlled_bytes > limits.controlled_bytes - usage.controlled_bytes) {
                return Status{StatusCode::resource_exhausted, "participant resource budget exceeded"};
            }
            try {
                ticket = std::make_shared<Reservation>(account_, controlled_bytes, snapshot_records);
            } catch (const std::bad_alloc&) {
                return Status{StatusCode::resource_exhausted, "cannot reserve participant resources"};
            }
            usage.controlled_bytes += controlled_bytes;
            usage.snapshot_records += snapshot_records;
        }
        return Ticket(std::move(ticket));
    }

    Usage usage() const {
        const std::lock_guard lock(account_->mutex);
        return account_->usage;
    }

  private:
    struct Account {
        explicit Account(Limits limits) : limits(limits) {}

        const Limits limits;
        std::mutex mutex;
        Usage usage;
    };

    struct Reservation {
        Reservation(std::shared_ptr<Account> account, std::size_t bytes, std::size_t records)
            : account(std::move(account)), bytes(bytes), records(records) {}

        ~Reservation() {
            const std::lock_guard lock(account->mutex);
            account->usage.controlled_bytes -= bytes;
            account->usage.snapshot_records -= records;
        }

        const std::shared_ptr<Account> account;
        const std::size_t bytes, records;
    };

    const std::shared_ptr<Account> account_;
};

} // namespace nexora::detail
