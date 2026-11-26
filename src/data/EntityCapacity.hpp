#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nexora/core/Result.hpp>
#include <nexora/core/Types.hpp>

namespace nexora::detail {
// Counts also fit LocalIndex: loops and sentinel counts must not wrap.
inline Result<std::size_t> checked_entity_count(std::uint64_t count) {
    if (count > std::numeric_limits<LocalIndex>::max())
        return Status{StatusCode::invalid_argument, "integer conversion out of range"};
    return static_cast<std::size_t>(count);
}
inline Result<std::size_t> checked_array_bytes(std::uint64_t count, std::size_t element_size) {
    if (element_size == 0 || count > std::numeric_limits<std::size_t>::max() / element_size)
        return Status{StatusCode::invalid_argument, "array byte size exceeds capacity"};
    return static_cast<std::size_t>(count) * element_size;
}
// The row count is already bounded by LocalIndex.
inline Result<std::size_t> checked_csr_bytes(std::size_t rows, std::uint64_t entries) {
    static_assert(std::numeric_limits<LocalIndex>::max() <
                  std::numeric_limits<std::size_t>::max() / sizeof(Offset));
    const auto offsets = (rows + 1) * sizeof(Offset);
    const auto indices = checked_array_bytes(entries, sizeof(LocalIndex));
    if (!indices.ok())
        return indices.status();
    if (offsets > std::numeric_limits<std::size_t>::max() - indices.value())
        return Status{StatusCode::invalid_argument, "CSR aggregate byte size exceeds capacity"};
    return offsets + indices.value();
}
inline Result<std::size_t> checked_csr_capacity(std::uint64_t rows, std::uint64_t columns,
                                               std::uint64_t entries) {
    const auto row_count = checked_entity_count(rows);
    if (!row_count.ok())
        return row_count.status();
    const auto column_count = checked_entity_count(columns);
    if (!column_count.ok())
        return column_count.status();
    return checked_csr_bytes(row_count.value(), entries);
}
inline Result<std::uint64_t> checked_increment(std::uint64_t value) {
    if (value == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::invalid_state, "version counter exhausted"};
    return value + 1;
}
template <class Tag> Result<StrongValue<Tag>> checked_next_version(StrongValue<Tag> version) {
    const auto next = checked_increment(version.value());
    if (!next.ok())
        return next.status();
    return StrongValue<Tag>{next.value()};
}
} // namespace nexora::detail
