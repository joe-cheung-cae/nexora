#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Types.hpp>

namespace nexora::detail {
// Counts also fit LocalIndex: loops and sentinel counts must not wrap.
inline Result<std::size_t> checked_entity_count(std::uint64_t count) {
    const auto local_count = checked_integer_cast<LocalIndex>(count);
    if (!local_count.ok())
        return local_count.status();
    return checked_integer_cast<std::size_t>(count);
}
inline Result<LocalIndex> checked_local_index(std::uint64_t index, std::uint64_t count) {
    const auto capacity = checked_entity_count(count);
    if (!capacity.ok())
        return capacity.status();
    if (index >= count)
        return Status{StatusCode::invalid_argument, "local index is outside entity order"};
    return checked_integer_cast<LocalIndex>(index);
}
inline Result<std::size_t> checked_array_bytes(std::uint64_t count, std::size_t element_size) {
    if (element_size == 0 || count > std::numeric_limits<std::size_t>::max() / element_size)
        return Status{StatusCode::invalid_argument, "array byte size exceeds capacity"};
    const auto size = checked_integer_cast<std::size_t>(count);
    if (!size.ok())
        return size.status();
    return size.value() * element_size;
}
inline Result<std::size_t> checked_csr_capacity(std::uint64_t rows, std::uint64_t columns,
                                                std::uint64_t entries) {
    if (rows == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::invalid_argument, "CSR row sentinel exceeds capacity"};
    const auto row_count = checked_entity_count(rows);
    const auto column_count = checked_entity_count(columns);
    if (!row_count.ok())
        return row_count.status();
    if (!column_count.ok())
        return column_count.status();
    const auto offsets = checked_array_bytes(rows + 1, sizeof(Offset));
    const auto indices = checked_array_bytes(entries, sizeof(LocalIndex));
    if (!offsets.ok())
        return offsets.status();
    if (!indices.ok())
        return indices.status();
    if (offsets.value() > std::numeric_limits<std::size_t>::max() - indices.value())
        return Status{StatusCode::invalid_argument, "CSR aggregate byte size exceeds capacity"};
    return offsets.value() + indices.value();
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
