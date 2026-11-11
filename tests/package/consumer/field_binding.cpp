#include <nexora/core/Real.hpp>
#include <nexora/data/TransportView.hpp>

#include <nexora/data/FieldAccess.hpp>
#include <nexora/data/FieldBinding.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

using ExpectedReal = std::conditional_t<NEXORA_TEST_EXPECT_DOUBLE_PRECISION, double, float>;
static_assert(std::is_same_v<nexora::real, ExpectedReal>);
static_assert(nexora::real_scalar_type ==
              (std::is_same_v<ExpectedReal, float> ? nexora::ScalarType::float32
                                                  : nexora::ScalarType::float64));
using View = Kokkos::View<nexora::real*>;
static_assert(std::is_same_v<View::value_type, ExpectedReal>);
using ReadField = nexora::ReadField<View>;
using WriteField = nexora::WriteField<View>;

static_assert(std::is_same_v<decltype(std::declval<const ReadField&>().view()), View::const_type>);
static_assert(std::is_same_v<decltype(std::declval<const WriteField&>().view()), View>);
static_assert(std::is_move_constructible_v<ReadField> && !std::is_copy_constructible_v<ReadField>);
static_assert(std::is_move_constructible_v<WriteField> && !std::is_copy_constructible_v<WriteField>);
static_assert(!std::is_default_constructible_v<nexora::ReadLease>);
static_assert(!std::is_default_constructible_v<nexora::WriteLease>);
static_assert(std::is_nothrow_move_constructible_v<nexora::ReadLease> &&
              std::is_nothrow_move_assignable_v<nexora::ReadLease> &&
              !std::is_copy_constructible_v<nexora::ReadLease>);
static_assert(std::is_nothrow_move_constructible_v<nexora::WriteLease> &&
              std::is_nothrow_move_assignable_v<nexora::WriteLease> &&
              !std::is_copy_constructible_v<nexora::WriteLease>);

using Transport = nexora::TransportView<nexora::real>;
using Values = std::remove_cvref_t<decltype(std::declval<const Transport&>().values())>;
using StableIds = std::remove_cvref_t<decltype(std::declval<const Transport&>().stable_ids())>;

static_assert(std::is_move_constructible_v<Transport> && !std::is_copy_constructible_v<Transport>);
static_assert(
    std::is_same_v<Values, Kokkos::View<const ExpectedReal**, Kokkos::LayoutRight, Kokkos::HostSpace>>);
static_assert(StableIds::rank() == 1 && std::is_same_v<StableIds::value_type, const std::uint64_t>);
static_assert(std::is_same_v<decltype(std::declval<const Transport&>().purpose()),
                             nexora::TransportPurpose>);
static_assert(noexcept(std::declval<Transport&>().complete()));
