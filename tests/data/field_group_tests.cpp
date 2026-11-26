#include <Kokkos_Core.hpp>
#include <array>
#include <barrier>
#include <data/FieldIssuer.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace nexora;
using nexora::detail::FieldAxis;
using nexora::detail::FieldBindingSpec;
using nexora::detail::FieldIssuer;
using Values = Kokkos::View<real*, Kokkos::Serial>;

FieldDescriptor descriptor(FieldAccess access = FieldAccess::read) {
    FieldDescriptor field;
    field.name = "values";
    field.association = Association::node;
    field.units = "1";
    field.access = access;
    return field;
}

template <class View> auto read_request(const FieldIssuer& issuer, const View& view) {
    const std::array axes{FieldAxis{view.extent(0)}};
    return issuer.request_read(view, descriptor(), issuer.stamp(), {Association::node, "1", axes});
}

template <class View>
auto write_request(const FieldIssuer& issuer, const View& view,
                   FieldAccess access = FieldAccess::overwrite) {
    const std::array axes{FieldAxis{view.extent(0)}};
    return issuer.request_write(view, descriptor(access), issuer.stamp(),
                                {Association::node, "1", axes}, access);
}

template <class Request> concept ExposesAuthority = requires(Request request) {
    request.value().authority;
};
template <class Request> concept ExposesAccess = requires(Request request) {
    request.value().access;
};
template <class Request> concept ExposesBinding = requires(Request request) {
    request.value().binding;
};
using ReadRequest =
    decltype(read_request(std::declval<const FieldIssuer&>(), std::declval<const Values&>()));
using WriteRequest =
    decltype(write_request(std::declval<const FieldIssuer&>(), std::declval<const Values&>()));
static_assert(!ExposesAuthority<ReadRequest> && !ExposesAuthority<WriteRequest>);
static_assert(!ExposesAccess<ReadRequest> && !ExposesAccess<WriteRequest>);
static_assert(!ExposesBinding<ReadRequest> && !ExposesBinding<WriteRequest>);

template <class T> T take(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

TEST(FieldGroup, AliasLifetime) {
    const Values values("shared allocation", 4);
    const auto left = Kokkos::subview(values, Kokkos::pair{0, 2});
    const auto right = Kokkos::subview(values, Kokkos::pair{2, 4});
    using Subview = std::remove_cv_t<decltype(left)>;
    const auto issuer = take(FieldIssuer::create_at_validated_time(values, FieldStamp{}));

    std::optional<ReadField<Subview>> retained_reader;
    {
        auto bindings = take(FieldIssuer::grant(
            read_request(issuer, left), read_request(issuer, right), read_request(issuer, left)));
        static_assert(
            std::is_same_v<std::tuple_element_t<0, decltype(bindings)>, ReadField<Subview>>);
        ASSERT_EQ(std::get<0>(bindings).view().data(), left.data());
        ASSERT_EQ(std::get<1>(bindings).view().data(), right.data());
        ASSERT_EQ(std::get<2>(bindings).view().data(), left.data());
        retained_reader.emplace(std::move(std::get<1>(bindings)));
    }
    const std::array axes{FieldAxis{4}};
    const FieldBindingSpec spec{Association::node, "1", axes};
    {
        auto second_reader = issuer.read(values, descriptor(), issuer.stamp(), spec);
        ASSERT_TRUE(second_reader.ok());
        ASSERT_FALSE(FieldIssuer::grant(write_request(issuer, left)).ok());
    }
    ASSERT_FALSE(FieldIssuer::grant(write_request(issuer, left)).ok());
    retained_reader.reset();

    for (const auto mode : {FieldAccess::overwrite, FieldAccess::accumulate}) {
        std::optional<WriteField<Subview>> retained_writer;
        {
            auto bindings = take(FieldIssuer::grant(write_request(issuer, left, mode),
                                                    write_request(issuer, right, mode)));
            ASSERT_EQ(std::get<0>(bindings).view().data(), left.data());
            ASSERT_EQ(std::get<1>(bindings).view().data(), right.data());
            retained_writer.emplace(std::move(std::get<1>(bindings)));
        }
        ASSERT_FALSE(issuer.read(values, descriptor(), issuer.stamp(), spec).ok());
        retained_writer.reset();
        ASSERT_TRUE(issuer.read(values, descriptor(), issuer.stamp(), spec).ok());
    }

    auto mixed = FieldIssuer::grant(read_request(issuer, left), write_request(issuer, right));
    ASSERT_EQ(mixed.status().code(), StatusCode::invalid_argument);
    auto incompatible_writes = FieldIssuer::grant(
        write_request(issuer, left), write_request(issuer, right, FieldAccess::accumulate));
    ASSERT_EQ(incompatible_writes.status().code(), StatusCode::invalid_argument);
    ASSERT_TRUE(FieldIssuer::grant(write_request(issuer, values)).ok());
}

TEST(FieldGroup, FailedGroupLeavesGrantsUnchanged) {
    const Values first("first allocation", 2);
    const Values second("second allocation", 2);
    const auto first_issuer = take(FieldIssuer::create_at_validated_time(first, FieldStamp{}));
    auto second_issuer = take(FieldIssuer::create_at_validated_time(second, FieldStamp{}));
    ASSERT_LT(first_issuer.stamp().storage_incarnation, second_issuer.stamp().storage_incarnation);
    {
        const auto existing_reader = FieldIssuer::grant(read_request(second_issuer, second));
        ASSERT_TRUE(existing_reader.ok());
        const auto conflict = FieldIssuer::grant(write_request(second_issuer, second),
                                                 write_request(first_issuer, first));
        ASSERT_EQ(conflict.status().code(), StatusCode::invalid_state);
        ASSERT_TRUE(FieldIssuer::grant(write_request(first_issuer, first)).ok());
        ASSERT_FALSE(FieldIssuer::grant(write_request(second_issuer, second)).ok());
    }

    const std::array axes{FieldAxis{2}};
    const FieldBindingSpec spec{Association::node, "1", axes};
    auto stale = second_issuer.request_read(second, descriptor(), first_issuer.stamp(), spec);
    auto wrong_shape = first_issuer.request_read(first, descriptor(), first_issuer.stamp(),
                                                 {Association::node, "1", {}});
    auto invalid_group = FieldIssuer::grant(write_request(first_issuer, first), std::move(stale),
                                            std::move(wrong_shape));
    ASSERT_EQ(invalid_group.status().code(), StatusCode::invalid_state);
    ASSERT_TRUE(FieldIssuer::grant(write_request(first_issuer, first)).ok());

    auto pending = read_request(second_issuer, second);
    auto retained = std::move(pending);
    auto moved_from = FieldIssuer::grant(write_request(first_issuer, first), std::move(pending));
    ASSERT_EQ(moved_from.status().code(), StatusCode::invalid_state);
    ASSERT_TRUE(FieldIssuer::grant(write_request(first_issuer, first)).ok());
    auto already_granted = FieldIssuer::grant(read_request(second_issuer, second));
    ASSERT_TRUE(already_granted.ok());
    second_issuer.close();
    auto closed = FieldIssuer::grant(write_request(first_issuer, first), std::move(retained));
    ASSERT_EQ(closed.status().code(), StatusCode::invalid_state);
    ASSERT_EQ(std::get<0>(already_granted.value()).view().data(), second.data());
    ASSERT_TRUE(FieldIssuer::grant(write_request(first_issuer, first)).ok());
}

TEST(FieldGroup, ConcurrentGroupsUseOneLockOrder) {
    const Values first("concurrent first allocation", 2);
    const Values second("concurrent second allocation", 2);
    const auto first_issuer = take(FieldIssuer::create_at_validated_time(first, FieldStamp{}));
    const auto second_issuer = take(FieldIssuer::create_at_validated_time(second, FieldStamp{}));
    constexpr unsigned attempts = 64;
    using Requests = std::vector<std::pair<WriteRequest, WriteRequest>>;
    auto prepare = [&](const FieldIssuer& a, const Values& a_view, const FieldIssuer& b,
                       const Values& b_view) {
        Requests requests;
        requests.reserve(attempts);
        for (unsigned iteration = 0; iteration < attempts; ++iteration)
            requests.emplace_back(write_request(a, a_view), write_request(b, b_view));
        return requests;
    };
    auto ascending_requests = prepare(first_issuer, first, second_issuer, second);
    auto descending_requests = prepare(second_issuer, second, first_issuer, first);
    std::array<StatusCode, attempts> ascending_status{};
    std::array<StatusCode, attempts> descending_status{};
    std::barrier rendezvous(2);
    auto compete = [&](Requests& requests, std::array<StatusCode, attempts>& statuses) {
        for (unsigned iteration = 0; iteration < attempts; ++iteration) {
            auto& [a_request, b_request] = requests[iteration];
            rendezvous.arrive_and_wait();
            const auto bindings = FieldIssuer::grant(std::move(a_request), std::move(b_request));
            statuses[iteration] = bindings.status().code();
            rendezvous.arrive_and_wait();
        }
    };
    std::jthread ascending([&] { compete(ascending_requests, ascending_status); });
    std::jthread descending([&] { compete(descending_requests, descending_status); });
    ascending.join();
    descending.join();
    for (unsigned iteration = 0; iteration < attempts; ++iteration) {
        SCOPED_TRACE(iteration);
        if (ascending_status[iteration] == StatusCode::success) {
            EXPECT_EQ(descending_status[iteration], StatusCode::invalid_state);
        } else {
            EXPECT_EQ(ascending_status[iteration], StatusCode::invalid_state);
            EXPECT_EQ(descending_status[iteration], StatusCode::success);
        }
    }
}
} // namespace
