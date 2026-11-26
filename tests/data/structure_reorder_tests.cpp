#include <gtest/gtest.h>
#include "structure_fixture.hpp"

namespace {
using namespace nexora::test_fixture;
TEST(StructureReorder, Configuration) {
    const Kokkos::Serial exec;
    auto config = fixture_config();
    config.q.pop_back();
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.fields[2].entity_set = "different";
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.trial.attempt = 1;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.storage_incarnation = 1;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.accepted = false;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.time.physical_time = std::numeric_limits<double>::quiet_NaN();
    const auto invalid_time = StructureState::create(exec, config);
    ASSERT_EQ(invalid_time.status().code(), StatusCode::invalid_argument);
    ASSERT_EQ(invalid_time.status().message(), "invalid field time or validity interval");
    config = fixture_config();
    config.stamp.valid_from = 1.0;
    config.stamp.valid_until = 2.0;
    const auto outside_time = StructureState::create(exec, config);
    ASSERT_EQ(outside_time.status().code(), StatusCode::invalid_argument);
    ASSERT_EQ(outside_time.status().message(), "field time is outside its validity interval");
    config = fixture_config();
    config.fields[2].components = 2;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.fields[0].scalar_type = real_scalar_type;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.indices[0] = 4;
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.ids[1] = config.ids[0];
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    config.history[0] = std::numeric_limits<real>::quiet_NaN();
    ASSERT_FALSE(StructureState::create(exec, config).ok());
    config = fixture_config();
    auto state = require(StructureState::create(exec, config));
    config.ids[0] = GlobalEntityId{999};
    config.x[0] = config.q[0] = config.history[0] = real{999};
    config.indices[0] = 3;
    expect_bank(*state->snapshot(), false);
}

TEST(StructureReorder, Reorder) {
    const Kokkos::Serial exec;
    auto state = require(StructureState::create(exec, fixture_config()));
    const auto original = state->snapshot();
    expect_bank(*original, false);
    const auto packed = pack_bank(exec, *original);
    const auto old_stamp = original->q.grant.stamp();
    auto unchanged = [&] {
        ASSERT_EQ(state->snapshot(), original);
        expect_bank(*state->snapshot(), false);
        ASSERT_EQ(state->snapshot()->q.grant.stamp(), old_stamp);
    };
    ASSERT_TRUE(!state->reorder(std::array{GlobalEntityId{10}, GlobalEntityId{10},
                                     GlobalEntityId{20}, GlobalEntityId{40}}).ok());
    ASSERT_NO_FATAL_FAILURE(unchanged());
    ASSERT_TRUE(!state->reorder(std::array{GlobalEntityId{10}, GlobalEntityId{30},
                                     GlobalEntityId{20}, GlobalEntityId{99}}).ok());
    ASSERT_NO_FATAL_FAILURE(unchanged());
    // Rejected real input leaves the original issuers open.
    const auto still_packable = pack_bank(exec, *original);
    ASSERT_EQ(still_packable.values()(0, 0), real{100});
    ASSERT_TRUE(state->reorder(reordered_ids).ok());
    const auto reordered = state->snapshot();
    expect_bank(*reordered, true);
    expect_bank(*original, false);
    auto expected_stamp = original->logical;
    expected_stamp.layout = LayoutVersion{8};
    ASSERT_EQ(reordered->logical, expected_stamp);
    ASSERT_GT(reordered->q.grant.stamp().storage_incarnation, old_stamp.storage_incarnation);
    ASSERT_NE(reordered->x.values.data(), original->x.values.data());
    ASSERT_NE(reordered->q.values.data(), original->q.values.data());
    ASSERT_NE(reordered->history.values.data(), original->history.values.data());
    const std::array axes{FieldAxis{4}};
    ASSERT_TRUE(!reordered->q.issuer.read(reordered->q.values, descriptor(3), old_stamp,
                                     {Association::particle, "1", axes}).ok());
    ASSERT_TRUE(!original->q.issuer.read(original->q.values, descriptor(3), old_stamp,
                                    {Association::particle, "1", axes}).ok());
    ASSERT_NE(original->order.layout(), reordered->order.layout());
    ASSERT_NE(original->adjacency.rows().layout(), reordered->order.layout());
    ASSERT_NE(original->adjacency.columns().layout(), reordered->order.layout());
    receive_reordered(exec, packed, *reordered);
    ASSERT_TRUE(state->reorder(original_ids).ok());
    const auto inverse = state->snapshot();
    expect_bank(*inverse, false);
    expected_stamp.layout = LayoutVersion{9};
    ASSERT_EQ(inverse->logical, expected_stamp);
    expect_bank(*original, false);
    expect_bank(*reordered, true);
}
TEST(StructureReorder, NextCommitAndLifetime) {
    const Kokkos::Serial exec;
    std::shared_ptr<const Bank> original, reordered, committed;
    {
        auto state = require(StructureState::create(exec, fixture_config()));
        original = state->snapshot();
        ASSERT_TRUE(state->reorder(reordered_ids).ok());
        reordered = state->snapshot();
        ASSERT_EQ(state->commit_values(std::array<real, 4>{21, 11,
                      std::numeric_limits<real>::quiet_NaN(), 31}).code(), StatusCode::invalid_argument);
        ASSERT_EQ(state->snapshot(), reordered);
        expect_bank(*reordered, true);
        const auto packed = pack_bank(exec, *reordered);
        ASSERT_EQ(packed.values()(0, 0), real{200});
        ASSERT_TRUE(state->commit_values(std::array<real, 4>{21, 11, 41, 31}).ok());
        committed = state->snapshot();
        expect_bank(*committed, true, true);
        auto expected_stamp = reordered->logical;
        expected_stamp.accepted_version = StateVersion{12};
        ASSERT_EQ(committed->logical, expected_stamp);
        ASSERT_GT(committed->q.grant.stamp().storage_incarnation, reordered->q.grant.stamp().storage_incarnation);
        ASSERT_NE(committed->order.ids().data(), reordered->order.ids().data());
        ASSERT_GT(committed->ids.grant.stamp().storage_incarnation,
                  reordered->ids.grant.stamp().storage_incarnation);
        ASSERT_EQ(committed->adjacency.offsets().data(), reordered->adjacency.offsets().data());
        // Unchanged CSR remains a read-only shared allocation on value commit.
        ASSERT_EQ(committed->adjacency.indices().data(), reordered->adjacency.indices().data());
    }
    expect_bank(*original, false);
    expect_bank(*reordered, true);
    expect_bank(*committed, true, true);
}

TEST(StructureReorder, ExhaustedLayoutVersionPreservesBank) {
    const Kokkos::Serial exec;
    auto exhausted = require(StructureState::create(
        exec, fixture_config(LayoutVersion{std::numeric_limits<std::uint64_t>::max()})));
    const auto before = exhausted->snapshot();
    ASSERT_FALSE(exhausted->reorder(reordered_ids).ok());
    ASSERT_EQ(exhausted->snapshot(), before);
    expect_bank(*before, false);
}
} // namespace
