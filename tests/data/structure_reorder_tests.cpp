#ifdef NDEBUG
#undef NDEBUG
#endif
#include "structure_fixture.hpp"

namespace {
using namespace nexora::test_fixture;
void test_configuration(const Kokkos::Serial& exec) {
    auto config = fixture_config();
    config.q.pop_back();
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.fields[2].entity_set = "different";
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.trial.attempt = 1;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.storage_incarnation = 1;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.stamp.accepted = false;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.fields[2].components = 2;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.fields[0].scalar_type = real_scalar_type;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.indices[0] = 4;
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.ids[1] = config.ids[0];
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    config.history[0] = std::numeric_limits<real>::quiet_NaN();
    assert(!StructureState::create(exec, config).ok());
    config = fixture_config();
    auto state = require(StructureState::create(exec, config));
    config.ids[0] = GlobalEntityId{999};
    config.x[0] = config.q[0] = config.history[0] = real{999};
    config.indices[0] = 3;
    expect_bank(*state->snapshot(), false);
}

void test_reorder(const Kokkos::Serial& exec) {
    auto state = require(StructureState::create(exec, fixture_config()));
    const auto original = state->snapshot();
    expect_bank(*original, false);
    const auto packed = pack_bank(exec, *original);
    const auto old_stamp = original->q.grant.stamp();
    auto unchanged = [&] {
        assert(state->snapshot() == original);
        expect_bank(*state->snapshot(), false);
        assert(state->snapshot()->q.grant.stamp() == old_stamp);
    };
    assert(!state->reorder(std::array{GlobalEntityId{10}, GlobalEntityId{10},
                                     GlobalEntityId{20}, GlobalEntityId{40}}).ok());
    unchanged();
    assert(!state->reorder(std::array{GlobalEntityId{10}, GlobalEntityId{30},
                                     GlobalEntityId{20}, GlobalEntityId{99}}).ok());
    unchanged();
    assert(!state->reorder(reordered_ids, StructureState::Fault::after_partial_candidate).ok());
    unchanged();
    assert(!state->reorder(reordered_ids, StructureState::Fault::before_publish).ok());
    unchanged();
    // The failed pre-publication bank did not close the original issuers.
    const auto still_packable = pack_bank(exec, *original);
    assert(still_packable.values()(0, 0) == real{100});
    assert(state->reorder(reordered_ids).ok());
    const auto reordered = state->snapshot();
    expect_bank(*reordered, true);
    expect_bank(*original, false);
    auto expected_stamp = original->logical;
    expected_stamp.layout = LayoutVersion{8};
    assert(reordered->logical == expected_stamp);
    assert(reordered->q.grant.stamp().storage_incarnation > old_stamp.storage_incarnation);
    assert(reordered->x.values.data() != original->x.values.data());
    assert(reordered->q.values.data() != original->q.values.data());
    assert(reordered->history.values.data() != original->history.values.data());
    const std::array axes{FieldAxis{4}};
    assert(!reordered->q.issuer.read(reordered->q.values, descriptor(3), old_stamp,
                                     {Association::particle, "1", axes}).ok());
    assert(!original->q.issuer.read(original->q.values, descriptor(3), old_stamp,
                                    {Association::particle, "1", axes}).ok());
    assert(!original->order.validate_for(reordered->order.entity_namespace(),
                                          reordered->order.layout()).ok());
    assert(!original->adjacency.validate_for(reordered->order, reordered->order,
                                              reordered->logical.topology).ok());
    receive_reordered(exec, packed, *reordered);
    assert(state->reorder(original_ids).ok());
    const auto inverse = state->snapshot();
    expect_bank(*inverse, false);
    expected_stamp.layout = LayoutVersion{9};
    assert(inverse->logical == expected_stamp);
    expect_bank(*original, false);
    expect_bank(*reordered, true);
}
void test_next_commit_and_lifetime(const Kokkos::Serial& exec) {
    std::shared_ptr<const Bank> original, reordered, committed;
    {
        auto state = require(StructureState::create(exec, fixture_config()));
        original = state->snapshot();
        assert(state->reorder(reordered_ids).ok());
        reordered = state->snapshot();
        assert(state->commit_values(std::array<real, 4>{21, 11, 41, 31}).ok());
        committed = state->snapshot();
        expect_bank(*committed, true, true);
        auto expected_stamp = reordered->logical;
        expected_stamp.accepted_version = StateVersion{12};
        assert(committed->logical == expected_stamp);
        assert(committed->q.grant.stamp().storage_incarnation > reordered->q.grant.stamp().storage_incarnation);
        // Unchanged CSR remains a read-only shared allocation on value commit.
        assert(committed->adjacency.indices().data() == reordered->adjacency.indices().data());
    }
    expect_bank(*original, false);
    expect_bank(*reordered, true);
    expect_bank(*committed, true, true);
}
} // namespace
int main() {
    Kokkos::ScopeGuard guard;
    const Kokkos::Serial exec;
    test_configuration(exec);
    test_reorder(exec);
    test_next_commit_and_lifetime(exec);
    auto exhausted = require(StructureState::create(
        exec, fixture_config(LayoutVersion{std::numeric_limits<std::uint64_t>::max()})));
    const auto before = exhausted->snapshot();
    assert(!exhausted->reorder(reordered_ids).ok());
    assert(exhausted->snapshot() == before);
    expect_bank(*before, false);
}
