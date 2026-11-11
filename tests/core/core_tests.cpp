#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <nexora/core/ArrayView.hpp>
#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Types.hpp>
#include <type_traits>
#include <vector>
struct Base {
    int value;
};
struct Derived : Base {
    int extra;
};
int main() {
    using namespace nexora;
#if NEXORA_ENABLE_DOUBLE_PRECISION
    static_assert(std::is_same_v<real, double>);
#else
    static_assert(std::is_same_v<real, float>);
#endif
    static_assert(!std::is_convertible<FieldId, DomainId>::value, "IDs must not mix");
    static_assert(!std::is_constructible<ArrayView<Base>, std::vector<Derived>&>::value,
                  "array covariance changes stride");
    static_assert(!std::is_constructible<ArrayView<const Base>, const std::vector<Derived>&>::value,
                  "const array covariance changes stride");
    static_assert(std::is_constructible<ArrayView<const int>, std::vector<int>&>::value,
                  "adding const is allowed");
    static_assert(!std::is_constructible<ArrayView<const int>, std::vector<int>&&>::value,
                  "view cannot borrow temporary");
    static_assert(!std::is_constructible<ArrayView<const int>, const std::vector<int>&&>::value,
                  "view cannot borrow const temporary");
    assert(checked_integer_cast<LocalIndex>(42).value() == 42);
    assert(!checked_integer_cast<LocalIndex>(-1).ok());
    assert(!checked_integer_cast<LocalIndex>(std::numeric_limits<std::uint64_t>::max()).ok());
    assert(!checked_integer_cast<std::int8_t>(128).ok());
    assert(!checked_integer_cast<std::int8_t>(-129).ok());
    bool threw = false;
    try {
        ArrayView<int> bad(nullptr, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::vector<int> v{1, 2};
    ArrayView<int> view(v);
    view[0] = 3;
    assert(v[0] == 3);
    threw = false;
    try {
        view.at(2);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    Status s;
    assert(s.ok());
    Result<int> result(Status{StatusCode::invalid_argument, "bad"});
    assert(!result.ok());
    threw = false;
    try {
        Result<int> invalid(Status{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
