#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <nexora/exec/Buffer.hpp>
#include <nexora/exec/ExecContext.hpp>
#include <type_traits>
#include <utility>
template <class T, class = void> struct HasRvalueView : std::false_type {};
template <class T>
struct HasRvalueView<T, std::void_t<decltype(std::declval<T&&>().view())>> : std::true_type {};
int main() {
    using namespace nexora;
    static_assert(!std::is_copy_constructible<Buffer>::value, "buffer owns memory");
    static_assert(!HasRvalueView<Buffer>::value && !HasRvalueView<const Buffer>::value,
                  "temporary buffer cannot lend a view");
    Buffer a(8);
    a.view()[0] = std::byte{42};
    Buffer b(std::move(a));
    assert(a.size() == 0 && b.size() == 8 && b.view()[0] == std::byte{42});
    Buffer c;
    c = std::move(b);
    assert(b.size() == 0 && c.size() == 8);
    ExecContext ctx;
    int called = 0;
    auto ok = ctx.submit([&] {
        ++called;
        return Status{};
    });
    assert(ok.wait().ok() && called == 1);
    auto failed = ctx.submit([] { return Status{StatusCode::execution_failure, "failure"}; });
    auto blocked = ctx.submit(
        [&] {
            ++called;
            return Status{};
        },
        {ok, failed});
    assert(!blocked.wait().ok() && blocked.wait().code() == StatusCode::execution_failure &&
           called == 1);
}
