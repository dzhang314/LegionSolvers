#ifndef LEGION_SOLVERS_METAPROGRAMMING_UTILITIES_HPP
#define LEGION_SOLVERS_METAPROGRAMMING_UTILITIES_HPP

#include <string>


namespace LegionSolvers {


    template <typename... TS>
    struct TypeList {};


    template <int... NS>
    struct IntList {};


    template <typename T>
    struct ToString;

    template <>
    struct ToString<IntList<>> {
        static std::string value() { return std::string{""}; }
    };

    template <int N, int... NS>
    struct ToString<IntList<N, NS...>> {
        static std::string value() {
            return std::string{"_"} + std::to_string(N) +
                   ToString<IntList<NS...>>::value();
        }
    };


    template <typename L>
    struct ListLength;

    template <>
    struct ListLength<IntList<>> {
        static constexpr int value = 0;
    };

    template <int N, int... NS>
    struct ListLength<IntList<N, NS...>> {
        static constexpr int value = 1 + ListLength<IntList<NS...>>::value;
    };

    template <>
    struct ListLength<TypeList<>> {
        static constexpr int value = 0;
    };

    template <typename T, typename... TS>
    struct ListLength<TypeList<T, TS...>> {
        static constexpr int value = 1 + ListLength<TypeList<TS...>>::value;
    };


    template <typename L, typename T>
    struct ListIndex;

    template <typename L, typename... LS, typename T>
    struct ListIndex<TypeList<L, LS...>, T> {
        static constexpr int value = 1 + ListIndex<TypeList<LS...>, T>::value;
    };

    template <typename L, typename... LS>
    struct ListIndex<TypeList<L, LS...>, L> {
        static constexpr int value = 0;
    };


    template <typename T, int N>
    struct NestedPointer {
        typedef typename NestedPointer<T, N - 1>::type *type;
    };

    template <typename T>
    struct NestedPointer<T, 0> {
        typedef T type;
    };


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_METAPROGRAMMING_UTILITIES_HPP
