#ifndef LEGION_SOLVERS_KOKKOS_UTILITIES_HPP
#define LEGION_SOLVERS_KOKKOS_UTILITIES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

#include "MetaprogrammingUtilities.hpp"


namespace LegionSolvers {


    template <typename KokkosExecutionSpace, typename T, int N>
    using KokkosOffsetView = Kokkos::Experimental::OffsetView<
        typename NestedPointer<const T, N>::type,
        Kokkos::LayoutStride,
        typename KokkosExecutionSpace::memory_space
    >;


    template <typename KokkosExecutionSpace, int N>
    struct KokkosRangePolicyFactory {

        using ResultType = Kokkos::Experimental::MDRangePolicy<
            Kokkos::Experimental::Rank<N>,
            KokkosExecutionSpace
        >;

        static ResultType create(Legion::Rect<N> rect,
                                 Legion::Context ctx, Legion::Runtime *rt) {
            Kokkos::Array<typename ResultType::index_type, N>
            lower_bounds{}, upper_bounds{};
            for (int i = 0; i < N; ++i) {
                lower_bounds[i] = rect.lo[i];
                upper_bounds[i] = rect.hi[i] + 1;
            }
            return ResultType{
                rt->get_executing_processor(ctx).kokkos_work_space(),
                lower_bounds, upper_bounds
            };
        }

    }; // struct KokkosRangePolicyFactory


    template <typename KokkosExecutionSpace>
    struct KokkosRangePolicyFactory<KokkosExecutionSpace, 1> {

        using ResultType = Kokkos::RangePolicy<KokkosExecutionSpace>;

        static ResultType create(Legion::Rect<1> rect,
                                 Legion::Context ctx, Legion::Runtime *rt) {
            return ResultType{
                rt->get_executing_processor(ctx).kokkos_work_space(),
                static_cast<typename ResultType::index_type>(rect.lo.x),
                static_cast<typename ResultType::index_type>(rect.hi.x + 1)
            };
        }

    }; // struct KokkosRangePolicyFactory<KokkosExecutionSpace, 1>


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_KOKKOS_UTILITIES_HPP