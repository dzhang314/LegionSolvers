#ifndef LEGION_SOLVERS_KOKKOS_UTILITIES_HPP
#define LEGION_SOLVERS_KOKKOS_UTILITIES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>
#include <legion.h>

#include "MetaprogrammingUtilities.hpp"


namespace LegionSolvers {


    template <typename KokkosExecutionSpace, typename ENTRY_T, int DIM>
    using KokkosConstOffsetView = Kokkos::Experimental::OffsetView<
        typename NestedPointer<const ENTRY_T, DIM>::type,
        Kokkos::LayoutStride,
        typename KokkosExecutionSpace::memory_space
    >;


    template <typename KokkosExecutionSpace, typename ENTRY_T, int DIM>
    using KokkosMutableOffsetView = Kokkos::Experimental::OffsetView<
        typename NestedPointer<ENTRY_T, DIM>::type,
        Kokkos::LayoutStride,
        typename KokkosExecutionSpace::memory_space
    >;


    template <typename KokkosExecutionSpace, int DIM, typename COORD_T>
    struct KokkosRangeFactory {

        using ResultType = Kokkos::MDRangePolicy<
            Kokkos::Rank<DIM>,
            Kokkos::IndexType<COORD_T>,
            KokkosExecutionSpace
        >;

        static ResultType create(Legion::Context ctx, Legion::Runtime *rt,
                                 Legion::Rect<DIM, COORD_T> rect) {
            Kokkos::Array<typename ResultType::index_type, DIM>
            lower_bounds{}, upper_bounds{};
            for (int i = 0; i < DIM; ++i) {
                lower_bounds[i] = rect.lo[i];
                upper_bounds[i] = rect.hi[i] + 1;
            }
            return ResultType{
                rt->get_executing_processor(ctx).kokkos_work_space(),
                lower_bounds, upper_bounds
            };
        }

    }; // struct KokkosRangeFactory


    template <typename KokkosExecutionSpace, typename COORD_T>
    struct KokkosRangeFactory<KokkosExecutionSpace, 1, COORD_T> {

        using ResultType = Kokkos::RangePolicy<
            Kokkos::IndexType<COORD_T>,
            KokkosExecutionSpace
        >;

        static ResultType create(Legion::Context ctx, Legion::Runtime *rt,
                                 Legion::Rect<1, COORD_T> rect) {
            return ResultType{
                rt->get_executing_processor(ctx).kokkos_work_space(),
                static_cast<typename ResultType::index_type>(rect.lo.x),
                static_cast<typename ResultType::index_type>(rect.hi.x + 1)
            };
        }

    }; // struct KokkosRangeFactory<KokkosExecutionSpace, 1, COORD_T>


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_KOKKOS_UTILITIES_HPP
