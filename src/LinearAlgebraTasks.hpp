#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP

#include <Kokkos_Core.hpp>
#include <legion.h>

#include "KokkosUtilities.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename ExecutionSpace, typename T, int N>
    struct KokkosAxpyFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, T, N> y_view;
        const T alpha;
        const KokkosConstOffsetView<ExecutionSpace, T, N> x_view;

        explicit KokkosAxpyFunctor(
            Realm::AffineAccessor<T, N, Legion::coord_t> y_accessor,
            const T &a,
            Realm::AffineAccessor<T, N, Legion::coord_t> x_accessor
        ) : y_view{y_accessor}, alpha{a}, x_view{x_accessor} {}

        KOKKOS_INLINE_FUNCTION void operator()(int a) const {
            y_view(a) += alpha * x_view(a);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b) const {
            y_view(a, b) += alpha * x_view(a, b);
        }

        KOKKOS_INLINE_FUNCTION void operator()(int a, int b, int c) const {
            y_view(a, b, c) += alpha * x_view(a, b, c);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d
        ) const {
            y_view(a, b, c, d) += alpha * x_view(a, b, c, d);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e
        ) const {
            y_view(a, b, c, d, e) += alpha * x_view(a, b, c, d, e);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f
        ) const {
            y_view(a, b, c, d, e, f) += alpha * x_view(a, b, c, d, e, f);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e, int f, int g
        ) const {
            y_view(a, b, c, d, e, f, g) += alpha * x_view(a, b, c, d, e, f, g);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d,
            int e, int f, int g, int h
        ) const {
            y_view(a, b, c, d, e, f, g, h) +=
                alpha * x_view(a, b, c, d, e, f, g, h);
        }

        KOKKOS_INLINE_FUNCTION void operator()(
            int a, int b, int c, int d, int e,
            int f, int g, int h, int i
        ) const {
            y_view(a, b, c, d, e, f, g, h, i) +=
                alpha * x_view(a, b, c, d, e, f, g, h, i);
        }

    }; // struct KokkosAxpyFunctor


    template <typename T, int N>
    struct AxpyTask : TaskTD<AXPY_TASK_BLOCK_ID, AxpyTask, T, N> {

        static constexpr const char *task_base_name = "axpy";

        static constexpr bool is_leaf = true;

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskBody

    }; // struct AxpyTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP
