#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <legion.h>

#include "DistributedCOOMatrix.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T>
    constexpr T laplacian_1d_kernel_size(T length) {
        return 3 * length - 2;
    }


    template <typename T>
    constexpr T laplacian_2d_kernel_size(T height, T width) {
        return (4 * 2 +                          // four corners
                (height - 2) * 2 * 3 +           // vertical edges
                (width - 2) * 2 * 3 +            // horizontal edges
                (height - 2) * (width - 2) * 4 + // grid interior
                width * height);                 // self-interaction
    }


    template <typename ExecutionSpace, typename T>
    struct KokkosFillCOONegativeLaplacian1DFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, Legion::Point<1>, 1> i_view;
        const KokkosMutableOffsetView<ExecutionSpace, Legion::Point<1>, 1> j_view;
        const KokkosMutableOffsetView<ExecutionSpace, T, 1> entry_view;

        explicit KokkosFillCOONegativeLaplacian1DFunctor(
            Realm::AffineAccessor<Legion::Point<1>, 1, Legion::coord_t> i_accessor,
            Realm::AffineAccessor<Legion::Point<1>, 1, Legion::coord_t> j_accessor,
            Realm::AffineAccessor<T, 1, Legion::coord_t> entry_accessor
        ) : i_view(i_accessor), j_view(j_accessor), entry_view(entry_accessor) {}

        KOKKOS_INLINE_FUNCTION void operator()(int k) const {
            i_view(k) = Legion::Point<1>{(k + 1) / 3};
            j_view(k) = Legion::Point<1>{k - 2 * ((k + 1) / 3)};
            entry_view(k) = (k % 3) ? -1.0 : +2.0;
        }

    }; // struct KokkosFillCOONegativeLaplacian1DFunctor


    template <typename T>
    struct FillCOONegativeLaplacian1DTask : public TaskT<
        FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID,
        FillCOONegativeLaplacian1DTask, T
    > {

        static constexpr const char *
        task_base_name = "fill_coo_negative_laplacian_1d";

        static constexpr bool is_inner = false;

        static constexpr bool is_leaf = true;

        struct Args {
            Legion::FieldID fid_i;
            Legion::FieldID fid_j;
            Legion::FieldID fid_entry;
            Legion::coord_t grid_length;
        };

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct FillCOONegativeLaplacian1DTask


    template <typename ExecutionSpace, typename T>
    struct KokkosFillCSRNegativeLaplacian1DFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, Legion::Point<1>, 1> col_view;
        const KokkosMutableOffsetView<ExecutionSpace, T, 1> entry_view;

        explicit KokkosFillCSRNegativeLaplacian1DFunctor(
            Realm::AffineAccessor<Legion::Point<1>, 1, Legion::coord_t> col_accessor,
            Realm::AffineAccessor<T, 1, Legion::coord_t> entry_accessor
        ) : col_view(col_accessor), entry_view(entry_accessor) {}

        KOKKOS_INLINE_FUNCTION void operator()(int k) const {
            col_view(k) = Legion::Point<1>{k - 2 * ((k + 1) / 3)};
            entry_view(k) = (k % 3) ? -1.0 : +2.0;
        }

    }; // struct KokkosFillCSRNegativeLaplacian1DFunctor


    template <typename T>
    struct FillCSRNegativeLaplacian1DTask : public TaskT<
        FILL_CSR_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID,
        FillCSRNegativeLaplacian1DTask, T
    > {

        static constexpr const char *
        task_base_name = "fill_csr_negative_laplacian_1d";

        static constexpr bool is_inner = false;

        static constexpr bool is_leaf = true;

        struct Args {
            Legion::FieldID fid_col;
            Legion::FieldID fid_entry;
            Legion::coord_t grid_length;
        };

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct FillCSRNegativeLaplacian1DTask


    template <typename ExecutionSpace, typename T>
    struct KokkosFillCSRNegativeLaplacian1DRowptrFunctor {

        const KokkosMutableOffsetView<ExecutionSpace, Legion::Rect<1>, 1> rowptr_view;
        Legion::coord_t grid_length;

        explicit KokkosFillCSRNegativeLaplacian1DRowptrFunctor(
            Realm::AffineAccessor<Legion::Rect<1>, 1, Legion::coord_t> rowptr_accessor,
            Legion::coord_t grid_length
        ) : rowptr_view(rowptr_accessor), grid_length(grid_length) {}

        KOKKOS_INLINE_FUNCTION void operator()(int k) const {
            if (k == 0) {
                rowptr_view(k) = Legion::Rect<1>{0, 1};
            } else if (k == (grid_length - 1)) {
                rowptr_view(k) = Legion::Rect<1>{3 * grid_length - 4,
                                                 3 * grid_length - 3};
            } else {
                rowptr_view(k) = Legion::Rect<1>{3 * k - 1, 3 * k + 1};
            }
        }

    }; // struct KokkosFillCSRNegativeLaplacian1DRowptrFunctor


    template <typename T>
    struct FillCSRNegativeLaplacian1DRowptrTask : public TaskT<
        FILL_CSR_NEGATIVE_LAPLACIAN_1D_ROWPTR_TASK_BLOCK_ID,
        FillCSRNegativeLaplacian1DRowptrTask, T
    > {

        static constexpr const char *
        task_base_name = "fill_csr_negative_laplacian_1d_rowptr";

        static constexpr bool is_inner = false;

        static constexpr bool is_leaf = true;

        struct Args {
            Legion::FieldID fid_rowptr;
            Legion::coord_t grid_length;
        };

        using return_type = void;

        template <typename KokkosExecutionSpace>
        struct KokkosTaskTemplate {

            static void task_body(
                const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *rt
            );

        }; // struct KokkosTaskTemplate

    }; // struct FillCSRNegativeLaplacian1DRowptrTask


    template <typename T>
    struct FillCOONegativeLaplacian2DTask : public TaskT<
        FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID,
        FillCOONegativeLaplacian2DTask, T
    > {

        static constexpr const char *
        task_base_name = "fill_coo_negative_laplacian_2d";

        static constexpr bool is_inner = false;

        static constexpr bool is_leaf = true;

        struct Args {
            Legion::FieldID fid_i;
            Legion::FieldID fid_j;
            Legion::FieldID fid_entry;
            Legion::coord_t grid_height;
            Legion::coord_t grid_width;
        };

        using return_type = void;

        static void task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct FillCOONegativeLaplacian2DTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
