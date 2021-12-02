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

        static void task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct FillCOONegativeLaplacian1DTask


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


    template <typename ENTRY_T,
              typename VECTOR_COORD_T, typename KERNEL_COORD_T,
              int KERNEL_COLOR_DIM, typename KERNEL_COLOR_COORD_T>
    DistributedCOOMatrixT<
        ENTRY_T, 1, 1, 1, KERNEL_COLOR_DIM, KERNEL_COORD_T,
        VECTOR_COORD_T, VECTOR_COORD_T, KERNEL_COLOR_COORD_T
    > negative_laplacian_1d_coo(
        Legion::IndexSpaceT<1, VECTOR_COORD_T> index_space,
        Legion::IndexSpaceT<KERNEL_COLOR_DIM, KERNEL_COLOR_COORD_T>
            matrix_color_space,
        Legion::Context ctx, Legion::Runtime *rt
    ) {
        constexpr int VECTOR_DIM = 1;
        constexpr int KERNEL_DIM = 1;

        const Legion::DomainT<VECTOR_DIM, VECTOR_COORD_T> domain =
            rt->get_index_space_domain(ctx, index_space);

        const VECTOR_COORD_T grid_size = domain.volume();

        const Legion::Rect<VECTOR_DIM, VECTOR_COORD_T>
        vector_rect{0, grid_size - 1};

        assert(domain.bounds == vector_rect);

        const KERNEL_COORD_T kernel_size = laplacian_1d_kernel_size(
            static_cast<KERNEL_COORD_T>(grid_size)
        );

        const Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>
        kernel_rect{0, kernel_size - 1};

        const Legion::IndexSpaceT<KERNEL_DIM, KERNEL_COORD_T> matrix_index_space =
            rt->create_index_space(ctx, kernel_rect);

        DistributedCOOMatrixT<
            ENTRY_T, KERNEL_DIM, VECTOR_DIM, VECTOR_DIM, KERNEL_COLOR_DIM,
            KERNEL_COORD_T, VECTOR_COORD_T, VECTOR_COORD_T, KERNEL_COLOR_COORD_T
        > coo_matrix{
            "negative_laplacian_1d_coo", matrix_index_space,
            index_space, index_space, matrix_color_space, ctx, rt
        };

        const typename FillCOONegativeLaplacian1DTask<ENTRY_T>::Args args{
            coo_matrix.fid_i, coo_matrix.fid_j, coo_matrix.fid_entry, grid_size
        };

        Legion::TaskLauncher launcher{
            FillCOONegativeLaplacian1DTask<ENTRY_T>::task_id,
            Legion::TaskArgument{&args, sizeof(args)}
        };

        launcher.add_region_requirement(Legion::RegionRequirement{
            coo_matrix.kernel_region, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
            coo_matrix.kernel_region
        });

        launcher.add_field(0, coo_matrix.fid_i);
        launcher.add_field(0, coo_matrix.fid_j);
        launcher.add_field(0, coo_matrix.fid_entry);

        rt->execute_task(ctx, launcher);
        return coo_matrix;
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
