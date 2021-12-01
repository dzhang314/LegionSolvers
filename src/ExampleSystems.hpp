#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    constexpr Legion::coord_t
    laplacian_1d_kernel_size(Legion::coord_t length) {
        return 3 * length - 2;
    }


    constexpr Legion::coord_t
    laplacian_2d_kernel_size(Legion::coord_t height, Legion::coord_t width) {
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
