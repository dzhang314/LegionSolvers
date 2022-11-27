#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP_INCLUDED
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "COOMatrix.hpp"       // for COOMatrix
#include "CSRMatrix.hpp"       // for CSRMatrix
#include "LegionUtilities.hpp" // for TaskFlags
#include "TaskBaseClasses.hpp" // for TaskTDDDIII
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <typename ENTRY_T>
COOMatrix<ENTRY_T> coo_negative_laplacian_1d(
    Legion::Context ctx,
    Legion::Runtime *rt,
    Legion::coord_t grid_size,
    Legion::IndexSpace launch_space
);


template <typename ENTRY_T>
CSRMatrix<ENTRY_T> csr_negative_laplacian_1d(
    Legion::Context ctx,
    Legion::Runtime *rt,
    Legion::coord_t grid_size,
    Legion::IndexSpace launch_space
);


template <typename T>
constexpr T laplacian_2d_kernel_size(T height, T width) {
    return (
        4 * 2 +                          // four corners
        (height - 2) * 2 * 3 +           // vertical edges
        (width - 2) * 2 * 3 +            // horizontal edges
        (height - 2) * (width - 2) * 4 + // grid interior
        width * height                   // self-interaction
    );
}


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
struct FillCOONegativeLaplacianTask
    : public TaskTDDDIII<
          FILL_COO_NEGATIVE_LAPLACIAN_TASK_BLOCK_ID,
          FillCOONegativeLaplacianTask,
          ENTRY_T,
          KERNEL_DIM,
          DOMAIN_DIM,
          RANGE_DIM,
          KERNEL_COORD_T,
          DOMAIN_COORD_T,
          RANGE_COORD_T> {

    static constexpr const char *task_base_name = "fill_coo_negative_laplacian";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_row;
        Legion::FieldID fid_col;
        KERNEL_COORD_T grid_shape[KERNEL_DIM];
    };

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct FillCOONegativeLaplacianTask


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
struct FillCSRNegativeLaplacianTask
    : public TaskTDDDIII<
          FILL_CSR_NEGATIVE_LAPLACIAN_TASK_BLOCK_ID,
          FillCSRNegativeLaplacianTask,
          ENTRY_T,
          KERNEL_DIM,
          DOMAIN_DIM,
          RANGE_DIM,
          KERNEL_COORD_T,
          DOMAIN_COORD_T,
          RANGE_COORD_T> {

    static constexpr const char *task_base_name = "fill_csr_negative_laplacian";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_col;
        KERNEL_COORD_T grid_shape[KERNEL_DIM];
    };

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct FillCSRNegativeLaplacianTask


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
struct FillCSRNegativeLaplacianRowptrTask
    : public TaskTDDDIII<
          FILL_CSR_NEGATIVE_LAPLACIAN_ROWPTR_TASK_BLOCK_ID,
          FillCSRNegativeLaplacianRowptrTask,
          ENTRY_T,
          KERNEL_DIM,
          DOMAIN_DIM,
          RANGE_DIM,
          KERNEL_COORD_T,
          DOMAIN_COORD_T,
          RANGE_COORD_T> {

    static constexpr const char *task_base_name =
        "fill_csr_negative_laplacian_rowptr";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    struct Args {
        Legion::FieldID fid_rowptr;
        KERNEL_COORD_T grid_shape[KERNEL_DIM];
    };

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct FillCSRNegativeLaplacianRowptrTask


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP_INCLUDED
