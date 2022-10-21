#ifndef LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED

#include "LegionUtilities.hpp" // for TaskFlags
#include "TaskBaseClasses.hpp" // for TaskTDDDIII
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
struct CSRMatvecTask : public TaskTDDDIII<
                           CSR_MATVEC_TASK_BLOCK_ID,
                           CSRMatvecTask,
                           ENTRY_T,
                           KERNEL_DIM,
                           DOMAIN_DIM,
                           RANGE_DIM,
                           KERNEL_COORD_T,
                           DOMAIN_COORD_T,
                           RANGE_COORD_T> {

    static constexpr const char *task_base_name = "csr_matvec";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct CSRMatvecTask


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
struct CSRRmatvecTask : public TaskTDDDIII<
                            CSR_RMATVEC_TASK_BLOCK_ID,
                            CSRRmatvecTask,
                            ENTRY_T,
                            KERNEL_DIM,
                            DOMAIN_DIM,
                            RANGE_DIM,
                            KERNEL_COORD_T,
                            DOMAIN_COORD_T,
                            RANGE_COORD_T> {

    static constexpr const char *task_base_name = "csr_rmatvec";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct CSRRmatvecTask


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED
