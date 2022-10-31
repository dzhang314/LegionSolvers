#ifndef LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED

#include "LegionUtilities.hpp" // for TaskFlags, LEGION_SOLVERS_KDR_TEMPLATE
#include "TaskBaseClasses.hpp" // for TaskTDDDIII
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


LEGION_SOLVERS_KDR_TEMPLATE
struct CSRMatvecTask : public TaskTDDDIII<
                           CSR_MATVEC_TASK_BLOCK_ID,
                           CSRMatvecTask,
                           LEGION_SOLVERS_KDR_TEMPLATE_ARGS> {
    static constexpr const char *task_base_name = "csr_matvec";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
#endif
};


LEGION_SOLVERS_KDR_TEMPLATE
struct CSRRmatvecTask : public TaskTDDDIII<
                            CSR_RMATVEC_TASK_BLOCK_ID,
                            CSRRmatvecTask,
                            LEGION_SOLVERS_KDR_TEMPLATE_ARGS> {
    static constexpr const char *task_base_name = "csr_rmatvec";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
#endif
};


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_CSR_MATRIX_TASKS_HPP_INCLUDED
