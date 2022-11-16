#ifndef LEGION_SOLVERS_COO_MATRIX_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_COO_MATRIX_TASKS_HPP_INCLUDED

#include "LegionUtilities.hpp" // for TaskFlags, LEGION_SOLVERS_KDR_TEMPLATE
#include "TaskBaseClasses.hpp" // for TaskTDDDIII
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


LEGION_SOLVERS_KDR_TEMPLATE
struct COOMatvecTask : public TaskTDDDIII<
                           COO_MATVEC_TASK_BLOCK_ID,
                           COOMatvecTask,
                           LEGION_SOLVERS_KDR_TEMPLATE_ARGS> {
    static constexpr const char *task_base_name = "coo_matvec";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_row;
        Legion::FieldID fid_col;
    };
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


LEGION_SOLVERS_KDR_TEMPLATE
struct COORmatvecTask : public TaskTDDDIII<
                            COO_RMATVEC_TASK_BLOCK_ID,
                            COORmatvecTask,
                            LEGION_SOLVERS_KDR_TEMPLATE_ARGS> {
    static constexpr const char *task_base_name = "coo_rmatvec";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_row;
        Legion::FieldID fid_col;
    };
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_COO_MATRIX_TASKS_HPP_INCLUDED
