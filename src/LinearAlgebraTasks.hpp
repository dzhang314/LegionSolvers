#ifndef LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED

#include "LegionUtilities.hpp" // for TaskFlags
#include "TaskBaseClasses.hpp" // for TaskTDI
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <typename ENTRY_T, int DIM, typename COORD_T>
struct ScalTask
    : public TaskTDI<SCAL_TASK_BLOCK_ID, ScalTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "scal";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct AxpyTask
    : public TaskTDI<AXPY_TASK_BLOCK_ID, AxpyTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "axpy";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct XpayTask
    : public TaskTDI<XPAY_TASK_BLOCK_ID, XpayTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "xpay";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(void);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct DotTask
    : public TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T> {
    static constexpr const char *task_base_name = "dot_product";
    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(ENTRY_T);
    LEGION_SOLVERS_DECLARE_CUDA_TASK;
};


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED
