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

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct ScalTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct AxpyTask
    : public TaskTDI<AXPY_TASK_BLOCK_ID, ScalTask, ENTRY_T, DIM, COORD_T> {

    static constexpr const char *task_base_name = "axpy";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct AxpyTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct XpayTask
    : public TaskTDI<XPAY_TASK_BLOCK_ID, ScalTask, ENTRY_T, DIM, COORD_T> {

    static constexpr const char *task_base_name = "xpay";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = void;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct XpayTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct DotTask
    : public TaskTDI<DOT_TASK_BLOCK_ID, DotTask, ENTRY_T, DIM, COORD_T> {

    static constexpr const char *task_base_name = "dot_product";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = ENTRY_T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct DotTask


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED
