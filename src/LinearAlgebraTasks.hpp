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

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    static return_type gpu_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
#endif

}; // struct ScalTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct AxpyTask
    : public TaskTDI<AXPY_TASK_BLOCK_ID, AxpyTask, ENTRY_T, DIM, COORD_T> {

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

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    static return_type gpu_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
#endif

}; // struct AxpyTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct XpayTask
    : public TaskTDI<XPAY_TASK_BLOCK_ID, XpayTask, ENTRY_T, DIM, COORD_T> {

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

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    static return_type gpu_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
#endif

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

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    static return_type gpu_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
#endif

}; // struct DotTask

template <typename ENTRY_T>
inline ENTRY_T get_alpha(const std::vector<Legion::Future> &futures) {
    if (futures.size() == 0) {
        return static_cast<ENTRY_T>(1);
    } else if (futures.size() == 1) {
        return futures[0].get_result<ENTRY_T>();
    } else if (futures.size() == 2) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        return f0 / f1;
    } else if (futures.size() == 3) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        return f0 * f1 / f2;
    } else if (futures.size() == 4) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        const ENTRY_T f3 = futures[3].get_result<ENTRY_T>();
        return f0 * f1 / (f2 * f3);
    } else {
        assert(false);
        return static_cast<ENTRY_T>(1);
    }
}

} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LINEAR_ALGEBRA_TASKS_HPP_INCLUDED
