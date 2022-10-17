#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED

#include <iostream>

#include <legion.h>

#include "LegionUtilities.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"

namespace LegionSolvers {


template <typename T>
struct PrintScalarTask
    : public TaskT<PRINT_SCALAR_TASK_BLOCK_ID, PrintScalarTask, T> {

    static constexpr const char *task_base_name = "print_scalar";

    static constexpr const TaskFlags flags = TaskFlags::LEAF;

    using return_type = int;

    static int task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert((task->futures.size() == 1) || (task->futures.size() == 2));
        Legion::Future x = task->futures[0];
        std::cout << x.get_result<T>() << std::endl;
        return 0;
    }

}; // struct PrintScalarTask


template <typename T>
struct NegateScalarTask
    : public TaskT<NEGATE_SCALAR_TASK_BLOCK_ID, NegateScalarTask, T> {

    static constexpr const char *task_base_name = "negate_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static T task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert(task->futures.size() == 1);
        Legion::Future x = task->futures[0];
        return -x.get_result<T>();
    }

}; // struct NegateScalarTask


template <typename T>
struct AddScalarTask
    : public TaskT<ADD_SCALAR_TASK_BLOCK_ID, AddScalarTask, T> {

    static constexpr const char *task_base_name = "add_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static T task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert(task->futures.size() == 2);
        Legion::Future x = task->futures[0];
        Legion::Future y = task->futures[1];
        return x.get_result<T>() + y.get_result<T>();
    }

}; // struct AddScalarTask


template <typename T>
struct SubtractScalarTask
    : public TaskT<SUBTRACT_SCALAR_TASK_BLOCK_ID, SubtractScalarTask, T> {

    static constexpr const char *task_base_name = "subtract_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static T task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert(task->futures.size() == 2);
        Legion::Future x = task->futures[0];
        Legion::Future y = task->futures[1];
        return x.get_result<T>() - y.get_result<T>();
    }

}; // struct SubtractScalarTask


template <typename T>
struct MultiplyScalarTask
    : public TaskT<MULTIPLY_SCALAR_TASK_BLOCK_ID, MultiplyScalarTask, T> {

    static constexpr const char *task_base_name = "multiply_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static T task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert(task->futures.size() == 2);
        Legion::Future x = task->futures[0];
        Legion::Future y = task->futures[1];
        return x.get_result<T>() * y.get_result<T>();
    }

}; // struct MultiplyScalarTask


template <typename T>
struct DivideScalarTask
    : public TaskT<DIVIDE_SCALAR_TASK_BLOCK_ID, DivideScalarTask, T> {

    static constexpr const char *task_base_name = "divide_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static T task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
        assert(task->futures.size() == 2);
        Legion::Future x = task->futures[0];
        Legion::Future y = task->futures[1];
        return x.get_result<T>() / y.get_result<T>();
    }

}; // struct DivideScalarTask


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
