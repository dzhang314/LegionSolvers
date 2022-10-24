#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "LegionUtilities.hpp" // for TaskFlags
#include "TaskBaseClasses.hpp" // for TaskT
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <typename T>
struct PrintScalarTask
    : public TaskT<PRINT_SCALAR_TASK_BLOCK_ID, PrintScalarTask, T> {

    static constexpr const char *task_base_name = "print_scalar";

    static constexpr const TaskFlags flags = TaskFlags::LEAF;

    using return_type = int;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct PrintScalarTask


template <int DIM>
struct PrintIndexTask
    : public TaskD<PRINT_INDEX_TASK_BLOCK_ID, PrintIndexTask, DIM> {

    static constexpr const char *task_base_name = "print_index";

    static constexpr const TaskFlags flags = TaskFlags::LEAF;

    using return_type = void;

    static void task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct PrintIndexTask


template <typename T>
struct NegateScalarTask
    : public TaskT<NEGATE_SCALAR_TASK_BLOCK_ID, NegateScalarTask, T> {

    static constexpr const char *task_base_name = "negate_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct NegateScalarTask


template <typename T>
struct AddScalarTask
    : public TaskT<ADD_SCALAR_TASK_BLOCK_ID, AddScalarTask, T> {

    static constexpr const char *task_base_name = "add_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct AddScalarTask


template <typename T>
struct SubtractScalarTask
    : public TaskT<SUBTRACT_SCALAR_TASK_BLOCK_ID, SubtractScalarTask, T> {

    static constexpr const char *task_base_name = "subtract_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct SubtractScalarTask


template <typename T>
struct MultiplyScalarTask
    : public TaskT<MULTIPLY_SCALAR_TASK_BLOCK_ID, MultiplyScalarTask, T> {

    static constexpr const char *task_base_name = "multiply_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct MultiplyScalarTask


template <typename T>
struct DivideScalarTask
    : public TaskT<DIVIDE_SCALAR_TASK_BLOCK_ID, DivideScalarTask, T> {

    static constexpr const char *task_base_name = "divide_scalar";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    using return_type = T;

    static return_type task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );

}; // struct DivideScalarTask


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
