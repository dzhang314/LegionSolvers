#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "LegionUtilities.hpp" // for LEGION_SOLVERS_DECLARE_TASK, TaskFlags
#include "TaskBaseClasses.hpp" // for TaskT, TaskDI
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


template <typename T>
struct PrintScalarTask
    : public TaskT<PRINT_SCALAR_TASK_BLOCK_ID, PrintScalarTask, T> {
    static constexpr const char *task_base_name = "print_scalar";
    static constexpr TaskFlags flags = TaskFlags::LEAF;
    LEGION_SOLVERS_DECLARE_TASK(int);
};


template <typename T>
struct NegateScalarTask
    : public TaskT<NEGATE_SCALAR_TASK_BLOCK_ID, NegateScalarTask, T> {
    static constexpr const char *task_base_name = "negate_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct AddScalarTask
    : public TaskT<ADD_SCALAR_TASK_BLOCK_ID, AddScalarTask, T> {
    static constexpr const char *task_base_name = "add_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct SubtractScalarTask
    : public TaskT<SUBTRACT_SCALAR_TASK_BLOCK_ID, SubtractScalarTask, T> {
    static constexpr const char *task_base_name = "subtract_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct MultiplyScalarTask
    : public TaskT<MULTIPLY_SCALAR_TASK_BLOCK_ID, MultiplyScalarTask, T> {
    static constexpr const char *task_base_name = "multiply_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct DivideScalarTask
    : public TaskT<DIVIDE_SCALAR_TASK_BLOCK_ID, DivideScalarTask, T> {
    static constexpr const char *task_base_name = "divide_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct SqrtScalarTask
    : public TaskT<SQRT_SCALAR_TASK_BLOCK_ID, SqrtScalarTask, T> {
    static constexpr const char *task_base_name = "sqrt_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct RSqrtScalarTask
    : public TaskT<RSQRT_SCALAR_TASK_BLOCK_ID, RSqrtScalarTask, T> {
    static constexpr const char *task_base_name = "rsqrt_scalar";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <typename T>
struct DummyTask : public TaskT<DUMMY_TASK_BLOCK_ID, DummyTask, T> {
    static constexpr const char *task_base_name = "dummy";
    static constexpr TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;
    LEGION_SOLVERS_DECLARE_TASK(T);
};


template <int DIM, typename COORD_T>
struct PrintIndexTask
    : public TaskDI<PRINT_INDEX_TASK_BLOCK_ID, PrintIndexTask, DIM, COORD_T> {
    static constexpr const char *task_base_name = "print_index";
    static constexpr TaskFlags flags = TaskFlags::LEAF;
    LEGION_SOLVERS_DECLARE_TASK(void);
};


template <typename ENTRY_T, int DIM, typename COORD_T>
struct RandomFillTask : public TaskTDI<
                            RANDOM_FILL_TASK_BLOCK_ID,
                            RandomFillTask,
                            ENTRY_T,
                            DIM,
                            COORD_T> {
    static constexpr const char *task_base_name = "random_fill";
    static constexpr TaskFlags flags = TaskFlags::LEAF;
    LEGION_SOLVERS_DECLARE_TASK(void);
};


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
