#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
#define LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED

#include <iostream>

#include <legion.h>

#include "LegionUtilities.hpp"
#include "TaskBaseClasses.hpp"

namespace LegionSolvers {


template <typename T>
struct PrintScalarTask
    : public TaskT<PRINT_SCALAR_TASK_BLOCK_ID, PrintScalarTask, T> {

    static constexpr const char *task_base_name = "print_scalar";

    static constexpr const TaskFlags flags = TaskFlags::LEAF;

    using return_type = int;

    int task_body(
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


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP_INCLUDED
