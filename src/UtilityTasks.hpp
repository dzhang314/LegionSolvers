#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T, int DIM>
    struct RandomFillTask : TaskTD<RANDOM_FILL_TASK_BLOCK_ID,
                                   RandomFillTask, T, DIM> {

        static constexpr const char *task_base_name = "random_fill";

        static constexpr bool is_leaf = true;

        using return_type = void;

        static void task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct RandomFillTask


    template <typename T, int DIM>
    struct PrintVectorTask : TaskTD<PRINT_VECTOR_TASK_BLOCK_ID,
                                    PrintVectorTask, T, DIM> {

        static constexpr const char *task_base_name = "print_vector";

        static constexpr bool is_leaf = true;

        using return_type = void;

        static void task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct PrintVectorTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP
