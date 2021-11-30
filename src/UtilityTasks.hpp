#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T>
    struct AdditionTask : public TaskT<ADDITION_TASK_BLOCK_ID,
                                       AdditionTask, T> {

        static constexpr const char *task_base_name = "addition";

        static constexpr bool is_leaf = true;

        using return_type = T;

        static T task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct AdditionTask


    template <typename T>
    struct SubtractionTask : public TaskT<SUBTRACTION_TASK_BLOCK_ID,
                                          SubtractionTask, T> {

        static constexpr const char *task_base_name = "subtraction";

        static constexpr bool is_leaf = true;

        using return_type = T;

        static T task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct SubtractionTask


    template <typename T>
    struct NegationTask : public TaskT<NEGATION_TASK_BLOCK_ID,
                                       NegationTask, T> {

        static constexpr const char *task_base_name = "negation";

        static constexpr bool is_leaf = true;

        using return_type = T;

        static T task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct NegationTask


    template <typename T>
    struct MultiplicationTask : public TaskT<MULTIPLICATION_TASK_BLOCK_ID,
                                             MultiplicationTask, T> {

        static constexpr const char *task_base_name = "multiplication";

        static constexpr bool is_leaf = true;

        using return_type = T;

        static T task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct MultiplicationTask


    template <typename T>
    struct DivisionTask : public TaskT<DIVISION_TASK_BLOCK_ID,
                                       DivisionTask, T> {

        static constexpr const char *task_base_name = "division";

        static constexpr bool is_leaf = true;

        using return_type = T;

        static T task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct DivisionTask


    template <typename T>
    struct AssertSmallTask : public TaskT<ASSERT_SMALL_TASK_BLOCK_ID,
                                          AssertSmallTask, T> {

        static constexpr const char *task_base_name = "assert_small";

        static constexpr bool is_leaf = true;

        using return_type = void;

        static void task_body(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx, Legion::Runtime *rt
        );

    }; // struct DivisionTask


    template <typename T, int DIM>
    struct RandomFillTask : public TaskTD<RANDOM_FILL_TASK_BLOCK_ID,
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
    struct PrintVectorTask : public TaskTD<PRINT_VECTOR_TASK_BLOCK_ID,
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
