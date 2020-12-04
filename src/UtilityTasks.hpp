#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T>
    struct AdditionTask : TaskT<ADDITION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() + b.get_result<T>();
        }

    }; // struct AdditionTask


    template <typename T>
    Legion::Future add(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{AdditionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct SubtractionTask : TaskT<SUBTRACTION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() - b.get_result<T>();
        }

    }; // struct SubtractionTask


    template <typename T>
    Legion::Future subtract(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{SubtractionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct NegationTask : TaskT<NEGATION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 1);
            Legion::Future a = task->futures[0];

            return -a.get_result<T>();
        }

    }; // struct NegationTask


    template <typename T>
    Legion::Future negate(Legion::Future a, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{NegationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct MultiplicationTask : TaskT<MULTIPLICATION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() * b.get_result<T>();
        }

    }; // struct MultiplicationTask


    template <typename T>
    Legion::Future multiply(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{MultiplicationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct DivisionTask : TaskT<DIVISION_TASK_BLOCK_ID, T> {

        static T task(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context,
                      Legion::Runtime *) {

            assert(task->futures.size() == 2);
            Legion::Future a = task->futures[0];
            Legion::Future b = task->futures[1];

            return a.get_result<T>() / b.get_result<T>();
        }

    }; // struct DivisionTask


    template <typename T>
    Legion::Future divide(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DivisionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP
