#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <iostream>
#include <random>
#include <string>

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <typename T>
    struct AdditionTask : TaskT<ADDITION_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "addition"; }

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
    Legion::Future add(Legion::Future a, Legion::Future b,
                       Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{AdditionTask<T>::task_id,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct SubtractionTask : TaskT<SUBTRACTION_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "subtraction"; }

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
    Legion::Future subtract(Legion::Future a, Legion::Future b,
                            Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{SubtractionTask<T>::task_id,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct NegationTask : TaskT<NEGATION_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "negation"; }

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
    Legion::Future negate(Legion::Future a,
                          Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{NegationTask<T>::task_id,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct MultiplicationTask : TaskT<MULTIPLICATION_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "multiplication"; }

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
    Legion::Future multiply(Legion::Future a, Legion::Future b,
                            Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{MultiplicationTask<T>::task_id,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T>
    struct DivisionTask : TaskT<DIVISION_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "division"; }

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
    Legion::Future divide(Legion::Future a, Legion::Future b,
                          Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DivisionTask<T>::task_id,
                                      Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T, int DIM>
    struct DummyTask : TaskTD<DUMMY_TASK_BLOCK_ID, DummyTask, T, DIM> {

        static std::string task_base_name() { return "dummy_task"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            // const auto &region = regions[0];

            assert(task->regions.size() == 1);
            const auto &region_req = task->regions[0];

            assert(region_req.privilege_fields.size() == 1);
            // const Legion::FieldID fid = *region_req.privilege_fields.begin();

        }

    }; // struct DummyTask


    template <typename T>
    void dummy_task(Legion::LogicalRegion region, Legion::FieldID fid,
                    Legion::IndexPartition partition,
                    Legion::Context ctx, Legion::Runtime *rt) {
        Legion::IndexLauncher launcher{
            DummyTask<T, 0>::task_id(region.get_dim()),
            rt->get_index_partition_color_space_name(partition),
            Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement{
            rt->get_logical_partition(region, partition),
            0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        rt->execute_index_space(ctx, launcher);
    }


    template <typename T>
    void random_fill(Legion::LogicalRegion region, Legion::FieldID fid,
                     Legion::IndexPartition partition,
                     Legion::Context ctx, Legion::Runtime *rt,
                     T low = static_cast<T>(0), T high = static_cast<T>(1)) {
        const T args[2] = {low, high};
        Legion::IndexLauncher launcher{
            RandomFillTask<T, 0>::task_id(region.get_dim()),
            rt->get_index_partition_color_space_name(partition),
            Legion::TaskArgument{&args, 2 * sizeof(T)}, Legion::ArgumentMap{}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement{
            rt->get_logical_partition(region, partition),
            0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        rt->execute_index_space(ctx, launcher);
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP
