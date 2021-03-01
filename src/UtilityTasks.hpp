#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <iostream>
#include <random>
#include <string>

#include <legion.h>

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
    Legion::Future add(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{AdditionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
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
    Legion::Future subtract(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{SubtractionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
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
    Legion::Future negate(Legion::Future a, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{NegationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
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
    Legion::Future multiply(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{MultiplicationTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
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
    Legion::Future divide(Legion::Future a, Legion::Future b, Legion::Context ctx, Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{DivisionTask<T>::task_id, Legion::TaskArgument{nullptr, 0}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(a);
        launcher.add_future(b);
        return rt->execute_task(ctx, launcher);
    }


    template <typename T, int DIM>
    struct ConstantFillTask : TaskTD<CONSTANT_FILL_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "constant_fill"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &region = regions[0];

            assert(task->regions.size() == 1);
            const auto &region_req = task->regions[0];

            assert(region_req.privilege_fields.size() == 1);
            const Legion::FieldID fid = *region_req.privilege_fields.begin();

            assert(task->arglen == sizeof(T));
            const T fill_value = *reinterpret_cast<const T *>(task->args);

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> entry_writer{region, fid};
            for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) { entry_writer[*iter] = fill_value; }
        }

    }; // struct ConstantFillTask


    template <typename T>
    void zero_fill(Legion::LogicalRegion region,
                   Legion::FieldID fid,
                   Legion::IndexPartition partition,
                   Legion::Context ctx,
                   Legion::Runtime *rt) {
        const T zero = static_cast<T>(0);
        Legion::IndexLauncher launcher{ConstantFillTask<T, 0>::task_id(region.get_dim()),
                                       rt->get_index_partition_color_space_name(partition),
                                       Legion::TaskArgument{&zero, sizeof(T)}, Legion::ArgumentMap{}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement{rt->get_logical_partition(region, partition), 0,
                                                                  LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        rt->execute_index_space(ctx, launcher);
    }


    template <typename T, int DIM>
    struct RandomFillTask : TaskTD<RANDOM_FILL_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "random_fill"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &region = regions[0];

            assert(task->regions.size() == 1);
            const auto &region_req = task->regions[0];

            assert(region_req.privilege_fields.size() == 1);
            const Legion::FieldID fid = *region_req.privilege_fields.begin();

            assert(task->arglen == 2 * sizeof(T));
            const T *arg_ptr = reinterpret_cast<const T *>(task->args);
            const T low = arg_ptr[0];
            const T high = arg_ptr[1];

            std::random_device rng{};
            std::uniform_real_distribution<T> entry_dist{low, high};
            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> entry_writer{region, fid};
            for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
                entry_writer[*iter] = entry_dist(rng);
            }
        }

    }; // struct RandomFillTask


    template <typename T>
    void random_fill(Legion::LogicalRegion region,
                     Legion::FieldID fid,
                     Legion::IndexPartition partition,
                     Legion::Context ctx,
                     Legion::Runtime *rt,
                     T low = static_cast<T>(0),
                     T high = static_cast<T>(1)) {
        const T args[2] = {low, high};
        Legion::IndexLauncher launcher{RandomFillTask<T, 0>::task_id(region.get_dim()),
                                       rt->get_index_partition_color_space_name(partition),
                                       Legion::TaskArgument{&args, 2 * sizeof(T)}, Legion::ArgumentMap{}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement{rt->get_logical_partition(region, partition), 0,
                                                                  LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        rt->execute_index_space(ctx, launcher);
    }


    template <typename T, int DIM>
    struct CopyTask : TaskTD<COPY_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "copy"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 2);
            const auto &dst = regions[0];
            const auto &src = regions[1];

            assert(task->regions.size() == 2);
            const auto &dst_req = task->regions[0];
            const auto &src_req = task->regions[1];

            assert(dst_req.privilege_fields.size() == 1);
            const Legion::FieldID dst_fid = *dst_req.privilege_fields.begin();

            assert(src_req.privilege_fields.size() == 1);
            const Legion::FieldID src_fid = *src_req.privilege_fields.begin();

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM> dst_writer{dst, dst_fid};
            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> src_reader{src, src_fid};

            for (Legion::PointInDomainIterator<DIM> iter{dst}; iter(); ++iter) {
                dst_writer[*iter] = src_reader[*iter];
            }
        }

    }; // struct CopyTask


    template <typename T, int DIM>
    struct PrintVectorTask : TaskTD<PRINT_VECTOR_TASK_BLOCK_ID, T, DIM> {

        static std::string task_name() { return "print_vector"; }

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &vector = regions[0];

            assert(task->regions.size() == 1);
            const auto &vector_req = task->regions[0];

            assert(vector_req.privilege_fields.size() == 1);
            const Legion::FieldID vector_fid = *vector_req.privilege_fields.begin();

            const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM> entry_reader{vector, vector_fid};

            if (task->arglen == 0) {
                for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
                    std::cout << task->index_point << ' ' << *iter << ": " << entry_reader[*iter] << '\n';
                }
            } else {
                const std::string vector_name{reinterpret_cast<const char *>(task->args)};
                for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
                    std::cout << vector_name << ' ' << task->index_point << ' ' << *iter << ": " << entry_reader[*iter]
                              << '\n';
                }
            }
            std::cout << std::flush;
        }

    }; // struct PrintVectorTask


    template <typename T>
    void print_vector(Legion::LogicalRegion region,
                      Legion::FieldID fid,
                      const std::string &name,
                      Legion::Context ctx,
                      Legion::Runtime *rt) {
        Legion::TaskLauncher launcher{LegionSolvers::PrintVectorTask<T, 0>::task_id(region.get_dim()),
                                      Legion::TaskArgument{name.c_str(), name.length() + 1}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement{region, LEGION_READ_ONLY, LEGION_EXCLUSIVE, region});
        launcher.add_field(0, fid);
        rt->execute_task(ctx, launcher);
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP
