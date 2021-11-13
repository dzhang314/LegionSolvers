#ifndef LEGION_SOLVERS_UTILITY_TASKS_HPP
#define LEGION_SOLVERS_UTILITY_TASKS_HPP

#include <iostream>
#include <random>
#include <string>

#include <legion.h>

#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


namespace LegionSolvers {


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


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_UTILITY_TASKS_HPP
