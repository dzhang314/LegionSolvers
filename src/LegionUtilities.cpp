#include "LegionUtilities.hpp"

#include <cassert> // for assert

#include "UtilityTasks.hpp" // for LegionSolvers::PrintIndexTask


Legion::FieldSpace LegionSolvers::create_field_space(
    const std::vector<std::size_t> &field_sizes,
    const std::vector<Legion::FieldID> &field_ids,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::vector<Legion::FieldID> field_ids_copy{field_ids};
    const Legion::FieldSpace result =
        rt->create_field_space(ctx, field_sizes, field_ids_copy);
    assert(field_ids == field_ids_copy);
    return result;
}


void LegionSolvers::print_index_partition(
    const std::string &name,
    Legion::IndexPartition index_partition,
    Legion::Context ctx, Legion::Runtime *rt
) {
    const Legion::IndexSpace color_space =
        rt->get_index_partition_color_space_name(index_partition);
    const int color_dim = color_space.get_dim();
    const Legion::FieldSpace dummy_field_space =
        LegionSolvers::create_field_space({1}, {0}, ctx, rt);
    const Legion::LogicalRegion dummy_region = rt->create_logical_region(ctx,
        rt->get_parent_index_space(index_partition),
        dummy_field_space
    );
    rt->fill_field(ctx, dummy_region, dummy_region, 0, '\0');
    Legion::IndexLauncher launcher{
        LegionSolvers::PrintIndexTask<0>::task_id(color_dim),
        color_space,
        Legion::TaskArgument{name.c_str(), name.size() + 1},
        Legion::ArgumentMap{}
    };
    launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    launcher.add_region_requirement(Legion::RegionRequirement{
        rt->get_logical_partition(dummy_region, index_partition), 0,
        LEGION_READ_ONLY, LEGION_EXCLUSIVE, dummy_region
    });
    launcher.add_field(0, 0);
    rt->execute_index_space(ctx, launcher);
    rt->destroy_logical_region(ctx, dummy_region);
    rt->destroy_field_space(ctx, dummy_field_space);
}
