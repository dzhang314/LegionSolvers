#include "LegionUtilities.hpp"

#include <cassert> // for assert

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_USE_*
#include "UtilityTasks.hpp"   // for PrintIndexTask


Legion::FieldSpace LegionSolvers::create_field_space(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::vector<std::size_t> &field_sizes,
    const std::vector<Legion::FieldID> &field_ids
) {
    std::vector<Legion::FieldID> field_ids_copy{field_ids};
    const Legion::FieldSpace result =
        rt->create_field_space(ctx, field_sizes, field_ids_copy);
    assert(field_ids == field_ids_copy);
    return result;
}


void LegionSolvers::print_index_partition(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::string &name,
    Legion::IndexPartition index_partition
) {
    const Legion::IndexSpace color_space =
        rt->get_index_partition_color_space_name(index_partition);
    const Legion::FieldSpace dummy_field_space =
        create_field_space(ctx, rt, {1}, {0});
    const Legion::LogicalRegion dummy_region = rt->create_logical_region(
        ctx, rt->get_parent_index_space(index_partition), dummy_field_space
    );
    rt->fill_field(ctx, dummy_region, dummy_region, 0, '\0');
    Legion::IndexLauncher launcher(
        PrintIndexTask<0, void>::task_id(color_space),
        color_space,
        Legion::TaskArgument(name.c_str(), name.size() + 1),
        Legion::ArgumentMap()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_region_requirement(Legion::RegionRequirement(
        rt->get_logical_partition(dummy_region, index_partition),
        0,
        LEGION_READ_ONLY,
        LEGION_EXCLUSIVE,
        dummy_region
    ));
    launcher.add_field(0, 0);
    rt->execute_index_space(ctx, launcher);
    rt->destroy_logical_region(ctx, dummy_region);
    rt->destroy_field_space(ctx, dummy_field_space);
}


template <typename ENTRY_T>
ENTRY_T LegionSolvers::get_alpha(const std::vector<Legion::Future> &futures) {
    if (futures.size() == 0) {
        return static_cast<ENTRY_T>(1);
    } else if (futures.size() == 1) {
        return futures[0].get_result<ENTRY_T>();
    } else if (futures.size() == 2) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        return f0 / f1;
    } else if (futures.size() == 3) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        return (f0 * f1) / f2;
    } else if (futures.size() == 4) {
        const ENTRY_T f0 = futures[0].get_result<ENTRY_T>();
        const ENTRY_T f1 = futures[1].get_result<ENTRY_T>();
        const ENTRY_T f2 = futures[2].get_result<ENTRY_T>();
        const ENTRY_T f3 = futures[3].get_result<ENTRY_T>();
        return (f0 * f1) / (f2 * f3);
    } else {
        assert(false);
        return static_cast<ENTRY_T>(0);
    }
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template float LegionSolvers::get_alpha<float>(const std::vector<Legion::Future> &);
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template double LegionSolvers::get_alpha<double>(const std::vector<Legion::Future> &);
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
