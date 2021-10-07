#include "LegionUtilities.hpp"

#include <tuple>


template <int DIM>
Legion::LogicalRegionT<DIM> LegionSolvers::create_region(
    Legion::IndexSpaceT<DIM> index_space,
    const std::vector<std::pair<std::size_t, Legion::FieldID>> &fields,
    Legion::Context ctx, Legion::Runtime *rt
) {
    const auto field_space = rt->create_field_space(ctx);
    auto allocator = rt->create_field_allocator(ctx, field_space);
    for (const auto [field_size, field_id] : fields) {
        allocator.allocate_field(field_size, field_id);
    }
    return rt->create_logical_region(ctx, index_space, field_space);
}


[[maybe_unused]]
static constexpr auto instantiations = std::make_tuple(
    LegionSolvers::create_region<1>,
    LegionSolvers::create_region<2>,
    LegionSolvers::create_region<3>
);
