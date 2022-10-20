#include "LegionUtilities.hpp"

#include <cassert> // for assert


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
