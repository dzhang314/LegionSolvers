#include <legion.h>

#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs : Legion::FieldID {
    FID_ENTRY,
};

void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
    const Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 99});
    const Legion::IndexSpaceT<1> color_space = rt->create_index_space(ctx, Legion::Rect<1>{0, 10});

    LegionSolvers::DistributedVector<double, 1, Legion::coord_t, 1, Legion::coord_t> vector{index_space, color_space, ctx, rt};
    vector.zero_fill();
    vector.print();
}



int main(int argc, char **argv) {
    LegionSolvers::preregister_cpu_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", false, false
    );
    LegionSolvers::PrintVectorTask<double, 1>::preregister_cpu(true);
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
