#include <iostream>

#include <legion.h>

#include "Initialize.hpp"
#include "LegionUtilities.hpp"

enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };

void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    std::cout << "Hello, world!" << std::endl;
}

int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::initialize(false);
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID,
        "top_level",
        TaskFlags::REPLICABLE | TaskFlags::INNER,
        Legion::Processor::LOC_PROC
    );
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    return Legion::Runtime::start(argc, argv);
}
