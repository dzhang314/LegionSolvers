#include <iostream>

#include <legion.h>

#include "LegionUtilities.hpp"
#include "Scalar.hpp"

enum TaskIDs : Legion::TaskID {
    CHILD_TASK_ID,
    TOP_LEVEL_TASK_ID
};


// void child_task(const Legion::Task *,
//                 const std::vector<Legion::PhysicalRegion> &,
//                 Legion::Context ctx, Legion::Runtime *rt) {
//     std::cout << "Hello from the child task!" << std::endl;
// }


void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
    std::cout << "Hello from the top-level task!" << std::endl;
    // {
    //     Legion::TaskLauncher launcher{CHILD_TASK_ID,
    //                                   Legion::TaskArgument{nullptr, 0}};
    //     launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    //     rt->execute_task(ctx, launcher);
    // }
}


int main(int argc, char **argv) {
    // LegionSolvers::preregister_cpu_task<child_task>(
    //     CHILD_TASK_ID, "child", false, true, false
    // );
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level",
        LegionSolvers::TaskFlags::REPLICABLE | LegionSolvers::TaskFlags::INNER
    );
    // LegionSolvers::preregister_solver_tasks(false);
    // Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    // Legion::Runtime::set_top_level_task_mapper_id(
    //     LegionSolvers::LEGION_SOLVERS_MAPPER_ID);
    return Legion::Runtime::start(argc, argv);
}
