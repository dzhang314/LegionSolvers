#include <algorithm>
#include <cassert>

#include <legion.h>
#include <realm/cmdline.h>

#include "CGSolver.hpp"
#include "Initialize.hpp"
#include "LegionUtilities.hpp"
#include "PartitionedVector.hpp"
#include "SquarePlanner.hpp"
#include "StencilGenerator.hpp"


enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };


void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    constexpr int DIM = 3;
    using COORD_T = Legion::coord_t;
    using ENTRY_T = double;

    Legion::Rect<DIM, COORD_T> bounds = {{-1, -2, -3}, {+2, +1, +3}};

    std::vector<std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>> offsets;
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{0, 0, 0}, +6.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{-1, 0, 0}, -1.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{+1, 0, 0}, -1.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{0, -1, 0}, -1.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{0, +1, 0}, -1.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{0, 0, -1}, -1.0);
    offsets.emplace_back(Legion::Point<DIM, COORD_T>{0, 0, +1}, -1.0);

    const LegionSolvers::COOMatrix<ENTRY_T> matrix =
        LegionSolvers::create_coo_stencil_matrix(ctx, rt, bounds, offsets, 4);
    matrix.print(rt->create_index_space(ctx, bounds));
}


int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::initialize(true, true);
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID,
        "top_level",
        Legion::Processor::LOC_PROC,
        TaskFlags::INNER | TaskFlags::REPLICABLE
    );
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::set_top_level_task_mapper_id(
        LegionSolvers::LEGION_SOLVERS_MAPPER_ID
    );
    return Legion::Runtime::start(argc, argv, false, false);
}
