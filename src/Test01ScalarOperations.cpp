#include <iostream>

#include <legion.h>

#include "LegionSolversMapper.hpp"
#include "LegionUtilities.hpp"
#include "Scalar.hpp"
#include "TaskRegistration.hpp"

enum TaskIDs : Legion::TaskID { TOP_LEVEL_TASK_ID };

void top_level_task(
    const Legion::Task *,
    const std::vector<Legion::PhysicalRegion> &,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    {
        LegionSolvers::Scalar<float> x{ctx, rt, 2.0};
        LegionSolvers::Scalar<float> y{ctx, rt, 10.0};
        LegionSolvers::Scalar<float> z = x + y;
        LegionSolvers::Scalar<float> w = z / (x + x);
        LegionSolvers::Scalar<float> v = w - x;
        assert(v.get_value() == 1.0);
    }
    {
        LegionSolvers::Scalar<double> x{ctx, rt, 2.0};
        LegionSolvers::Scalar<double> y{ctx, rt, 10.0};
        LegionSolvers::Scalar<double> z = x + y;
        LegionSolvers::Scalar<double> w = z / (x + x);
        LegionSolvers::Scalar<double> v = w - x;
        assert(v.get_value() == 1.0);
    }
}

int main(int argc, char **argv) {
    using LegionSolvers::TaskFlags;
    LegionSolvers::preregister_tasks(false);
    LegionSolvers::preregister_task<top_level_task>(
        TOP_LEVEL_TASK_ID, "top_level", TaskFlags::REPLICABLE | TaskFlags::INNER
    );
    Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    Legion::Runtime::add_registration_callback(
        LegionSolvers::mapper_registration_callback
    );
    return Legion::Runtime::start(argc, argv);
}
