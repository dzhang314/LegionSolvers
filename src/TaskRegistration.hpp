#ifndef LEGION_SOLVERS_TASK_REGISTRATION_HPP_INCLUDED
#define LEGION_SOLVERS_TASK_REGISTRATION_HPP_INCLUDED

#include "UtilityTasks.hpp" // for *ScalarTask

namespace LegionSolvers {


void preregister_tasks(bool verbose = true) {
    LegionSolvers::PrintScalarTask<float>::preregister(verbose);
    LegionSolvers::PrintScalarTask<double>::preregister(verbose);
    LegionSolvers::NegateScalarTask<float>::preregister(verbose);
    LegionSolvers::NegateScalarTask<double>::preregister(verbose);
    LegionSolvers::AddScalarTask<float>::preregister(verbose);
    LegionSolvers::AddScalarTask<double>::preregister(verbose);
    LegionSolvers::SubtractScalarTask<float>::preregister(verbose);
    LegionSolvers::SubtractScalarTask<double>::preregister(verbose);
    LegionSolvers::MultiplyScalarTask<float>::preregister(verbose);
    LegionSolvers::MultiplyScalarTask<double>::preregister(verbose);
    LegionSolvers::DivideScalarTask<float>::preregister(verbose);
    LegionSolvers::DivideScalarTask<double>::preregister(verbose);
}


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_TASK_REGISTRATION_HPP_INCLUDED
