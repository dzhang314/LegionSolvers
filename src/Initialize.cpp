#include "Initialize.hpp"

#include "LibraryOptions.hpp"     // for LEGION_SOLVERS_USE_*
#include "LinearAlgebraTasks.hpp" // for ScalTask, AxpyTask, XpayTask, DotTask
#include "UtilityTasks.hpp"       // for *ScalarTask

void LegionSolvers::initialize(bool verbose = true) {

#ifdef LEGION_SOLVERS_USE_FLOAT
    LegionSolvers::PrintScalarTask<float>::preregister(verbose);
    LegionSolvers::NegateScalarTask<float>::preregister(verbose);
    LegionSolvers::AddScalarTask<float>::preregister(verbose);
    LegionSolvers::SubtractScalarTask<float>::preregister(verbose);
    LegionSolvers::MultiplyScalarTask<float>::preregister(verbose);
    LegionSolvers::DivideScalarTask<float>::preregister(verbose);
#endif // LEGION_SOLVERS_USE_FLOAT

#ifdef LEGION_SOLVERS_USE_DOUBLE
    LegionSolvers::PrintScalarTask<double>::preregister(verbose);
    LegionSolvers::NegateScalarTask<double>::preregister(verbose);
    LegionSolvers::AddScalarTask<double>::preregister(verbose);
    LegionSolvers::SubtractScalarTask<double>::preregister(verbose);
    LegionSolvers::MultiplyScalarTask<double>::preregister(verbose);
    LegionSolvers::DivideScalarTask<double>::preregister(verbose);
#endif // LEGION_SOLVERS_USE_DOUBLE
}
