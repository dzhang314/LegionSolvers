#include "TaskRegistration.hpp"

#include "LegionUtilities.hpp"
#include "UtilityTasks.hpp"
#include "LinearAlgebraTasks.hpp"
#include "COOMatrixTasks.hpp"


void LegionSolvers::preregister_solver_tasks(bool verbose) {

    LegionSolvers::AdditionTask<float>::preregister_cpu(verbose);
    LegionSolvers::AdditionTask<double>::preregister_cpu(verbose);
    LegionSolvers::SubtractionTask<float>::preregister_cpu(verbose);
    LegionSolvers::SubtractionTask<double>::preregister_cpu(verbose);
    LegionSolvers::NegationTask<float>::preregister_cpu(verbose);
    LegionSolvers::NegationTask<double>::preregister_cpu(verbose);
    LegionSolvers::MultiplicationTask<float>::preregister_cpu(verbose);
    LegionSolvers::MultiplicationTask<double>::preregister_cpu(verbose);
    LegionSolvers::DivisionTask<float>::preregister_cpu(verbose);
    LegionSolvers::DivisionTask<double>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<float, 1>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<float, 2>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<float, 3>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<double, 1>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<double, 2>::preregister_cpu(verbose);
    LegionSolvers::RandomFillTask<double, 3>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<float, 1>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<float, 2>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<float, 3>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<double, 1>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<double, 2>::preregister_cpu(verbose);
    LegionSolvers::PrintVectorTask<double, 3>::preregister_cpu(verbose);

    LegionSolvers::ScalTask<float, 1>::preregister_kokkos(verbose);
    LegionSolvers::ScalTask<float, 2>::preregister_kokkos(verbose);
    LegionSolvers::ScalTask<float, 3>::preregister_kokkos(verbose);
    LegionSolvers::ScalTask<double, 1>::preregister_kokkos(verbose);
    LegionSolvers::ScalTask<double, 2>::preregister_kokkos(verbose);
    LegionSolvers::ScalTask<double, 3>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<float, 1>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<float, 2>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<float, 3>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<double, 1>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<double, 2>::preregister_kokkos(verbose);
    LegionSolvers::AxpyTask<double, 3>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<float, 1>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<float, 2>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<float, 3>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<double, 1>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<double, 2>::preregister_kokkos(verbose);
    LegionSolvers::XpayTask<double, 3>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<float, 1>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<float, 2>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<float, 3>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<double, 1>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<double, 2>::preregister_kokkos(verbose);
    LegionSolvers::DotTask<double, 3>::preregister_kokkos(verbose);

    LegionSolvers::COOMatvecTask<float, 1, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 1, 3, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 2, 3, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<float, 3, 3, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 1, 3, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 2, 3, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 1, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 1, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 1, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 2, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 2, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 2, 3>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 3, 1>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 3, 2>::preregister_kokkos(verbose);
    LegionSolvers::COOMatvecTask<double, 3, 3, 3>::preregister_kokkos(verbose);

    Legion::Runtime::preregister_projection_functor(
        PFID_KDR_TO_K, new ProjectionOneLevel{0}
    );
    Legion::Runtime::preregister_projection_functor(
        PFID_KDR_TO_D, new ProjectionOneLevel{1}
    );
    Legion::Runtime::preregister_projection_functor(
        PFID_KDR_TO_R, new ProjectionOneLevel{2}
    );
    Legion::Runtime::preregister_projection_functor(
        PFID_KDR_TO_DR, new ProjectionTwoLevel{1, 2}
    );

}