#include "TaskRegistration.hpp"

#include "UtilityTasks.hpp"
#include "LinearAlgebraTasks.hpp"
#include "COOMatrixTasks.hpp"


void LegionSolvers::preregister_solver_tasks(bool verbose) {

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

}
