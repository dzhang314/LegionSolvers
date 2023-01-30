#include "Initialize.hpp"

#include "COOMatrixTasks.hpp"
#include "CSRMatrixTasks.hpp"
#include "ExampleSystems.hpp"      // for Fill*NegativeLaplacianTask
#include "LegionSolversMapper.hpp" // for mapper_registration_callback
#include "LibraryOptions.hpp"      // for LEGION_SOLVERS_USE_*
#include "LinearAlgebraTasks.hpp"  // for ScalTask, AxpyTask, XpayTask, DotTask
#include "UtilityTasks.hpp"        // for *ScalarTask

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    #include "CudaLibs.hpp"
#endif


// clang-format off
void LegionSolvers::initialize(bool verbose) {

    // LegionSolversMapper.hpp
    Legion::Runtime::add_registration_callback(
        LegionSolvers::mapper_registration_callback
    );

    #ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION
        Legion::Runtime::preregister_sharding_functor(
            LEGION_SOLVERS_SHARDING_FUNCTOR_ID,
            new BlockingShardingFunctor()
        );
    #endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION

    // UtilityTasks.hpp
    #ifdef LEGION_SOLVERS_USE_F32
        PrintScalarTask<float>::preregister(verbose);
        NegateScalarTask<float>::preregister(verbose);
        AddScalarTask<float>::preregister(verbose);
        SubtractScalarTask<float>::preregister(verbose);
        MultiplyScalarTask<float>::preregister(verbose);
        DivideScalarTask<float>::preregister(verbose);
    #endif // LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_F64
        PrintScalarTask<double>::preregister(verbose);
        NegateScalarTask<double>::preregister(verbose);
        AddScalarTask<double>::preregister(verbose);
        SubtractScalarTask<double>::preregister(verbose);
        MultiplyScalarTask<double>::preregister(verbose);
        DivideScalarTask<double>::preregister(verbose);
    #endif // LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            PrintIndexTask<1, int>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            PrintIndexTask<2, int>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            PrintIndexTask<3, int>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            PrintIndexTask<1, unsigned>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            PrintIndexTask<2, unsigned>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            PrintIndexTask<3, unsigned>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            PrintIndexTask<1, long long>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            PrintIndexTask<2, long long>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            PrintIndexTask<3, long long>::preregister(verbose);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
    #ifdef LEGION_SOLVERS_USE_F32
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<float, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<float, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<float, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<float, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<float, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<float, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<float, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<float, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<float, 3, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_F64
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<double, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<double, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<double, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<double, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<double, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<double, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                RandomFillTask<double, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                RandomFillTask<double, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                RandomFillTask<double, 3, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F64

    // LinearAlgebraTasks.hpp
    #ifdef LEGION_SOLVERS_USE_F32
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<float, 1, int>::preregister(verbose);
                AxpyTask<float, 1, int>::preregister(verbose);
                XpayTask<float, 1, int>::preregister(verbose);
                DotTask<float, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, int>::preregister(verbose);
                AxpyTask<float, 2, int>::preregister(verbose);
                XpayTask<float, 2, int>::preregister(verbose);
                DotTask<float, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, int>::preregister(verbose);
                AxpyTask<float, 3, int>::preregister(verbose);
                XpayTask<float, 3, int>::preregister(verbose);
                DotTask<float, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<float, 1, unsigned>::preregister(verbose);
                AxpyTask<float, 1, unsigned>::preregister(verbose);
                XpayTask<float, 1, unsigned>::preregister(verbose);
                DotTask<float, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, unsigned>::preregister(verbose);
                AxpyTask<float, 2, unsigned>::preregister(verbose);
                XpayTask<float, 2, unsigned>::preregister(verbose);
                DotTask<float, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, unsigned>::preregister(verbose);
                AxpyTask<float, 3, unsigned>::preregister(verbose);
                XpayTask<float, 3, unsigned>::preregister(verbose);
                DotTask<float, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<float, 1, long long>::preregister(verbose);
                AxpyTask<float, 1, long long>::preregister(verbose);
                XpayTask<float, 1, long long>::preregister(verbose);
                DotTask<float, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, long long>::preregister(verbose);
                AxpyTask<float, 2, long long>::preregister(verbose);
                XpayTask<float, 2, long long>::preregister(verbose);
                DotTask<float, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, long long>::preregister(verbose);
                AxpyTask<float, 3, long long>::preregister(verbose);
                XpayTask<float, 3, long long>::preregister(verbose);
                DotTask<float, 3, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_F64
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<double, 1, int>::preregister(verbose);
                AxpyTask<double, 1, int>::preregister(verbose);
                XpayTask<double, 1, int>::preregister(verbose);
                DotTask<double, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, int>::preregister(verbose);
                AxpyTask<double, 2, int>::preregister(verbose);
                XpayTask<double, 2, int>::preregister(verbose);
                DotTask<double, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, int>::preregister(verbose);
                AxpyTask<double, 3, int>::preregister(verbose);
                XpayTask<double, 3, int>::preregister(verbose);
                DotTask<double, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<double, 1, unsigned>::preregister(verbose);
                AxpyTask<double, 1, unsigned>::preregister(verbose);
                XpayTask<double, 1, unsigned>::preregister(verbose);
                DotTask<double, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, unsigned>::preregister(verbose);
                AxpyTask<double, 2, unsigned>::preregister(verbose);
                XpayTask<double, 2, unsigned>::preregister(verbose);
                DotTask<double, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, unsigned>::preregister(verbose);
                AxpyTask<double, 3, unsigned>::preregister(verbose);
                XpayTask<double, 3, unsigned>::preregister(verbose);
                DotTask<double, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<double, 1, long long>::preregister(verbose);
                AxpyTask<double, 1, long long>::preregister(verbose);
                XpayTask<double, 1, long long>::preregister(verbose);
                DotTask<double, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, long long>::preregister(verbose);
                AxpyTask<double, 2, long long>::preregister(verbose);
                XpayTask<double, 2, long long>::preregister(verbose);
                DotTask<double, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, long long>::preregister(verbose);
                AxpyTask<double, 3, long long>::preregister(verbose);
                XpayTask<double, 3, long long>::preregister(verbose);
                DotTask<double, 3, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F64

    // TODO: proper guards
    FillCOONegativeLaplacianTask<float, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    FillCOONegativeLaplacianTask<double, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    FillCSRNegativeLaplacianTask<float, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    FillCSRNegativeLaplacianTask<double, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    FillCSRNegativeLaplacianRowptrTask<float, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    FillCSRNegativeLaplacianRowptrTask<double, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    COOMatvecTask<float, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    COOMatvecTask<double, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    CSRMatvecTask<float, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);
    CSRMatvecTask<double, 1, 1, 1, Legion::coord_t, Legion::coord_t, Legion::coord_t>::preregister(verbose);

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    LoadCUDALibsTask::preregister(verbose);
#endif
}
// clang-format on
