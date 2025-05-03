#include "Initialize.hpp"

#include <iostream> // for std::cout, std::endl

#include "COOMatrixTasks.hpp"
#include "CSRMatrixTasks.hpp"
#include "ExampleSystems.hpp"      // for Fill*NegativeLaplacianTask
#include "LegionSolversMapper.hpp" // for mapper_registration_callback
#include "LibraryOptions.hpp"      // for LEGION_SOLVERS_USE_*
#include "LinearAlgebraTasks.hpp"  // for ScalTask, AxpyTask, XpayTask, DotTask
#include "StencilGenerator.hpp"    // for FillCOOStencilTask, FillCSRStencilTask
#include "UtilityTasks.hpp"        // for *ScalarTask

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    #include "CudaLibs.hpp"
#endif


// clang-format off
void LegionSolvers::initialize(bool print_info, bool verbose) {

    if (print_info) {
        std::cout << "[LegionSolvers] Initializing library..." << std::endl;

        #ifdef LEGION_USE_CUDA
            std::cout << "[LegionSolvers] CUDA support enabled." << std::endl;
        #else
            std::cout << "[LegionSolvers] CUDA support disabled." << std::endl;
        #endif // LEGION_USE_CUDA

        #ifdef REALM_USE_KOKKOS
            std::cout << "[LegionSolvers] Kokkos support enabled." << std::endl;
        #else
            std::cout << "[LegionSolvers] Kokkos support disabled." << std::endl;
        #endif // REALM_USE_KOKKOS

        std::cout << "[LegionSolvers] Supported entry types:" << std::endl;
        #ifdef LEGION_SOLVERS_USE_F32
            std::cout << "[LegionSolvers]   * 32-bit floating-point" << std::endl;
        #endif // LEGION_SOLVERS_USE_F32
        #ifdef LEGION_SOLVERS_USE_F64
            std::cout << "[LegionSolvers]   * 64-bit floating-point" << std::endl;
        #endif // LEGION_SOLVERS_USE_F64

        std::cout << "[LegionSolvers] Supported index types:" << std::endl;
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            std::cout << "[LegionSolvers]   * signed 32-bit integer" << std::endl;
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            std::cout << "[LegionSolvers]   * unsigned 32-bit integer" << std::endl;
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            std::cout << "[LegionSolvers]   * signed 64-bit integer" << std::endl;
        #endif // LEGION_SOLVERS_USE_S64_INDICES

        std::cout << "[LegionSolvers] Supported dimensions:" << std::endl;
        #if LEGION_SOLVERS_MAX_DIM >= 1
            std::cout << "[LegionSolvers]   * 1" << std::endl;
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            std::cout << "[LegionSolvers]   * 2" << std::endl;
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            std::cout << "[LegionSolvers]   * 3" << std::endl;
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    }

    // LegionSolversMapper.hpp
    Legion::Runtime::add_registration_callback(
        LegionSolvers::mapper_registration_callback
    );
    Legion::Runtime::preregister_sharding_functor(
        LEGION_SOLVERS_SHARDING_FUNCTOR_ID,
        new BlockingShardingFunctor()
    );

    // UtilityTasks.hpp
    #ifdef LEGION_SOLVERS_USE_F32
        PrintScalarTask<float>::preregister(verbose);
        NegateScalarTask<float>::preregister(verbose);
        AddScalarTask<float>::preregister(verbose);
        SubtractScalarTask<float>::preregister(verbose);
        MultiplyScalarTask<float>::preregister(verbose);
        DivideScalarTask<float>::preregister(verbose);
        SqrtScalarTask<float>::preregister(verbose);
        RSqrtScalarTask<float>::preregister(verbose);
        DummyTask<float>::preregister(verbose);
    #endif // LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_F64
        PrintScalarTask<double>::preregister(verbose);
        NegateScalarTask<double>::preregister(verbose);
        AddScalarTask<double>::preregister(verbose);
        SubtractScalarTask<double>::preregister(verbose);
        MultiplyScalarTask<double>::preregister(verbose);
        DivideScalarTask<double>::preregister(verbose);
        SqrtScalarTask<double>::preregister(verbose);
        RSqrtScalarTask<double>::preregister(verbose);
        DummyTask<double>::preregister(verbose);
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
                DotTask<float, 1, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, int>::preregister(verbose);
                AxpyTask<float, 2, int>::preregister(verbose);
                XpayTask<float, 2, int>::preregister(verbose);
                DotTask<float, 2, int>::preregister(verbose);
                DotTask<float, 2, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, int>::preregister(verbose);
                AxpyTask<float, 3, int>::preregister(verbose);
                XpayTask<float, 3, int>::preregister(verbose);
                DotTask<float, 3, int>::preregister(verbose);
                DotTask<float, 3, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<float, 1, unsigned>::preregister(verbose);
                AxpyTask<float, 1, unsigned>::preregister(verbose);
                XpayTask<float, 1, unsigned>::preregister(verbose);
                DotTask<float, 1, unsigned>::preregister(verbose);
                DotTask<float, 1, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, unsigned>::preregister(verbose);
                AxpyTask<float, 2, unsigned>::preregister(verbose);
                XpayTask<float, 2, unsigned>::preregister(verbose);
                DotTask<float, 2, unsigned>::preregister(verbose);
                DotTask<float, 2, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, unsigned>::preregister(verbose);
                AxpyTask<float, 3, unsigned>::preregister(verbose);
                XpayTask<float, 3, unsigned>::preregister(verbose);
                DotTask<float, 3, unsigned>::preregister(verbose);
                DotTask<float, 3, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<float, 1, long long>::preregister(verbose);
                AxpyTask<float, 1, long long>::preregister(verbose);
                XpayTask<float, 1, long long>::preregister(verbose);
                DotTask<float, 1, long long>::preregister(verbose);
                DotTask<float, 1, long long>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<float, 2, long long>::preregister(verbose);
                AxpyTask<float, 2, long long>::preregister(verbose);
                XpayTask<float, 2, long long>::preregister(verbose);
                DotTask<float, 2, long long>::preregister(verbose);
                DotTask<float, 2, long long>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<float, 3, long long>::preregister(verbose);
                AxpyTask<float, 3, long long>::preregister(verbose);
                XpayTask<float, 3, long long>::preregister(verbose);
                DotTask<float, 3, long long>::preregister(verbose);
                DotTask<float, 3, long long>::preregister_fb_future_dot(verbose);
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
                DotTask<double, 1, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, int>::preregister(verbose);
                AxpyTask<double, 2, int>::preregister(verbose);
                XpayTask<double, 2, int>::preregister(verbose);
                DotTask<double, 2, int>::preregister(verbose);
                DotTask<double, 2, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, int>::preregister(verbose);
                AxpyTask<double, 3, int>::preregister(verbose);
                XpayTask<double, 3, int>::preregister(verbose);
                DotTask<double, 3, int>::preregister(verbose);
                DotTask<double, 3, int>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<double, 1, unsigned>::preregister(verbose);
                AxpyTask<double, 1, unsigned>::preregister(verbose);
                XpayTask<double, 1, unsigned>::preregister(verbose);
                DotTask<double, 1, unsigned>::preregister(verbose);
                DotTask<double, 1, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, unsigned>::preregister(verbose);
                AxpyTask<double, 2, unsigned>::preregister(verbose);
                XpayTask<double, 2, unsigned>::preregister(verbose);
                DotTask<double, 2, unsigned>::preregister(verbose);
                DotTask<double, 2, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, unsigned>::preregister(verbose);
                AxpyTask<double, 3, unsigned>::preregister(verbose);
                XpayTask<double, 3, unsigned>::preregister(verbose);
                DotTask<double, 3, unsigned>::preregister(verbose);
                DotTask<double, 3, unsigned>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                ScalTask<double, 1, long long>::preregister(verbose);
                AxpyTask<double, 1, long long>::preregister(verbose);
                XpayTask<double, 1, long long>::preregister(verbose);
                DotTask<double, 1, long long>::preregister(verbose);
                DotTask<double, 1, long long>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                ScalTask<double, 2, long long>::preregister(verbose);
                AxpyTask<double, 2, long long>::preregister(verbose);
                XpayTask<double, 2, long long>::preregister(verbose);
                DotTask<double, 2, long long>::preregister(verbose);
                DotTask<double, 2, long long>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                ScalTask<double, 3, long long>::preregister(verbose);
                AxpyTask<double, 3, long long>::preregister(verbose);
                XpayTask<double, 3, long long>::preregister(verbose);
                DotTask<double, 3, long long>::preregister(verbose);
                DotTask<double, 3, long long>::preregister_fb_future_dot(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F64

    // StencilGenerator.hpp
    #ifdef LEGION_SOLVERS_USE_F32
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<float, 1, int>::preregister(verbose);
                FillCSRStencilTask<float, 1, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 1, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<float, 2, int>::preregister(verbose);
                FillCSRStencilTask<float, 2, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 2, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<float, 3, int>::preregister(verbose);
                FillCSRStencilTask<float, 3, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 3, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<float, 1, unsigned>::preregister(verbose);
                FillCSRStencilTask<float, 1, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 1, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<float, 2, unsigned>::preregister(verbose);
                FillCSRStencilTask<float, 2, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 2, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<float, 3, unsigned>::preregister(verbose);
                FillCSRStencilTask<float, 3, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 3, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<float, 1, long long>::preregister(verbose);
                FillCSRStencilTask<float, 1, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 1, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<float, 2, long long>::preregister(verbose);
                FillCSRStencilTask<float, 2, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 2, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<float, 3, long long>::preregister(verbose);
                FillCSRStencilTask<float, 3, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<float, 3, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<float, 3, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S64_INDICES
    #endif // LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_F64
        #ifdef LEGION_SOLVERS_USE_S32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<double, 1, int>::preregister(verbose);
                FillCSRStencilTask<double, 1, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 1, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 1, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<double, 2, int>::preregister(verbose);
                FillCSRStencilTask<double, 2, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 2, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 2, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<double, 3, int>::preregister(verbose);
                FillCSRStencilTask<double, 3, int>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 3, int>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 3, int>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_S32_INDICES
        #ifdef LEGION_SOLVERS_USE_U32_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<double, 1, unsigned>::preregister(verbose);
                FillCSRStencilTask<double, 1, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 1, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 1, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<double, 2, unsigned>::preregister(verbose);
                FillCSRStencilTask<double, 2, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 2, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 2, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<double, 3, unsigned>::preregister(verbose);
                FillCSRStencilTask<double, 3, unsigned>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 3, unsigned>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 3, unsigned>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 3
        #endif // LEGION_SOLVERS_USE_U32_INDICES
        #ifdef LEGION_SOLVERS_USE_S64_INDICES
            #if LEGION_SOLVERS_MAX_DIM >= 1
                FillCOOStencilTask<double, 1, long long>::preregister(verbose);
                FillCSRStencilTask<double, 1, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 1, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 1, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 1
            #if LEGION_SOLVERS_MAX_DIM >= 2
                FillCOOStencilTask<double, 2, long long>::preregister(verbose);
                FillCSRStencilTask<double, 2, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 2, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 2, long long>::preregister(verbose);
            #endif // LEGION_SOLVERS_MAX_DIM >= 2
            #if LEGION_SOLVERS_MAX_DIM >= 3
                FillCOOStencilTask<double, 3, long long>::preregister(verbose);
                FillCSRStencilTask<double, 3, long long>::preregister(verbose);
                FillLinearizedCOOStencilTask<double, 3, long long>::preregister(verbose);
                FillLinearizedCSRStencilTask<double, 3, long long>::preregister(verbose);
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
    COOMatvecTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COORmatvecTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COOPrintTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COOPrintTask<float, 1, 2, 2, long long, long long, long long>::preregister(verbose);
    COOPrintTask<float, 1, 3, 3, long long, long long, long long>::preregister(verbose);
    COOMatvecTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COORmatvecTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COOPrintTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    COOPrintTask<double, 1, 2, 2, long long, long long, long long>::preregister(verbose);
    COOPrintTask<double, 1, 3, 3, long long, long long, long long>::preregister(verbose);
    CSRMatvecTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRRmatvecTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<float, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<float, 1, 2, 2, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<float, 1, 3, 3, long long, long long, long long>::preregister(verbose);
    CSRMatvecTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRRmatvecTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<double, 1, 1, 1, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<double, 1, 2, 2, long long, long long, long long>::preregister(verbose);
    CSRPrintTask<double, 1, 3, 3, long long, long long, long long>::preregister(verbose);

#if defined(LEGION_USE_CUDA) && !defined(REALM_USE_KOKKOS)
    InitNCCLUniqueIDTask::preregister(verbose);
    LoadCUDALibsTask::preregister(verbose);
    UnloadCUDALibsTask::preregister(verbose);
#endif
}
// clang-format on
