#include "CudaLibs.hpp"

#include <mappers/default_mapper.h>
#include <stdio.h>

namespace LegionSolvers {

using namespace Legion;

/* static */
LoadCUDALibsTask::return_type LoadCUDALibsTask::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    get_cuda_stream();
    get_cublas_handle();
    get_cusparse_handle();
}

/* static */
void LoadCUDALibsTask::preregister(bool verbose) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task "
                  << std::string(LoadCUDALibsTask::task_name) << " with ID "
                  << LoadCUDALibsTask::TASK_ID << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{
        LoadCUDALibsTask::TASK_ID, LoadCUDALibsTask::task_name};
    registrar.add_constraint(Legion::ProcessorConstraint{
        Legion::Processor::TOC_PROC});
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<LoadCUDALibsTask::task_body>(
        registrar, LoadCUDALibsTask::task_name
    );
}

void loadCUDALibs(Legion::Context ctx, Legion::Runtime *runtime) {
    auto tunable = Legion::Mapping::DefaultMapper::DefaultTunables::
        DEFAULT_TUNABLE_GLOBAL_GPUS;
    auto num_gpus =
        runtime->select_tunable_value(ctx, tunable, LEGION_SOLVERS_MAPPER_ID)
            .get<size_t>();
    if (num_gpus == 0) return;
    // Launch the initialization task onto each GPU.
    auto launch_space =
        runtime->create_index_space(ctx, Legion::Rect<1>{0, num_gpus - 1});
    runtime
        ->execute_index_space(
            ctx,
            Legion::IndexLauncher{
                LoadCUDALibsTask::TASK_ID,
                launch_space,
                Legion::TaskArgument{},
                Legion::ArgumentMap{},
                Legion::Predicate::TRUE_PRED,
                false /* must */,
                LEGION_SOLVERS_MAPPER_ID}
        )
        .wait_all_results(true /* silence_warnings */);
    runtime->destroy_index_space(ctx, launch_space);
}

} // namespace LegionSolvers
