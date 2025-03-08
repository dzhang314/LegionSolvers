#include "CudaLibs.hpp"

#include <mappers/default_mapper.h>
#include <stdio.h>

namespace LegionSolvers {

using namespace Legion;

void InitNCCLUniqueIDTask::preregister(bool verbose) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task "
                  << std::string(InitNCCLUniqueIDTask::task_name) << " with ID "
                  << InitNCCLUniqueIDTask::TASK_ID << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{
        InitNCCLUniqueIDTask::TASK_ID, InitNCCLUniqueIDTask::task_name};
    registrar.add_constraint(Legion::ProcessorConstraint{
        Legion::Processor::TOC_PROC});
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<InitNCCLUniqueIDTask::return_type, InitNCCLUniqueIDTask::task_body>(
        registrar, InitNCCLUniqueIDTask::task_name
    );
}

InitNCCLUniqueIDTask::return_type InitNCCLUniqueIDTask::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    return id;
}

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
    assert(task->futures.size() == 1);
    // Initialize the NCCL communicator too.
    auto id   = task->futures[0].get_result<ncclUniqueId>();
    ncclComm_t comm;
    auto num_ranks = task->index_domain.get_volume();
    auto rank_id   = task->index_point[0];
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclCommInitRank(&comm, num_ranks, id, rank_id));
    CHECK_NCCL(ncclGroupEnd());
    set_nccl_comm(comm);
    get_nccl_comm();
}

/* static */
UnloadCUDALibsTask::return_type UnloadCUDALibsTask::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    ncclComm_t comm = get_nccl_comm();
    CHECK_NCCL(ncclCommFinalize(comm));
    CHECK_NCCL(ncclCommDestroy(comm));
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
    // First launch the NCCL initialization task.
    auto ncclid = runtime->execute_task(
      ctx,
      Legion::TaskLauncher(
        InitNCCLUniqueIDTask::TASK_ID,
	UntypedBuffer()
      )
    );

    // Launch the initialization task onto each GPU.
    auto launch_space =
        runtime->create_index_space(ctx, Legion::Rect<1>{0, num_gpus - 1});

    IndexLauncher itl(
      LoadCUDALibsTask::TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap(),
      Legion::Predicate::TRUE_PRED,
      false /* must */,
      LEGION_SOLVERS_MAPPER_ID
    );
    itl.add_future(ncclid);

    runtime
        ->execute_index_space(ctx, itl)
        .wait_all_results(true /* silence_warnings */);

#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    runtime->destroy_index_space(ctx, launch_space);
#endif // LEGION_SOLVERS_DISABLE_CLEANUP
}

/* static */
void UnloadCUDALibsTask::preregister(bool verbose) {
    if (verbose) {
        std::cout << "[LegionSolvers] Registering task "
                  << std::string(UnloadCUDALibsTask::task_name) << " with ID "
                  << UnloadCUDALibsTask::TASK_ID << "." << std::endl;
    }
    Legion::TaskVariantRegistrar registrar{
        UnloadCUDALibsTask::TASK_ID, UnloadCUDALibsTask::task_name};
    registrar.add_constraint(Legion::ProcessorConstraint{
        Legion::Processor::TOC_PROC});
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<UnloadCUDALibsTask::task_body>(
        registrar, UnloadCUDALibsTask::task_name
    );
}

void unloadCUDALibs(Legion::Context ctx, Legion::Runtime *runtime) {
    auto tunable = Legion::Mapping::DefaultMapper::DefaultTunables::
        DEFAULT_TUNABLE_GLOBAL_GPUS;
    auto num_gpus =
        runtime->select_tunable_value(ctx, tunable, LEGION_SOLVERS_MAPPER_ID)
            .get<size_t>();
    if (num_gpus == 0) return;

    // Make sure everything is done running.
    runtime->issue_execution_fence(ctx).wait();

    // Launch the cleanup task onto each GPU.
    auto launch_space =
        runtime->create_index_space(ctx, Legion::Rect<1>{0, num_gpus - 1});

    IndexLauncher itl(
      UnloadCUDALibsTask::TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap(),
      Legion::Predicate::TRUE_PRED,
      false /* must */,
      LEGION_SOLVERS_MAPPER_ID
    );
    runtime
        ->execute_index_space(ctx, itl)
        .wait_all_results(true /* silence_warnings */);

#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    runtime->destroy_index_space(ctx, launch_space);
#endif // LEGION_SOLVERS_DISABLE_CLEANUP
}

} // namespace LegionSolvers
