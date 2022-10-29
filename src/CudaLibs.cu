#include "CudaLibs.hpp"

#include <mappers/default_mapper.h>
#include <stdio.h>

namespace LegionSolvers {

using namespace Legion;

StreamView::~StreamView() {
    // TODO (rohany): We can have a static variable that controls whether
    //  or not we will do synchronization at the end of tasks.
    if (valid_) {
#ifndef NDEBUG
        CHECK_CUDA_STREAM(stream_);
#endif
        // We don't currently use the CUDA hijack, so we'll let realm handle
        // checking the stream for us at the end of the
        // #else
        //     CHECK_CUDA(cudaStreamSynchronize(stream_));
        // #endif
    }
}

StreamView::StreamView(StreamView &&rhs)
    : valid_(rhs.valid_), stream_(rhs.stream_) {
    rhs.valid_ = false;
}

StreamView &StreamView::operator=(StreamView &&rhs) {
    valid_ = rhs.valid_;
    stream_ = rhs.stream_;
    rhs.valid_ = false;
    return *this;
}

CUDALibraries::CUDALibraries() : cusparse_(nullptr) {}

cudaStream_t CUDALibraries::get_stream() {
    if (this->stream_ == nullptr) {
        CHECK_CUDA(
            cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking)
        );
    }
    return this->stream_;
}

cublasHandle_t CUDALibraries::get_cublas() {
    if (this->cublas_ == nullptr) {
        CHECK_CUBLAS(cublasCreate(&this->cublas_));
    }
    return this->cublas_;
}

cusparseHandle_t CUDALibraries::get_cusparse() {
    if (this->cusparse_ == nullptr) {
        CHECK_CUSPARSE(cusparseCreate(&this->cusparse_));
    }
    return this->cusparse_;
}

static CUDALibraries &get_cuda_libraries(Processor proc) {
    if (proc.kind() != Processor::TOC_PROC) {
        fprintf(
            stderr, "Illegal request for CUDA libraries for non-GPU processor"
        );
        assert(false);
    }

    static CUDALibraries cuda_libraries[LEGION_MAX_NUM_PROCS];
    const auto proc_id = proc.id & (LEGION_MAX_NUM_PROCS - 1);
    return cuda_libraries[proc_id];
}

StreamView get_cached_stream() {
    const auto proc = Processor::get_executing_processor();
    auto &lib = get_cuda_libraries(proc);
    return StreamView(lib.get_stream());
}

cublasHandle_t get_cublas() {
    const auto proc = Processor::get_executing_processor();
    auto &lib = get_cuda_libraries(proc);
    return lib.get_cublas();
}

cusparseHandle_t get_cusparse() {
    const auto proc = Processor::get_executing_processor();
    auto &lib = get_cuda_libraries(proc);
    return lib.get_cusparse();
}

/* static */
LoadCUDALibsTask::return_type LoadCUDALibsTask::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    const auto proc = Processor::get_executing_processor();
    auto &lib = get_cuda_libraries(proc);
    lib.get_stream();
    lib.get_cublas();
    lib.get_cusparse();
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
