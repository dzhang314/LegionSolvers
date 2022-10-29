#include "CSRMatrixTasks.hpp"

#include "CuSPARSEHelpers.hpp"
#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using namespace LegionSolvers;

template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void CSRMatvecTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    cuda_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {

    assert(regions.size() == 4);
    const auto &output_vec = regions[0];
    const auto &csr_matrix = regions[1];
    const auto &aux_region = regions[2];
    const auto &input_vec = regions[3];

    assert(task->regions.size() == 4);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &aux_req = task->regions[2];
    const auto &input_req = task->regions[3];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 2);
    assert(task->arglen == 2 * sizeof(Legion::FieldID));
    const Legion::FieldID *argptr =
        reinterpret_cast<const Legion::FieldID *>(task->args);
    const Legion::FieldID fid_col = argptr[0];
    const Legion::FieldID fid_entry = argptr[1];

    assert(aux_req.privilege_fields.size() == 1);
    const Legion::FieldID fid_rowptr = *aux_req.privilege_fields.begin();

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader{csr_matrix, fid_col};

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader{
        csr_matrix, fid_entry};

    const AffineReader<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_reader{aux_region, fid_rowptr};

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader{
        input_vec, input_fid};

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T> output_writer{
        output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>};

    auto stream = get_cached_stream();
    auto handle = get_cusparse();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto output_bounds = output_vec.get_bounds<RANGE_DIM, RANGE_COORD_T>();
    assert(output_bounds.dense());
    auto rows = output_bounds.bounds.volume();

    auto input_bounds = input_vec.get_bounds<DOMAIN_DIM, DOMAIN_COORD_T>();
    // The number of columns in this slice of the COO matrix is at most
    // the upper domain of the input vector, since the kernel and domain
    // are related by an image.
    static_assert(DOMAIN_DIM == 1);
    auto cols = input_bounds.bounds.hi[0] + 1;

    auto cusparse_csr = makeCuSparseCSR<
        ENTRY_T,
        KERNEL_DIM,
        DOMAIN_DIM,
        RANGE_DIM,
        KERNEL_COORD_T,
        DOMAIN_COORD_T,
        RANGE_COORD_T>(
        stream,
        rows,
        cols,
        aux_region.get_bounds<KERNEL_DIM, KERNEL_COORD_T>(),
        rowptr_reader,
        csr_matrix.get_bounds<KERNEL_DIM, KERNEL_COORD_T>(),
        col_reader,
        entry_reader
    );
    // There is an image relationship between col->input, so input should
    // be offset to the base of the image rather used directly.
    auto cusparse_input =
        makeShiftedCuSparseDnVec<ENTRY_T, decltype(input_reader)>(
            input_bounds, cols, input_reader
        );
    // There is no such relationship between output and row, so the
    // vector can be used directly.
    auto cusparse_output = makeCuSparseDnVec<ENTRY_T, decltype(output_writer)>(
        output_bounds, output_writer
    );

    ENTRY_T alpha = static_cast<ENTRY_T>(1.0);
    ENTRY_T beta = static_cast<ENTRY_T>(0.0);
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        cusparse_csr,
        cusparse_input,
        &beta,
        cusparse_output,
        cusparseDataType<ENTRY_T>(),
        CUSPARSE_MV_ALG_DEFAULT,
        &bufSize
    ));
    void *workspace = nullptr;
    if (bufSize > 0) {
        Legion::DeferredBuffer<char, 1> buf(
            {0, bufSize - 1}, Legion::Memory::GPU_FB_MEM
        );
        workspace = buf.ptr(0);
    }
    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        cusparse_csr,
        cusparse_input,
        &beta,
        cusparse_output,
        cusparseDataType<ENTRY_T>(),
        CUSPARSE_MV_ALG_DEFAULT,
        workspace
    ));

    // Clean up after ourselves.
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_input));
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_output));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_csr));
}

template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void CSRRmatvecTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    cuda_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {
    assert(false);
}

template void LegionSolvers::CSRMatvecTask<
    float,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    cuda_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
template void LegionSolvers::CSRMatvecTask<
    double,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    cuda_task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    );
