#include "CSRMatrixTasks.hpp"

#include "CuSPARSEHelpers.hpp"
#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using LegionSolvers::CSRMatvecTask;
using LegionSolvers::CSRRmatvecTask;


LEGION_SOLVERS_KDR_TEMPLATE
void CSRMatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::cuda_task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 4);
    const auto &output_vec = regions[0];
    const auto &csr_matrix = regions[1];
    const auto &rowptr_region = regions[2];
    const auto &input_vec = regions[3];

    assert(task->regions.size() == 4);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &rowptr_req = task->regions[2];
    const auto &input_req = task->regions[3];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 2);

    assert(rowptr_req.privilege_fields.size() == 1);
    const Legion::FieldID fid_rowptr = *rowptr_req.privilege_fields.begin();

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::DomainT<RANGE_DIM, RANGE_COORD_T> output_domain =
        output_vec.get_bounds<RANGE_DIM, RANGE_COORD_T>();

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T>
        output_writer(output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>);

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader(
        csr_matrix, args.fid_entry
    );

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader(csr_matrix, args.fid_col);

    const AffineReader<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_reader(rowptr_region, fid_rowptr);

    const Legion::DomainT<DOMAIN_DIM, DOMAIN_COORD_T> input_domain =
        input_vec.get_bounds<DOMAIN_DIM, DOMAIN_COORD_T>();

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader(
        input_vec, input_fid
    );

    assert(output_domain.dense());
    auto rows = output_domain.bounds.volume();
    // Break out if there are no rows to process.
    if (rows == 0) { return; }

    // The number of columns in this slice of the COO matrix is at most
    // the upper domain of the input vector, since the kernel and domain
    // are related by an image.
    static_assert(DOMAIN_DIM == 1);
    auto cols = input_domain.bounds.hi[0] + 1;

    auto stream = get_cuda_stream();
    auto handle = get_cusparse_handle();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

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
        rowptr_region.get_bounds<RANGE_DIM, RANGE_COORD_T>(),
        rowptr_reader,
        csr_matrix.get_bounds<KERNEL_DIM, KERNEL_COORD_T>(),
        col_reader,
        entry_reader
    );
    // There is an image relationship between col->input, so input should
    // be offset to the base of the image rather used directly.
    auto cusparse_input =
        makeShiftedCuSparseDnVec<ENTRY_T, decltype(input_reader)>(
            input_domain, cols, input_reader
        );
    // There is no such relationship between output and row, so the
    // vector can be used directly.
    auto cusparse_output = makeCuSparseDnVec<ENTRY_T, decltype(output_writer)>(
        output_domain, output_writer
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
        CUDA_DATA_TYPE<ENTRY_T>,
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
        CUDA_DATA_TYPE<ENTRY_T>,
        CUSPARSE_MV_ALG_DEFAULT,
        workspace
    ));

    // Clean up after ourselves.
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_input));
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_output));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_csr));
}


LEGION_SOLVERS_KDR_TEMPLATE
void CSRRmatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::cuda_task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(false);
}


// clang-format off
template void CSRMatvecTask<float, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRRmatvecTask<float, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRMatvecTask<double, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void CSRRmatvecTask<double, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
// clang-format on
