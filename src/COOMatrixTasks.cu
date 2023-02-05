#include "COOMatrixTasks.hpp"

#include "CuSPARSEHelpers.hpp"
#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using LegionSolvers::COOMatvecTask;
using LegionSolvers::COORmatvecTask;


LEGION_SOLVERS_KDR_TEMPLATE
void COOMatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::cuda_task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(regions.size() == 3);
    const auto &output_vec = regions[0];
    const auto &coo_matrix = regions[1];
    const auto &input_vec = regions[2];

    assert(task->regions.size() == 3);
    const auto &output_req = task->regions[0];
    const auto &matrix_req = task->regions[1];
    const auto &input_req = task->regions[2];

    assert(output_req.privilege_fields.size() == 1);
    const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

    assert(matrix_req.privilege_fields.size() == 3);

    assert(input_req.privilege_fields.size() == 1);
    const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::DomainT<RANGE_DIM, RANGE_COORD_T> output_domain =
        output_vec.get_bounds<RANGE_DIM, RANGE_COORD_T>();

    const AffineSumAccessor<ENTRY_T, RANGE_DIM, RANGE_COORD_T>
        output_writer(output_vec, output_fid, LEGION_REDOP_SUM<ENTRY_T>);

    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_reader(
        coo_matrix, args.fid_entry
    );

    const AffineReader<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        row_reader(coo_matrix, args.fid_row);

    const AffineReader<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_reader(coo_matrix, args.fid_col);

    const Legion::DomainT<DOMAIN_DIM, DOMAIN_COORD_T> input_domain =
        input_vec.get_bounds<DOMAIN_DIM, DOMAIN_COORD_T>();

    const AffineReader<ENTRY_T, DOMAIN_DIM, DOMAIN_COORD_T> input_reader(
        input_vec, input_fid
    );

    auto stream = get_cuda_stream();
    auto handle = get_cusparse_handle();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    const Legion::Domain coo_bounds =
        coo_matrix.get_bounds<KERNEL_DIM, KERNEL_COORD_T>();
    // If there are no coordinates to process, break out.
    if (coo_bounds.empty()) { return; }
    // The number of rows in this slice of the COO matrix is at most
    // the upper domain of the output vector, since the kernel and range
    // are related by an image.
    static_assert(RANGE_DIM == 1);
    auto rows = output_domain.bounds.hi[0] + 1;
    // The number of columns in this slice of the COO matrix is at most
    // the upper domain of the input vector, since the kernel and domain
    // are related by an image.
    static_assert(DOMAIN_DIM == 1);
    auto cols = input_domain.bounds.hi[0] + 1;

    // Construct our cuSPARSE objects from individual regions.
    auto cusparse_coo = makeCuSparseCOO<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>(
        rows, cols, coo_bounds, row_reader, col_reader, entry_reader
    );
    // There are image relationships between row->output and col->input,
    // so these vectors should be offset to the base of the image rather
    // used directly.
    auto cusparse_input =
        makeShiftedCuSparseDnVec<ENTRY_T, decltype(input_reader)>(
            input_domain, cols, input_reader
        );
    auto cusparse_output =
        makeShiftedCuSparseDnVec<ENTRY_T, decltype(output_writer)>(
            output_domain, rows, output_writer
        );

    ENTRY_T alpha = static_cast<ENTRY_T>(1.0);
    // Interestingly, we set beta = 1.0 here rather than 0.0. Because
    // we launch tasks over the output with a reduction requirement,
    // we can't just throw away the data that is already present in
    // the output instance with beta = 0.0, as that is not a valid
    // operation with reduction privileges. Instead, we have to just
    // accumulate onto the data that exists already, so we use beta = 1.0.
    ENTRY_T beta = static_cast<ENTRY_T>(1.0);
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        cusparse_coo,
        cusparse_input,
        &beta,
        cusparse_output,
        CUDA_DATA_TYPE<ENTRY_T>,
        CUSPARSE_SPMV_ALG_DEFAULT,
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
        cusparse_coo,
        cusparse_input,
        &beta,
        cusparse_output,
        CUDA_DATA_TYPE<ENTRY_T>,
        CUSPARSE_SPMV_ALG_DEFAULT,
        workspace
    ));

    // Clean up after ourselves.
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_input));
    CHECK_CUSPARSE(cusparseDestroyDnVec(cusparse_output));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_coo));
}


LEGION_SOLVERS_KDR_TEMPLATE
void COORmatvecTask<LEGION_SOLVERS_KDR_TEMPLATE_ARGS>::cuda_task_body(
    LEGION_SOLVERS_TASK_ARGS
) {
    assert(false);
}


// clang-format off
template void COOMatvecTask<float, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void COORmatvecTask<float, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void COOMatvecTask<double, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
template void COORmatvecTask<double, 1, 1, 1, long long, long long, long long>::cuda_task_body(LEGION_SOLVERS_TASK_ARGS);
// clang-format on
