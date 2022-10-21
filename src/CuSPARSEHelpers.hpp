#pragma once

#include "CudaLibs.hpp"
#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...

namespace LegionSolvers {

// Template dispatch for value type.
template <typename ENTRY_T>
cudaDataType cusparseDataType() {
  assert(false);
  return CUDA_R_32F;
}

template <>
cudaDataType cusparseDataType<float>() {
  return CUDA_R_32F;
}

template <>
cudaDataType cusparseDataType<double>() {
  return CUDA_R_64F;
}

// Template dispatch for the index type.
template <typename COORD_T>
cusparseIndexType_t cusparseIndexType() {
  assert(false);
  return CUSPARSE_INDEX_32I;
}

// TODO (rohany): Can we handle unsigned integers?
template <>
cusparseIndexType_t cusparseIndexType<int32_t>() {
  return CUSPARSE_INDEX_32I;
}

template <>
cusparseIndexType_t cusparseIndexType<int64_t>() {
  return CUSPARSE_INDEX_64I;
}

template<typename KERNEL_COORD_T, typename RANGE_COORD_T>
__global__
void convertGlobalRowptrToLocalIndPtr(
    size_t rows,
    RANGE_COORD_T lo,
    const AffineReader<Legion::Rect<1, KERNEL_COORD_T>, 1, RANGE_COORD_T> rowptr,
    Legion::DeferredBuffer<KERNEL_COORD_T, 1> indptr) {
    const auto idx = global_tid_1d();
    if (idx >= rows) return;

    // Offset each entry in the indptr array down to the first element in
    // rowptr to index this piece of the CSR array.
    indptr[idx] = static_cast<KERNEL_COORD_T>(rowptr[idx + lo].lo - rowptr[lo].lo);
    // We also need to fill in the final rows + 1 index of indptr to be
    // the total number of non-zeros. We'll have the first thread do this.
    if (idx == 0) {
        indptr[rows] = static_cast<KERNEL_COORD_T>(rowptr[rows - 1 + lo].hi + 1 - rowptr[lo].lo);
    }
}

// Utilities for constructing cuSPARSE matrices and vectors.
template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
cusparseSpMatDescr_t makeCuSparseCSR(
    StreamView& stream,
    int64_t rows,
    int64_t cols,
    const Legion::Domain rowptr_domain,
    const AffineReader<Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>, RANGE_DIM, RANGE_COORD_T> rowptr,
    const Legion::Domain kernel_domain,
    const AffineReader<Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>, KERNEL_DIM, KERNEL_COORD_T> cols,
    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entries
    ) {
    // cuSPARSE can only handle one-dimensional objects.
    static_assert(KERNEL_DIM == 1);
    static_assert(DOMAIN_DIM == 1);
    static_assert(RANGE_DIM == 1);
    // Ensure that these point types are bit-identical with their base types.
    static_assert(sizeof(Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>) == sizeof(DOMAIN_COORD_T));

    // First, we need to convert the Legion::Rect based rowptr
    // region into a standard indptr array that cuSPARSE understands.
    Legion::DeferredBuffer<KERNEL_COORD_T, KERNEL_DIM> indptr({0, rows}, Legion::Memory::GPU_FB_MEM);
    auto blocks = get_num_blocks_1d(rows);
    convertGlobalRowptrToLocalIndPtr<KERNEL_COORD_T, RANGE_COORD_T><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows,
        rowptr_domain.lo(),
        rowptr,
        indptr
    );

    // Next, use the local indptr array to construct the CSR array.
    cusparseSpMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateCsr(
       &handle,
       rows,
       cols,
       kernel_domain.get_volume(),
       indptr.ptr(0),
       cols.ptr(kernel_domain.lo()),
       entries.ptr(kernel_domain.lo()),
       cusparseIndexType<KERNEL_COORD_T>(),
       cusparseIndexType<DOMAIN_COORD_T>(),
       CUSPARSE_INDEX_BASE_ZERO,
       cusparseDataType<ENTRY_T>()
    ));
    return descr;
}

template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
cusparseSpMatDescr_t makeCuSparseCOO(
    int64_t rows,
    int64_t cols,
    const Legion::Domain domain,
    const AffineReader<Legion::Point<RANGE_DIM, RANGE_COORD_T>, KERNEL_DIM, KERNEL_COORD_T> row,
    const AffineReader<Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>, KERNEL_DIM, KERNEL_COORD_T> col,
    const AffineReader<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entries) {
    // cuSPARSE can only handle one-dimensional objects.
    static_assert(KERNEL_DIM == 1);
    static_assert(DOMAIN_DIM == 1);
    static_assert(RANGE_DIM == 1);
    // Ensure that these point types are bit-identical with their base types.
    static_assert(sizeof(Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>) == sizeof(DOMAIN_COORD_T));
    static_assert(sizeof(Legion::Point<RANGE_DIM, RANGE_COORD_T>) == sizeof(RANGE_COORD_T));

    // COO objects can be constructed directly from the base regions.
    cusparseSpMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateCoo(
        &descr,
        rows,
        cols,
        domain.get_volume(),
        row.ptr(domain.lo()),
        col.ptr(domain.lo()),
        entries.ptr(domain.lo()),
        cusparseIndexType<KERNEL_COORD_T>(),
        CUSPARSE_INDEX_BASE_ZERO,
        cusparseDataType<ENTRY_T>()
    ));
    return descr;
}

// TODO (rohany): comment on the difference between these two things.

template <typename ENTRY_T, typename Accessor>
cusparseDnVecDescr_t makeShiftedCuSparseDnVec(
    const Legion::Domain domain,
    size_t size,
    Accessor acc) {
    cusparseDnVecDescr_t descr;
    // TODO (rohany): Comment this...
    CHECK_CUSPARSE(cusparseCreateDnVec(
        &descr,
        size,
        acc.ptr(domain.lo()) - size_t(domain.lo()[0]),
        cusparseDataType<ENTRY_T>()
    ));
    return descr;
}

template <typename ENTRY_T, typename Accessor>
cusparseDnVecDescr_t makeCuSparseDnVec(
    const Legion::Domain domain,
    Accessor acc) {
    cusparseDnVecDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(
        &descr,
        domain.get_volume(),
        acc.ptr(domain.lo()),
        cusparseDataType<ENTRY_T>()
    ));
    return descr;
}

} // namespace LegionSolvers