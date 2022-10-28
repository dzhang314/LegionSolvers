#include "ExampleSystems.hpp"

#include <cassert> // for assert

using LegionSolvers::FillCOONegativeLaplacianTask;
using LegionSolvers::FillCSRNegativeLaplacianRowptrTask;
using LegionSolvers::FillCSRNegativeLaplacianTask;


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void FillCOONegativeLaplacianTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {

    // TODO: add partial specializations for other dimensionalities
    static_assert(KERNEL_DIM == 1);
    static_assert(DOMAIN_DIM == 1);
    static_assert(RANGE_DIM == 1);

    std::cout << "[LegionSolvers] Constructing COO 1D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        i_writer(matrix, args.fid_i);
    const AffineWriter<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        j_writer(matrix, args.fid_j);
    const AffineWriter<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_writer(
        matrix, args.fid_entry
    );

    const Legion::Domain matrix_domain =
        rt->get_index_space_domain(ctx, matrix_req.region.get_index_space());

    using PointIter = Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;

    for (PointIter it{matrix_domain}; it(); ++it) {
        const Legion::Point<KERNEL_DIM, KERNEL_COORD_T> p = *it;
        const KERNEL_COORD_T k = p[0];
        i_writer[p] = Legion::Point<RANGE_DIM, RANGE_COORD_T>(
            static_cast<RANGE_COORD_T>((k + 1) / 3)
        );
        j_writer[p] = Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>(
            static_cast<DOMAIN_COORD_T>(k - 2 * ((k + 1) / 3))
        );
        entry_writer[p] = static_cast<ENTRY_T>((k % 3) ? -1.0 : +2.0);
    }

    std::cout << "[LegionSolvers] Finished constructing COO 1D Laplacian."
              << std::endl;
}


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void FillCSRNegativeLaplacianTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {

    // TODO: add partial specializations for other dimensionalities
    static_assert(KERNEL_DIM == 1);
    static_assert(DOMAIN_DIM == 1);
    static_assert(RANGE_DIM == 1);

    std::cout << "[LegionSolvers] Constructing CSR 1D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<
        Legion::Point<RANGE_DIM, RANGE_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_writer(matrix, args.fid_col);
    const AffineWriter<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_writer(
        matrix, args.fid_entry
    );

    const Legion::Domain matrix_domain =
        rt->get_index_space_domain(ctx, matrix_req.region.get_index_space());

    using PointIter = Legion::PointInDomainIterator<KERNEL_DIM, KERNEL_COORD_T>;

    for (PointIter it{matrix_domain}; it(); ++it) {
        const Legion::Point<KERNEL_DIM, KERNEL_COORD_T> p = *it;
        const KERNEL_COORD_T k = p[0];
        col_writer[p] = Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>(
            static_cast<DOMAIN_COORD_T>(k - 2 * ((k + 1) / 3))
        );
        entry_writer[p] = static_cast<ENTRY_T>((k % 3) ? -1.0 : +2.0);
    }

    std::cout << "[LegionSolvers] Finished constructing CSR 1D Laplacian."
              << std::endl;
}


template <
    typename ENTRY_T,
    int KERNEL_DIM,
    int DOMAIN_DIM,
    int RANGE_DIM,
    typename KERNEL_COORD_T,
    typename DOMAIN_COORD_T,
    typename RANGE_COORD_T>
void FillCSRNegativeLaplacianRowptrTask<
    ENTRY_T,
    KERNEL_DIM,
    DOMAIN_DIM,
    RANGE_DIM,
    KERNEL_COORD_T,
    DOMAIN_COORD_T,
    RANGE_COORD_T>::
    task_body(
        const Legion::Task *task,
        const std::vector<Legion::PhysicalRegion> &regions,
        Legion::Context ctx,
        Legion::Runtime *rt
    ) {

    // TODO: add partial specializations for other dimensionalities
    static_assert(KERNEL_DIM == 1);
    static_assert(DOMAIN_DIM == 1);
    static_assert(RANGE_DIM == 1);

    std::cout << "[LegionSolvers] Constructing CSR 1D Laplacian row pointers..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &rowptr_region = regions[0];

    assert(task->regions.size() == 1);
    const auto &rowptr_req = task->regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const AffineWriter<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_writer(rowptr_region, args.fid_rowptr);

    const Legion::Domain rowptr_domain =
        rt->get_index_space_domain(ctx, rowptr_req.region.get_index_space());

    using PointIter = Legion::PointInDomainIterator<RANGE_DIM, RANGE_COORD_T>;
    const KERNEL_COORD_T grid_length = args.grid_shape[0];

    for (PointIter it{rowptr_domain}; it(); ++it) {
        const Legion::Point<RANGE_DIM, RANGE_COORD_T> p = *it;
        const KERNEL_COORD_T k = p[0];
        if (k == 0) {
            rowptr_writer[p] = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>(
                static_cast<KERNEL_COORD_T>(0), static_cast<KERNEL_COORD_T>(1)
            );
        } else if (k == grid_length - 1) {
            rowptr_writer[p] = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>(
                static_cast<KERNEL_COORD_T>(3 * grid_length - 4),
                static_cast<KERNEL_COORD_T>(3 * grid_length - 3)
            );
        } else {
            rowptr_writer[p] = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>(
                static_cast<KERNEL_COORD_T>(3 * k - 1),
                static_cast<KERNEL_COORD_T>(3 * k + 1)
            );
        }
    }

    std::cout << "[LegionSolvers] Finished constructing CSR 1D Laplacian "
              << "row pointers." << std::endl;
}


// TODO: conditionally compile all coordinate types
template void FillCOONegativeLaplacianTask<
    float,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void FillCOONegativeLaplacianTask<
    double,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void FillCSRNegativeLaplacianTask<
    float,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void FillCSRNegativeLaplacianTask<
    double,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void FillCSRNegativeLaplacianRowptrTask<
    float,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void FillCSRNegativeLaplacianRowptrTask<
    double,
    1,
    1,
    1,
    Legion::coord_t,
    Legion::coord_t,
    Legion::coord_t>::
    task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
