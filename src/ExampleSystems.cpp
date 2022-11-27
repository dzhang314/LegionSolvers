#include "ExampleSystems.hpp"

#include <cassert> // for assert

using LegionSolvers::COOMatrix;
using LegionSolvers::CSRMatrix;
using LegionSolvers::FillCOONegativeLaplacianTask;
using LegionSolvers::FillCSRNegativeLaplacianRowptrTask;
using LegionSolvers::FillCSRNegativeLaplacianTask;


template <typename ENTRY_T>
COOMatrix<ENTRY_T> LegionSolvers::coo_negative_laplacian_1d(
    Legion::Context ctx,
    Legion::Runtime *rt,
    Legion::coord_t grid_size,
    Legion::IndexSpace launch_space
) {
    constexpr int KERNEL_DIM = 1;
    constexpr int DOMAIN_DIM = 1;
    constexpr int RANGE_DIM = 1;

    using KERNEL_COORD_T = Legion::coord_t;
    using DOMAIN_COORD_T = Legion::coord_t;
    using RANGE_COORD_T = Legion::coord_t;

    using KernelRect = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>;
    using DomainIndex = Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>;
    using RangeIndex = Legion::Point<RANGE_DIM, RANGE_COORD_T>;

    constexpr Legion::FieldID FID_ENTRY = 0;
    constexpr Legion::FieldID FID_ROW = 1;
    constexpr Legion::FieldID FID_COL = 2;

    const KERNEL_COORD_T kernel_size = 3 * grid_size - 2;

    const Legion::IndexSpace kernel_space =
        rt->create_index_space(ctx, KernelRect{0, kernel_size - 1});

    const Legion::FieldSpace field_space = create_field_space(
        ctx,
        rt,
        {sizeof(ENTRY_T), sizeof(RangeIndex), sizeof(DomainIndex)},
        {FID_ENTRY, FID_ROW, FID_COL}
    );

    const Legion::LogicalRegion kernel_region =
        rt->create_logical_region(ctx, kernel_space, field_space);

    const Legion::IndexPartition kernel_partition =
        rt->create_equal_partition(ctx, kernel_space, launch_space);

    typename FillCOONegativeLaplacianTask<
        ENTRY_T,
        KERNEL_DIM,
        DOMAIN_DIM,
        RANGE_DIM,
        KERNEL_COORD_T,
        DOMAIN_COORD_T,
        RANGE_COORD_T>::Args args;
    args.fid_entry = FID_ENTRY;
    args.fid_row = FID_ROW;
    args.fid_col = FID_COL;
    args.grid_shape[0] = grid_size;

    Legion::IndexTaskLauncher launcher(
        LegionSolvers::FillCOONegativeLaplacianTask<
            ENTRY_T,
            KERNEL_DIM,
            DOMAIN_DIM,
            RANGE_DIM,
            KERNEL_COORD_T,
            DOMAIN_COORD_T,
            RANGE_COORD_T>::task_id,
        launch_space,
        Legion::TaskArgument(&args, sizeof(args)),
        Legion::ArgumentMap()
    );

    launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    launcher.add_region_requirement(Legion::RegionRequirement(
        rt->get_logical_partition(kernel_region, kernel_partition),
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        kernel_region
    ));
    launcher.add_field(0, FID_ENTRY);
    launcher.add_field(0, FID_ROW);
    launcher.add_field(0, FID_COL);
    rt->execute_index_space(ctx, launcher);

    COOMatrix<ENTRY_T> result(
        ctx, rt, kernel_region, FID_ENTRY, FID_ROW, FID_COL
    );

    rt->destroy_index_partition(ctx, kernel_partition);
    rt->destroy_logical_region(ctx, kernel_region);
    rt->destroy_field_space(ctx, field_space);
    rt->destroy_index_space(ctx, kernel_space);

    return result;
}


template <typename ENTRY_T>
CSRMatrix<ENTRY_T> LegionSolvers::csr_negative_laplacian_1d(
    Legion::Context ctx,
    Legion::Runtime *rt,
    Legion::coord_t grid_size,
    Legion::IndexSpace launch_space
) {
    constexpr int KERNEL_DIM = 1;
    constexpr int DOMAIN_DIM = 1;
    constexpr int RANGE_DIM = 1;

    using KERNEL_COORD_T = Legion::coord_t;
    using DOMAIN_COORD_T = Legion::coord_t;
    using RANGE_COORD_T = Legion::coord_t;

    using KernelRect = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>;
    using DomainIndex = Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>;
    using RangeRect = Legion::Rect<RANGE_DIM, RANGE_COORD_T>;

    constexpr Legion::FieldID FID_ENTRY = 0;
    constexpr Legion::FieldID FID_COL = 1;
    constexpr Legion::FieldID FID_ROWPTR = 0;

    const KERNEL_COORD_T kernel_size = 3 * grid_size - 2;

    const Legion::IndexSpace kernel_space =
        rt->create_index_space(ctx, KernelRect{0, kernel_size - 1});

    const Legion::FieldSpace kernel_field_space = create_field_space(
        ctx, rt, {sizeof(ENTRY_T), sizeof(DomainIndex)}, {FID_ENTRY, FID_COL}
    );

    const Legion::LogicalRegion kernel_region =
        rt->create_logical_region(ctx, kernel_space, kernel_field_space);

    const Legion::IndexPartition kernel_partition =
        rt->create_equal_partition(ctx, kernel_space, launch_space);

    typename LegionSolvers::FillCSRNegativeLaplacianTask<
        ENTRY_T,
        KERNEL_DIM,
        DOMAIN_DIM,
        RANGE_DIM,
        KERNEL_COORD_T,
        DOMAIN_COORD_T,
        RANGE_COORD_T>::Args kernel_args;
    kernel_args.fid_entry = FID_ENTRY;
    kernel_args.fid_col = FID_COL;
    kernel_args.grid_shape[0] = grid_size;

    Legion::IndexTaskLauncher kernel_launcher(
        LegionSolvers::FillCSRNegativeLaplacianTask<
            ENTRY_T,
            KERNEL_DIM,
            DOMAIN_DIM,
            RANGE_DIM,
            KERNEL_COORD_T,
            DOMAIN_COORD_T,
            RANGE_COORD_T>::task_id,
        launch_space,
        Legion::TaskArgument(&kernel_args, sizeof(kernel_args)),
        Legion::ArgumentMap()
    );

    kernel_launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    kernel_launcher.add_region_requirement(Legion::RegionRequirement(
        rt->get_logical_partition(kernel_region, kernel_partition),
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        kernel_region
    ));
    kernel_launcher.add_field(0, FID_COL);
    kernel_launcher.add_field(0, FID_ENTRY);
    rt->execute_index_space(ctx, kernel_launcher);

    const Legion::IndexSpace range_space =
        rt->create_index_space(ctx, RangeRect{0, grid_size - 1});

    const Legion::FieldSpace rowptr_field_space =
        create_field_space(ctx, rt, {sizeof(KernelRect)}, {FID_ROWPTR});

    const Legion::LogicalRegion rowptr_region =
        rt->create_logical_region(ctx, range_space, rowptr_field_space);

    const Legion::IndexPartition rowptr_partition =
        rt->create_equal_partition(ctx, range_space, launch_space);

    typename LegionSolvers::FillCSRNegativeLaplacianRowptrTask<
        ENTRY_T,
        KERNEL_DIM,
        DOMAIN_DIM,
        RANGE_DIM,
        KERNEL_COORD_T,
        DOMAIN_COORD_T,
        RANGE_COORD_T>::Args rowptr_args;
    rowptr_args.fid_rowptr = FID_ROWPTR;
    rowptr_args.grid_shape[0] = grid_size;

    Legion::IndexTaskLauncher rowptr_launcher(
        LegionSolvers::FillCSRNegativeLaplacianRowptrTask<
            ENTRY_T,
            KERNEL_DIM,
            DOMAIN_DIM,
            RANGE_DIM,
            KERNEL_COORD_T,
            DOMAIN_COORD_T,
            RANGE_COORD_T>::task_id,
        launch_space,
        Legion::TaskArgument(&rowptr_args, sizeof(rowptr_args)),
        Legion::ArgumentMap()
    );

    rowptr_launcher.map_id = LegionSolvers::LEGION_SOLVERS_MAPPER_ID;
    rowptr_launcher.add_region_requirement(Legion::RegionRequirement(
        rt->get_logical_partition(rowptr_region, rowptr_partition),
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        rowptr_region
    ));
    rowptr_launcher.add_field(0, FID_ROWPTR);
    rt->execute_index_space(ctx, rowptr_launcher);

    CSRMatrix<ENTRY_T> result(
        ctx, rt, kernel_region, FID_ENTRY, FID_COL, rowptr_region, FID_ROWPTR
    );

    rt->destroy_index_partition(ctx, rowptr_partition);
    rt->destroy_logical_region(ctx, rowptr_region);
    rt->destroy_field_space(ctx, rowptr_field_space);
    rt->destroy_index_space(ctx, range_space);

    rt->destroy_index_partition(ctx, kernel_partition);
    rt->destroy_logical_region(ctx, kernel_region);
    rt->destroy_field_space(ctx, kernel_field_space);
    rt->destroy_index_space(ctx, kernel_space);

    return result;
}


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
        row_writer(matrix, args.fid_row);
    const AffineWriter<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
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
        row_writer[p] = Legion::Point<RANGE_DIM, RANGE_COORD_T>(
            static_cast<RANGE_COORD_T>((k + 1) / 3)
        );
        col_writer[p] = Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>(
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


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    template COOMatrix<float> LegionSolvers::coo_negative_laplacian_1d<float>(Legion::Context, Legion::Runtime *, Legion::coord_t, Legion::IndexSpace);
    template CSRMatrix<float> LegionSolvers::csr_negative_laplacian_1d<float>(Legion::Context, Legion::Runtime *, Legion::coord_t, Legion::IndexSpace);
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    template COOMatrix<double> LegionSolvers::coo_negative_laplacian_1d<double>(Legion::Context, Legion::Runtime *, Legion::coord_t, Legion::IndexSpace);
    template CSRMatrix<double> LegionSolvers::csr_negative_laplacian_1d<double>(Legion::Context, Legion::Runtime *, Legion::coord_t, Legion::IndexSpace);
#endif // LEGION_SOLVERS_USE_F64
// clang-format on


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
