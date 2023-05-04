#include "StencilGenerator.hpp"

#include <algorithm> // for std::sort

using LegionSolvers::FillCOOStencilTask;
using LegionSolvers::FillCSRStencilTask;
using LegionSolvers::FillLinearizedCOOStencilTask;
using LegionSolvers::FillLinearizedCSRStencilTask;


template <typename ENTRY_T, int DIM, typename COORD_T>
void FillCOOStencilTask<ENTRY_T, DIM, COORD_T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen >= sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    using StencilData = std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>;

    assert((task->arglen - sizeof(Args)) % sizeof(StencilData) == 0);
    const std::size_t num_offsets =
        (task->arglen - sizeof(Args)) / sizeof(StencilData);
    std::vector<StencilData> offsets;
    offsets.reserve(num_offsets);
    const StencilData *offset_ptr = reinterpret_cast<const StencilData *>(
        reinterpret_cast<const char *>(task->args) + sizeof(Args)
    );
    for (std::size_t i = 0; i < num_offsets; ++i) {
        offsets.push_back(offset_ptr[i]);
    }
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_row_major(p.first, q.first)) { return true; }
                if (compare_row_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    case IndexOrder::COLUMN_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_column_major(p.first, q.first)) { return true; }
                if (compare_column_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    }

    constexpr int KERNEL_DIM = 1;
    constexpr int DOMAIN_DIM = DIM;
    constexpr int RANGE_DIM = DIM;

    using KERNEL_COORD_T = Legion::coord_t; // TODO
    using DOMAIN_COORD_T = COORD_T;
    using RANGE_COORD_T = COORD_T;

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

    Legion::coord_t kernel_index = 0;
    Legion::Point<DIM, COORD_T> point = args.bounds.lo;
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        do {
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        row_writer[kernel_index] = point;
                        col_writer[kernel_index] = shifted;
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
        } while (increment_row_major(point, args.bounds));
        break;
    case IndexOrder::COLUMN_MAJOR:
        do {
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        row_writer[kernel_index] = point;
                        col_writer[kernel_index] = shifted;
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
        } while (increment_column_major(point, args.bounds));
        break;
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void FillLinearizedCOOStencilTask<ENTRY_T, DIM, COORD_T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->regions.size() == 1);
    const auto &matrix_req = task->regions[0];

    assert(task->arglen >= sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    using StencilData = std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>;

    assert((task->arglen - sizeof(Args)) % sizeof(StencilData) == 0);
    const std::size_t num_offsets =
        (task->arglen - sizeof(Args)) / sizeof(StencilData);
    std::vector<StencilData> offsets;
    offsets.reserve(num_offsets);
    const StencilData *offset_ptr = reinterpret_cast<const StencilData *>(
        reinterpret_cast<const char *>(task->args) + sizeof(Args)
    );
    for (std::size_t i = 0; i < num_offsets; ++i) {
        offsets.push_back(offset_ptr[i]);
    }
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_row_major(p.first, q.first)) { return true; }
                if (compare_row_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    case IndexOrder::COLUMN_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_column_major(p.first, q.first)) { return true; }
                if (compare_column_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    }

    constexpr int KERNEL_DIM = 1;

    using KERNEL_COORD_T = Legion::coord_t; // TODO
    using DOMAIN_COORD_T = COORD_T;
    using RANGE_COORD_T = COORD_T;

    const AffineWriter<RANGE_COORD_T, KERNEL_DIM, KERNEL_COORD_T> row_writer(
        matrix, args.fid_row
    );
    const AffineWriter<DOMAIN_COORD_T, KERNEL_DIM, KERNEL_COORD_T> col_writer(
        matrix, args.fid_col
    );
    const AffineWriter<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_writer(
        matrix, args.fid_entry
    );

    const Legion::Domain matrix_domain =
        rt->get_index_space_domain(ctx, matrix_req.region.get_index_space());

    Legion::coord_t kernel_index = 0;
    Legion::Point<DIM, COORD_T> point = args.bounds.lo;
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        do {
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        row_writer[kernel_index] =
                            linearize_row_major(point, args.bounds);
                        col_writer[kernel_index] =
                            linearize_row_major(shifted, args.bounds);
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
        } while (increment_row_major(point, args.bounds));
        break;
    case IndexOrder::COLUMN_MAJOR:
        do {
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        row_writer[kernel_index] =
                            linearize_column_major(point, args.bounds);
                        col_writer[kernel_index] =
                            linearize_column_major(shifted, args.bounds);
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
        } while (increment_column_major(point, args.bounds));
        break;
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void FillCSRStencilTask<ENTRY_T, DIM, COORD_T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {

    assert(regions.size() == 2);
    const auto &matrix = regions[0];
    const auto &rowptr = regions[1];

    assert(task->regions.size() == 2);
    const auto &matrix_req = task->regions[0];
    const auto &rowptr_req = task->regions[1];

    assert(task->arglen >= sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    using StencilData = std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>;

    assert((task->arglen - sizeof(Args)) % sizeof(StencilData) == 0);
    const std::size_t num_offsets =
        (task->arglen - sizeof(Args)) / sizeof(StencilData);
    std::vector<StencilData> offsets;
    offsets.reserve(num_offsets);
    const StencilData *offset_ptr = reinterpret_cast<const StencilData *>(
        reinterpret_cast<const char *>(task->args) + sizeof(Args)
    );
    for (std::size_t i = 0; i < num_offsets; ++i) {
        offsets.push_back(offset_ptr[i]);
    }
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_row_major(p.first, q.first)) { return true; }
                if (compare_row_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    case IndexOrder::COLUMN_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_column_major(p.first, q.first)) { return true; }
                if (compare_column_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    }

    constexpr int KERNEL_DIM = 1;
    constexpr int DOMAIN_DIM = DIM;
    constexpr int RANGE_DIM = DIM;

    using KERNEL_COORD_T = Legion::coord_t; // TODO
    using DOMAIN_COORD_T = COORD_T;
    using RANGE_COORD_T = COORD_T;

    const AffineWriter<
        Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>,
        KERNEL_DIM,
        KERNEL_COORD_T>
        col_writer(matrix, args.fid_col);
    const AffineWriter<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_writer(
        matrix, args.fid_entry
    );

    const AffineWriter<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_writer(rowptr, args.fid_rowptr);

    const Legion::Domain matrix_domain =
        rt->get_index_space_domain(ctx, matrix_req.region.get_index_space());
    const Legion::Domain rowptr_domain =
        rt->get_index_space_domain(ctx, rowptr_req.region.get_index_space());

    Legion::coord_t kernel_index = 0;
    Legion::Point<DIM, COORD_T> point = args.bounds.lo;
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        do {
            const Legion::coord_t row_begin = kernel_index;
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        col_writer[kernel_index] = shifted;
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
            const Legion::coord_t row_end = kernel_index - 1;
            if (rowptr_domain.contains(point)) {
                rowptr_writer[point] = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>{
                    row_begin, row_end};
            }
        } while (increment_row_major(point, args.bounds));
        break;
    case IndexOrder::COLUMN_MAJOR:
        do {
            const Legion::coord_t row_begin = kernel_index;
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        col_writer[kernel_index] = shifted;
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
            const Legion::coord_t row_end = kernel_index - 1;
            if (rowptr_domain.contains(point)) {
                rowptr_writer[point] = Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>{
                    row_begin, row_end};
            }
        } while (increment_column_major(point, args.bounds));
        break;
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void FillLinearizedCSRStencilTask<ENTRY_T, DIM, COORD_T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt
) {
    assert(regions.size() == 2);
    const auto &matrix = regions[0];
    const auto &rowptr = regions[1];

    assert(task->regions.size() == 2);
    const auto &matrix_req = task->regions[0];
    const auto &rowptr_req = task->regions[1];

    assert(task->arglen >= sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    using StencilData = std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>;

    assert((task->arglen - sizeof(Args)) % sizeof(StencilData) == 0);
    const std::size_t num_offsets =
        (task->arglen - sizeof(Args)) / sizeof(StencilData);
    std::vector<StencilData> offsets;
    offsets.reserve(num_offsets);
    const StencilData *offset_ptr = reinterpret_cast<const StencilData *>(
        reinterpret_cast<const char *>(task->args) + sizeof(Args)
    );
    for (std::size_t i = 0; i < num_offsets; ++i) {
        offsets.push_back(offset_ptr[i]);
    }
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_row_major(p.first, q.first)) { return true; }
                if (compare_row_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    case IndexOrder::COLUMN_MAJOR:
        std::sort(
            offsets.begin(),
            offsets.end(),
            [](const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &p,
               const std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T> &q) {
                if (compare_column_major(p.first, q.first)) { return true; }
                if (compare_column_major(q.first, p.first)) { return false; }
                return (p.second < q.second);
            }
        );
        break;
    }

    constexpr int KERNEL_DIM = 1;
    constexpr int RANGE_DIM = 1;

    using KERNEL_COORD_T = Legion::coord_t; // TODO
    using DOMAIN_COORD_T = COORD_T;
    using RANGE_COORD_T = COORD_T;

    const AffineWriter<DOMAIN_COORD_T, KERNEL_DIM, KERNEL_COORD_T> col_writer(
        matrix, args.fid_col
    );
    const AffineWriter<ENTRY_T, KERNEL_DIM, KERNEL_COORD_T> entry_writer(
        matrix, args.fid_entry
    );

    const AffineWriter<
        Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>,
        RANGE_DIM,
        RANGE_COORD_T>
        rowptr_writer(rowptr, args.fid_rowptr);

    const Legion::Domain matrix_domain =
        rt->get_index_space_domain(ctx, matrix_req.region.get_index_space());
    const Legion::Domain rowptr_domain =
        rt->get_index_space_domain(ctx, rowptr_req.region.get_index_space());

    const auto [lower_bound, upper_bound] = leading_dimension_bounds(offsets);

    Legion::coord_t kernel_index = 0;
    Legion::coord_t bulk_slice_origin = 0;
    bool found_bulk_slice = false;
    Legion::Point<DIM, COORD_T> point = args.bounds.lo;
    switch (args.order) {
    case IndexOrder::ROW_MAJOR:
        do {
            if (tail_equal(point, args.bounds.lo)) {
                if (found_bulk_slice) {
                    const Legion::coord_t bulk_slice_size =
                        kernel_index - bulk_slice_origin;
                    while ((point[0] + lower_bound >= args.bounds.lo[0]) &&
                           (point[0] + upper_bound <= args.bounds.hi[0]) &&
                           (kernel_index + bulk_slice_size <
                            matrix_domain.lo()[0]) &&
                           (linearize_row_major(
                                increment_head(point), args.bounds
                            ) < rowptr_domain.lo()[0])) {
                        ++point[0];
                        kernel_index += bulk_slice_size;
                    }
                }
                if ((!found_bulk_slice) &&
                    (point[0] + lower_bound >= args.bounds.lo[0]) &&
                    (point[0] + upper_bound <= args.bounds.hi[0])) {
                    found_bulk_slice = true;
                    bulk_slice_origin = kernel_index;
                }
            }
            const COORD_T point_lin = linearize_row_major(point, args.bounds);
            const Legion::coord_t row_begin = kernel_index;
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        col_writer[kernel_index] =
                            linearize_row_major(shifted, args.bounds);
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
            const Legion::coord_t row_end = kernel_index - 1;
            if (rowptr_domain.contains(point_lin)) {
                rowptr_writer[point_lin] =
                    Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>{
                        row_begin, row_end};
            }
            if ((kernel_index > matrix_domain.hi()[0]) &&
                (point_lin > rowptr_domain.hi()[0])) {
                break;
            }
        } while (increment_row_major(point, args.bounds));
        break;
    case IndexOrder::COLUMN_MAJOR:
        do {
            const COORD_T point_lin =
                linearize_column_major(point, args.bounds);
            const Legion::coord_t row_begin = kernel_index;
            for (const auto [offset, entry] : offsets) {
                const auto shifted = point + offset;
                if (args.bounds.contains(shifted)) {
                    if (matrix_domain.contains(kernel_index)) {
                        col_writer[kernel_index] =
                            linearize_column_major(shifted, args.bounds);
                        entry_writer[kernel_index] = entry;
                    }
                    ++kernel_index;
                }
            }
            const Legion::coord_t row_end = kernel_index - 1;
            if (rowptr_domain.contains(point_lin)) {
                rowptr_writer[point_lin] =
                    Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>{
                        row_begin, row_end};
            }
        } while (increment_column_major(point, args.bounds));
        break;
    }
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void FillCOOStencilTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void FillCOOStencilTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void FillCOOStencilTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillCSRStencilTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCOOStencilTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void FillLinearizedCSRStencilTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
