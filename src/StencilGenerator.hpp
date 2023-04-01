#ifndef LEGION_SOLVERS_STENCIL_GENERATOR_HPP_INCLUDED
#define LEGION_SOLVERS_STENCIL_GENERATOR_HPP_INCLUDED

#include <algorithm> // for std::min, std::max
#include <cmath>     // for std::abs
#include <cstddef>   // for std::size_t
#include <cstring>   // for std::memcpy
#include <utility>   // for std::pair
#include <vector>    // for std::vector

#include <legion.h> // for Legion::*

#include "COOMatrix.hpp"       // for COOMatrix
#include "CSRMatrix.hpp"       // for CSRMatrix
#include "LegionUtilities.hpp" // for TaskFlags, LEGION_SOLVERS_DECLARE_TASK
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_MAPPER_ID
#include "TaskBaseClasses.hpp" // for TaskTDI
#include "TaskIDs.hpp"         // for *_TASK_BLOCK_ID

namespace LegionSolvers {


enum class IndexOrder {
    ROW_MAJOR,
    COLUMN_MAJOR,
}; // enum class IndexOrder


template <typename ENTRY_T, int DIM, typename COORD_T>
struct FillCOOStencilTask : public TaskTDI<
                                FILL_COO_STENCIL_TASK_BLOCK_ID,
                                FillCOOStencilTask,
                                ENTRY_T,
                                DIM,
                                COORD_T> {

    static constexpr const char *task_base_name = "fill_coo_stencil";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_row;
        Legion::FieldID fid_col;
        Legion::Rect<DIM, COORD_T> bounds;
        IndexOrder order;
        bool verbose;
    };

    LEGION_SOLVERS_DECLARE_TASK(void);

}; // struct FillCOOStencilTask


template <typename ENTRY_T, int DIM, typename COORD_T>
struct FillCSRStencilTask : public TaskTDI<
                                FILL_CSR_STENCIL_TASK_BLOCK_ID,
                                FillCSRStencilTask,
                                ENTRY_T,
                                DIM,
                                COORD_T> {

    static constexpr const char *task_base_name = "fill_csr_stencil";

    static constexpr const TaskFlags flags =
        TaskFlags::LEAF | TaskFlags::IDEMPOTENT | TaskFlags::REPLICABLE;

    struct Args {
        Legion::FieldID fid_entry;
        Legion::FieldID fid_col;
        Legion::FieldID fid_rowptr;
        Legion::Rect<DIM, COORD_T> bounds;
        IndexOrder order;
        bool verbose;
    };

    LEGION_SOLVERS_DECLARE_TASK(void);

}; // struct FillCSRStencilTask


template <int DIM, typename COORD_T>
constexpr bool compare_row_major(
    const Legion::Point<DIM, COORD_T> &p, const Legion::Point<DIM, COORD_T> &q
) {
    for (int i = 0; i < DIM; ++i) {
        if (p[i] < q[i]) { return true; }
        if (p[i] > q[i]) { return false; }
    }
    return false;
}


template <int DIM, typename COORD_T>
constexpr bool compare_column_major(
    const Legion::Point<DIM, COORD_T> &p, const Legion::Point<DIM, COORD_T> &q
) {
    for (int i = DIM - 1; i >= 0; --i) {
        if (p[i] < q[i]) { return true; }
        if (p[i] > q[i]) { return false; }
    }
    return false;
}


template <int DIM, typename COORD_T>
constexpr bool increment_row_major(
    Legion::Point<DIM, COORD_T> &point, const Legion::Rect<DIM, COORD_T> &bounds
) {
    for (int i = DIM - 1; i >= 0; --i) {
        if (point[i] >= bounds.hi[i]) {
            point[i] = bounds.lo[i];
        } else {
            ++point[i];
            return true;
        }
    }
    return false;
}


template <int DIM, typename COORD_T>
bool increment_column_major(
    Legion::Point<DIM, COORD_T> &point, const Legion::Rect<DIM, COORD_T> &bounds
) {
    for (int i = 0; i < DIM; ++i) {
        if (point[i] >= bounds.hi[i]) {
            point[i] = bounds.lo[i];
        } else {
            ++point[i];
            return true;
        }
    }
    return false;
}


template <int DIM, typename COORD_T>
Legion::Point<DIM - 1, COORD_T> tail(const Legion::Point<DIM, COORD_T> &p) {
    Legion::Point<DIM - 1, COORD_T> result;
    for (int i = 0; i < DIM - 1; ++i) { result[i] = p[i + 1]; }
    return result;
}


template <typename ENTRY_T, int DIM, typename COORD_T>
std::size_t calculate_stencil_size(
    const Legion::Rect<DIM, COORD_T> &bounds,
    const std::vector<std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>> &offsets
) {
    std::size_t result = static_cast<std::size_t>(0);
    if (offsets.empty()) { return result; }
    // At this point, we know that offsets is non-empty.
    // We proceed by induction on DIM.
    if constexpr (DIM == 1) {
        // Base case: Directly calculate the size of a 1-dimensional stencil.
        const std::size_t length = bounds.volume();
        for (const auto &[offset, entry] : offsets) {
            const std::size_t distance =
                static_cast<std::size_t>(std::abs(offset[0]));
            if (distance <= length) { result += length - distance; }
        }
    } else {
        // Inductive case: calculate the size of an (N + 1)-dimensional stencil
        // by slicing it into N-dimensional stencils and adding up their sizes.
        COORD_T min_offset = static_cast<COORD_T>(0); // look-back distance
        COORD_T max_offset = static_cast<COORD_T>(0); // look-ahead distance
        for (const auto &[offset, entry] : offsets) {
            min_offset = std::min(min_offset, offset[0]);
            max_offset = std::max(max_offset, offset[0]);
        }
        // For size calculation, it does not matter whether we use row-major
        // or column-major indexing. Here, we use row-major indexing.
        for (COORD_T i = bounds.lo[0]; i <= bounds.hi[0]; ++i) {
            Legion::Rect<DIM - 1, COORD_T> sub_bounds(
                tail(bounds.lo), tail(bounds.hi)
            );
            std::vector<std::pair<Legion::Point<DIM - 1, COORD_T>, ENTRY_T>>
                sub_offsets;
            for (const auto &[offset, entry] : offsets) {
                if ((bounds.lo[0] <= i + offset[0]) &&
                    (i + offset[0] <= bounds.hi[0])) {
                    sub_offsets.emplace_back(tail(offset), entry);
                }
            }
            if (sub_offsets.size() == offsets.size()) {
                assert(i == bounds.lo[0] - min_offset);
                const COORD_T i_new = bounds.hi[0] - max_offset;
                assert(i_new >= i);
                result += static_cast<std::size_t>(i_new - i + 1) *
                          calculate_stencil_size(sub_bounds, sub_offsets);
                i = i_new;
            } else {
                result += calculate_stencil_size(sub_bounds, sub_offsets);
            }
        }
    }
    return result;
}


template <typename ENTRY_T, int DIM, typename COORD_T>
COOMatrix<ENTRY_T> create_coo_stencil_matrix(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::Rect<DIM, COORD_T> &bounds,
    const std::vector<std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>> &offsets,
    std::size_t num_pieces,
    IndexOrder index_order = IndexOrder::ROW_MAJOR,
    bool verbose = false
) {
    // create index space
    constexpr int KERNEL_DIM = 1;
    using KERNEL_COORD_T = Legion::coord_t;
    const std::size_t stencil_size = calculate_stencil_size(bounds, offsets);
    const auto index_space = rt->create_index_space(
        ctx, Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>(0, stencil_size - 1)
    );

    // create field space
    constexpr Legion::FieldID FID_ENTRY = 0;
    constexpr Legion::FieldID FID_ROW = 1;
    constexpr Legion::FieldID FID_COL = 2;
    const auto field_space = create_field_space(
        ctx,
        rt,
        {sizeof(ENTRY_T),
         sizeof(Legion::Point<DIM, COORD_T>),
         sizeof(Legion::Point<DIM, COORD_T>)},
        {FID_ENTRY, FID_ROW, FID_COL}
    );

    // create logical region
    const auto matrix_region =
        rt->create_logical_region(ctx, index_space, field_space);

    // create equal partition into `num_pieces` pieces
    constexpr int COLOR_DIM = 1;
    using COLOR_COORD_T = Legion::coord_t;
    const auto color_space = rt->create_index_space(
        ctx, Legion::Rect<COLOR_DIM, COLOR_COORD_T>(0, num_pieces - 1)
    );
    const auto index_partition =
        rt->create_equal_partition(ctx, index_space, color_space);
    const auto logical_partition =
        rt->get_logical_partition(matrix_region, index_partition);

    // construct arguments for stencil fill task
    typename FillCOOStencilTask<ENTRY_T, DIM, COORD_T>::Args args;
    args.fid_entry = FID_ENTRY;
    args.fid_row = FID_ROW;
    args.fid_col = FID_COL;
    args.bounds = bounds;
    args.order = index_order;
    args.verbose = verbose;
    const std::size_t offsets_byte_size =
        offsets.size() *
        sizeof(std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>);
    std::vector<char> arg_buffer(sizeof(args) + offsets_byte_size);
    std::memcpy(arg_buffer.data(), &args, sizeof(args));
    std::memcpy(
        arg_buffer.data() + sizeof(args), offsets.data(), offsets_byte_size
    );

    // launch stencil fill task
    Legion::IndexTaskLauncher launcher(
        FillCOOStencilTask<ENTRY_T, DIM, COORD_T>::task_id,
        color_space,
        Legion::UntypedBuffer(
            arg_buffer.data(), sizeof(args) + offsets_byte_size
        ),
        Legion::ArgumentMap()
    );
    launcher.add_region_requirement(Legion::RegionRequirement(
        logical_partition,
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        matrix_region
    ));
    launcher.add_field(0, FID_ENTRY);
    launcher.add_field(0, FID_ROW);
    launcher.add_field(0, FID_COL);
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

    rt->execute_index_space(ctx, launcher);

    // return a COOMatrix object
    return COOMatrix<ENTRY_T>(
        ctx, rt, matrix_region, FID_ENTRY, FID_ROW, FID_COL
    );
}


template <typename ENTRY_T, int DIM, typename COORD_T>
CSRMatrix<ENTRY_T> create_csr_stencil_matrix(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::Rect<DIM, COORD_T> &bounds,
    const std::vector<std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>> &offsets,
    std::size_t num_pieces,
    IndexOrder index_order = IndexOrder::ROW_MAJOR,
    bool verbose = false
) {
    // create index spaces
    constexpr int KERNEL_DIM = 1;
    using KERNEL_COORD_T = Legion::coord_t;
    const std::size_t stencil_size = calculate_stencil_size(bounds, offsets);
    const auto kernel_space = rt->create_index_space(
        ctx, Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>(0, stencil_size - 1)
    );
    const auto range_space = rt->create_index_space(ctx, bounds);

    // create field spaces
    constexpr Legion::FieldID FID_ENTRY = 0;
    constexpr Legion::FieldID FID_COL = 1;
    const auto kernel_field_space = create_field_space(
        ctx,
        rt,
        {sizeof(ENTRY_T), sizeof(Legion::Point<DIM, COORD_T>)},
        {FID_ENTRY, FID_COL}
    );

    constexpr Legion::FieldID FID_ROWPTR = 0;
    const auto rowptr_field_space = create_field_space(
        ctx,
        rt,
        {sizeof(Legion::Rect<KERNEL_DIM, KERNEL_COORD_T>)},
        {FID_ROWPTR}
    );

    // create logical region
    const auto kernel_region =
        rt->create_logical_region(ctx, kernel_space, kernel_field_space);
    const auto rowptr_region =
        rt->create_logical_region(ctx, range_space, rowptr_field_space);

    // create equal partitions into `num_pieces` pieces
    constexpr int COLOR_DIM = 1;
    using COLOR_COORD_T = Legion::coord_t;
    const auto color_space = rt->create_index_space(
        ctx, Legion::Rect<COLOR_DIM, COLOR_COORD_T>(0, num_pieces - 1)
    );

    const auto kernel_index_partition =
        rt->create_equal_partition(ctx, kernel_space, color_space);
    const auto kernel_logical_partition =
        rt->get_logical_partition(kernel_region, kernel_index_partition);

    const auto rowptr_index_partition =
        rt->create_equal_partition(ctx, range_space, color_space);
    const auto rowptr_logical_partition =
        rt->get_logical_partition(rowptr_region, rowptr_index_partition);

    // construct arguments for stencil fill task
    typename FillCSRStencilTask<ENTRY_T, DIM, COORD_T>::Args args;
    args.fid_entry = FID_ENTRY;
    args.fid_col = FID_COL;
    args.fid_rowptr = FID_ROWPTR;
    args.bounds = bounds;
    args.order = index_order;
    args.verbose = verbose;
    const std::size_t offsets_byte_size =
        offsets.size() *
        sizeof(std::pair<Legion::Point<DIM, COORD_T>, ENTRY_T>);
    std::vector<char> arg_buffer(sizeof(args) + offsets_byte_size);
    std::memcpy(arg_buffer.data(), &args, sizeof(args));
    std::memcpy(
        arg_buffer.data() + sizeof(args), offsets.data(), offsets_byte_size
    );

    // launch stencil fill task
    Legion::IndexTaskLauncher launcher(
        FillCSRStencilTask<ENTRY_T, DIM, COORD_T>::task_id,
        color_space,
        Legion::UntypedBuffer(
            arg_buffer.data(), sizeof(args) + offsets_byte_size
        ),
        Legion::ArgumentMap()
    );
    launcher.add_region_requirement(Legion::RegionRequirement(
        kernel_logical_partition,
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        kernel_region
    ));
    launcher.add_field(0, FID_ENTRY);
    launcher.add_field(0, FID_COL);
    launcher.add_region_requirement(Legion::RegionRequirement(
        rowptr_logical_partition,
        0,
        LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE,
        rowptr_region
    ));
    launcher.add_field(1, FID_ROWPTR);
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

    rt->execute_index_space(ctx, launcher);

    // return a CSRMatrix object
    return CSRMatrix<ENTRY_T>(
        ctx, rt, kernel_region, FID_ENTRY, FID_COL, rowptr_region, FID_ROWPTR
    );
}


} // namespace LegionSolvers

#endif // #define LEGION_SOLVERS_STENCIL_GENERATOR_HPP_INCLUDED
