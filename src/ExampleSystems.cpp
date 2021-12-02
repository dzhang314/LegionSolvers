#include "ExampleSystems.hpp"

#include <iostream>


template <typename T>
void LegionSolvers::FillCOONegativeLaplacian1DTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing COO 1D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1>
    j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1>
    entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_length; ++i) {
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<T>(2.0);
        ++iter;
    }

    for (Legion::coord_t i = 0; i < args.grid_length - 1; ++i) {
        i_writer[*iter] = Legion::Point<1>{i + 1};
        j_writer[*iter] = Legion::Point<1>{i};
        entry_writer[*iter] = static_cast<T>(-1.0);
        ++iter;
        i_writer[*iter] = Legion::Point<1>{i};
        j_writer[*iter] = Legion::Point<1>{i + 1};
        entry_writer[*iter] = static_cast<T>(-1.0);
        ++iter;
    }
    std::cout << "[LegionSolvers] Finished constructing COO 1D Laplacian."
              << std::endl;
}


template <typename T>
void LegionSolvers::FillCOONegativeLaplacian2DTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    std::cout << "[LegionSolvers] Constructing COO 2D Laplacian matrix..."
              << std::endl;

    assert(regions.size() == 1);
    const auto &matrix = regions[0];

    assert(task->arglen == sizeof(Args));
    const Args args = *reinterpret_cast<const Args *>(task->args);

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1>
    i_writer{matrix, args.fid_i};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1>
    j_writer{matrix, args.fid_j};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1>
    entry_writer{matrix, args.fid_entry};

    Legion::PointInDomainIterator<1> iter{matrix};
    for (Legion::coord_t i = 0; i < args.grid_height; ++i) {
        for (Legion::coord_t j = 0; j < args.grid_width; ++j) {
            i_writer[*iter] = Legion::Point<2>{i, j};
            j_writer[*iter] = Legion::Point<2>{i, j};
            entry_writer[*iter] = static_cast<T>(4.0);
            ++iter;
            if (i > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i - 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j - 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (i + 1 < args.grid_height) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i + 1, j};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
            if (j + 1 < args.grid_width) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j + 1};
                entry_writer[*iter] = static_cast<T>(-1.0);
                ++iter;
            }
        }
    }
    std::cout << "[LegionSolvers] Finished constructing COO 2D Laplacian."
              << std::endl;
}


template void LegionSolvers::FillCOONegativeLaplacian1DTask<float >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian1DTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian2DTask<float >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::FillCOONegativeLaplacian2DTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
