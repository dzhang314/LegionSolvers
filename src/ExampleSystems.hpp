#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <legion.h>


constexpr Legion::coord_t GRID_HEIGHT = 100;
constexpr Legion::coord_t GRID_WIDTH = 200;


enum COOMatrixFieldIDs : Legion::FieldID {
    FID_COO_I = 101,
    FID_COO_J = 102,
    FID_COO_ENTRY = 103,
};


enum VectorFieldIDs : Legion::FieldID {
    FID_VEC_ENTRY = 200,
};


constexpr Legion::coord_t laplacian_2d_kernel_size(Legion::coord_t height, Legion::coord_t width) {
    return 8 + (height - 2) * 2 * 3 + (width - 2) * 2 * 3 + (height - 2) * (width - 2) * 4 + width * height;
}


void fill_negative_laplacian_2d_task(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &regions,
                                     Legion::Context ctx,
                                     Legion::Runtime *rt) {

    assert(regions.size() == 1);
    const auto &negative_laplacian_matrix = regions[0];

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> i_writer{negative_laplacian_matrix,
                                                                                    FID_COO_I};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> j_writer{negative_laplacian_matrix,
                                                                                    FID_COO_J};
    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, double, 1> entry_writer{negative_laplacian_matrix, FID_COO_ENTRY};

    Legion::PointInDomainIterator<1> iter{negative_laplacian_matrix};
    for (Legion::coord_t i = 0; i < GRID_HEIGHT; ++i) {
        for (Legion::coord_t j = 0; j < GRID_WIDTH; ++j) {

            i_writer[*iter] = Legion::Point<2>{i, j};
            j_writer[*iter] = Legion::Point<2>{i, j};
            entry_writer[*iter] = 4.0;
            ++iter;

            if (i > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i - 1, j};
                entry_writer[*iter] = -1.0;
                ++iter;
            }

            if (j > 0) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j - 1};
                entry_writer[*iter] = -1.0;
                ++iter;
            }

            if (i + 1 < GRID_HEIGHT) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i + 1, j};
                entry_writer[*iter] = -1.0;
                ++iter;
            }

            if (j + 1 < GRID_WIDTH) {
                i_writer[*iter] = Legion::Point<2>{i, j};
                j_writer[*iter] = Legion::Point<2>{i, j + 1};
                entry_writer[*iter] = -1.0;
                ++iter;
            }
        }
    }
}


#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
