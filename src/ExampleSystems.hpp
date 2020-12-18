#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <legion.h>

#include "TaskIDs.hpp"


constexpr Legion::coord_t GRID_HEIGHT = 5;
constexpr Legion::coord_t GRID_WIDTH = 5;


enum COOMatrixFieldIDs : Legion::FieldID {
    FID_COO_I = 101,
    FID_COO_J = 102,
    FID_COO_ENTRY = 103,
};


enum VectorFieldIDs : Legion::FieldID {
    FID_VEC_ENTRY = 200,
};


namespace LegionSolvers {


    constexpr Legion::coord_t laplacian_2d_kernel_size(Legion::coord_t height, Legion::coord_t width) {
        return 8 + (height - 2) * 2 * 3 + (width - 2) * 2 * 3 + (height - 2) * (width - 2) * 4 + width * height;
    }


    template <typename T>
    struct FillCOONegativeLaplacian2DTask : public TaskT<FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "fill_coo_negative_laplacian_2d"; }

        struct Args {
            Legion::FieldID fid_i;
            Legion::FieldID fid_j;
            Legion::FieldID fid_entry;
            Legion::coord_t grid_height;
            Legion::coord_t grid_width;
        };

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &matrix = regions[0];

            assert(task->arglen == sizeof(Args));
            const Args args = *reinterpret_cast<const Args *>(task->args);

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> i_writer{matrix, args.fid_i};
            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<2>, 1> j_writer{matrix, args.fid_j};
            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1> entry_writer{matrix, args.fid_entry};

            Legion::PointInDomainIterator<1> iter{matrix};
            for (Legion::coord_t i = 0; i < args.grid_height; ++i) {
                for (Legion::coord_t j = 0; j < args.grid_width; ++j) {

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

                    if (i + 1 < args.grid_height) {
                        i_writer[*iter] = Legion::Point<2>{i, j};
                        j_writer[*iter] = Legion::Point<2>{i + 1, j};
                        entry_writer[*iter] = -1.0;
                        ++iter;
                    }

                    if (j + 1 < args.grid_width) {
                        i_writer[*iter] = Legion::Point<2>{i, j};
                        j_writer[*iter] = Legion::Point<2>{i, j + 1};
                        entry_writer[*iter] = -1.0;
                        ++iter;
                    }
                }
            }
        }

    }; // struct FillCOONegativeLaplacian2DTask


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
