#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <int DIM>
    Legion::LogicalRegionT<DIM> create_region(Legion::IndexSpaceT<DIM> index_space,
                                              const std::vector<std::pair<std::size_t, Legion::FieldID>> &fields,
                                              Legion::Context ctx,
                                              Legion::Runtime *rt) {
        Legion::FieldSpace field_space = rt->create_field_space(ctx);
        Legion::FieldAllocator allocator = rt->create_field_allocator(ctx, field_space);
        for (const auto [field_size, field_id] : fields) { allocator.allocate_field(field_size, field_id); }
        return rt->create_logical_region(ctx, index_space, field_space);
    }


    constexpr Legion::coord_t laplacian_1d_kernel_size(Legion::coord_t length) { return 3 * length - 2; }


    template <typename T>
    struct FillCOONegativeLaplacian1DTask : public TaskT<FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "fill_coo_negative_laplacian_1d"; }

        struct Args {
            Legion::FieldID fid_i;
            Legion::FieldID fid_j;
            Legion::FieldID fid_entry;
            Legion::coord_t grid_length;
        };

        static void task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx,
                         Legion::Runtime *rt) {

            assert(regions.size() == 1);
            const auto &matrix = regions[0];

            assert(task->arglen == sizeof(Args));
            const Args args = *reinterpret_cast<const Args *>(task->args);

            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1> i_writer{matrix, args.fid_i};
            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, Legion::Point<1>, 1> j_writer{matrix, args.fid_j};
            const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, 1> entry_writer{matrix, args.fid_entry};


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
        }

    }; // struct FillCOONegativeLaplacian1DTask


    template <typename T>
    Legion::LogicalRegionT<1> coo_negative_laplacian_1d(Legion::FieldID fid_i,
                                                        Legion::FieldID fid_j,
                                                        Legion::FieldID fid_entry,
                                                        Legion::coord_t length,
                                                        Legion::Context ctx,
                                                        Legion::Runtime *rt) {

        const Legion::coord_t kernel_size = laplacian_1d_kernel_size(length);
        const Legion::LogicalRegionT<1> negative_laplacian = create_region(
            rt->create_index_space(ctx, Legion::Rect<1>{0, kernel_size - 1}),
            {{sizeof(Legion::Point<1>), fid_i}, {sizeof(Legion::Point<1>), fid_j}, {sizeof(T), fid_entry}}, ctx, rt);
        const typename FillCOONegativeLaplacian1DTask<T>::Args args{fid_i, fid_j, fid_entry, length};
        Legion::TaskLauncher launcher{FillCOONegativeLaplacian1DTask<T>::task_id,
                                      Legion::TaskArgument{&args, sizeof(args)}};
        launcher.add_region_requirement(
            Legion::RegionRequirement{negative_laplacian, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, negative_laplacian});
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_field(0, fid_i);
        launcher.add_field(0, fid_j);
        launcher.add_field(0, fid_entry);
        rt->execute_task(ctx, launcher);
        return negative_laplacian;
    }


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
        }

    }; // struct FillCOONegativeLaplacian2DTask


    template <typename T>
    Legion::LogicalRegionT<1> coo_negative_laplacian_2d(Legion::FieldID fid_i,
                                                        Legion::FieldID fid_j,
                                                        Legion::FieldID fid_entry,
                                                        Legion::coord_t height,
                                                        Legion::coord_t width,
                                                        Legion::Context ctx,
                                                        Legion::Runtime *rt) {

        const Legion::coord_t kernel_size = laplacian_2d_kernel_size(height, width);
        const Legion::LogicalRegionT<1> negative_laplacian = create_region(
            rt->create_index_space(ctx, Legion::Rect<1>{0, kernel_size - 1}),
            {{sizeof(Legion::Point<2>), fid_i}, {sizeof(Legion::Point<2>), fid_j}, {sizeof(T), fid_entry}}, ctx, rt);
        const typename FillCOONegativeLaplacian2DTask<T>::Args args{fid_i, fid_j, fid_entry, height, width};
        Legion::TaskLauncher launcher{FillCOONegativeLaplacian2DTask<T>::task_id,
                                      Legion::TaskArgument{&args, sizeof(args)}};
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(
            Legion::RegionRequirement{negative_laplacian, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, negative_laplacian});
        launcher.add_field(0, fid_i);
        launcher.add_field(0, fid_j);
        launcher.add_field(0, fid_entry);
        rt->execute_task(ctx, launcher);
        return negative_laplacian;
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
