#ifndef LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP
#define LEGION_SOLVERS_EXAMPLE_SYSTEMS_HPP

#include <iostream>

#include <legion.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    template <int DIM>
    Legion::LogicalRegionT<DIM> create_region(
        Legion::IndexSpaceT<DIM> index_space,
        const std::vector<std::pair<std::size_t, Legion::FieldID>> &fields,
        Legion::Context ctx, Legion::Runtime *rt
    ) {
        Legion::FieldSpace field_space = rt->create_field_space(ctx);
        Legion::FieldAllocator allocator = rt->create_field_allocator(ctx, field_space);
        for (const auto [field_size, field_id] : fields) { allocator.allocate_field(field_size, field_id); }
        return rt->create_logical_region(ctx, index_space, field_space);
    }


    constexpr Legion::coord_t laplacian_1d_kernel_size(Legion::coord_t length) { return 3 * length - 2; }


    template <typename T>
    struct FillCOONegativeLaplacian1DTask : public TaskT<FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "fill_coo_negative_laplacian_1d"; }



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


    template <typename T>
    struct FillCOONegativeLaplacian2DTask : public TaskT<FILL_COO_NEGATIVE_LAPLACIAN_2D_TASK_BLOCK_ID, T> {

        static std::string task_name() { return "fill_coo_negative_laplacian_2d"; }



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
