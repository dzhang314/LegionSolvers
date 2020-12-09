#ifndef LEGION_SOLVERS_PLANNER_HPP
#define LEGION_SOLVERS_PLANNER_HPP

#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <legion.h>

#include "COOMatrix.hpp"
#include "LinearAlgebraTasks.hpp"
#include "LinearOperator.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename T>
    class Planner {


        std::vector<std::pair<Legion::IndexSpace, Legion::IndexPartition>> dimensions;
        std::vector<std::pair<Legion::LogicalRegion, Legion::FieldID>> right_hand_sides;
        std::vector<std::tuple<int, int, std::unique_ptr<LinearOperator>>> operators;


      public:
        const std::vector<std::pair<Legion::IndexSpace, Legion::IndexPartition>> &get_dimensions() const noexcept {
            return dimensions;
        }


        std::size_t
        add_rhs(Legion::LogicalRegion rhs_region, Legion::FieldID fid_rhs, Legion::IndexPartition partition) {
            right_hand_sides.emplace_back(rhs_region, fid_rhs);
            dimensions.emplace_back(rhs_region.get_index_space(), partition);
            return right_hand_sides.size() - 1;
        }


        template <int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
        void add_coo_matrix(int rhs_index,
                            int sol_index,
                            Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
                            Legion::FieldID fid_i,
                            Legion::FieldID fid_j,
                            Legion::FieldID fid_entry,
                            Legion::Context ctx,
                            Legion::Runtime *rt) {
            const Legion::IndexPartition domain_partition = dimensions[sol_index].second;
            assert(domain_partition.get_dim() == DOMAIN_DIM);
            const Legion::IndexPartition range_partition = dimensions[rhs_index].second;
            assert(range_partition.get_dim() == RANGE_DIM);
            operators.emplace_back(rhs_index, sol_index,
                                   std::make_unique<COOMatrix<T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>>(
                                       matrix_region, fid_i, fid_j, fid_entry,
                                       Legion::IndexPartitionT<DOMAIN_DIM>{domain_partition},
                                       Legion::IndexPartitionT<RANGE_DIM>{range_partition}, ctx, rt));
        }


        void matvec(Legion::FieldID fid_dst,
                    Legion::FieldID fid_src,
                    const std::vector<Legion::LogicalRegion> &workspace,
                    Legion::Context ctx,
                    Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (const auto &[dst_index, src_index, matrix] : operators) {
                matrix->matvec(workspace[dst_index], fid_dst, workspace[src_index], fid_src, ctx, rt);
            }
        }


        void copy_rhs(Legion::FieldID fid_dst,
                      const std::vector<Legion::LogicalRegion> &workspace,
                      Legion::Context ctx,
                      Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{CopyTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                                               rt->get_index_partition_color_space_name(dimensions[i].second),
                                               Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(
                    Legion::RegionRequirement{rt->get_logical_partition(ctx, workspace[i], dimensions[i].second), 0,
                                              LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_dst);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, right_hand_sides[i].first, dimensions[i].second), 0,
                    LEGION_READ_ONLY, LEGION_EXCLUSIVE, right_hand_sides[i].first});
                launcher.add_field(1, right_hand_sides[i].second);
                rt->execute_index_space(ctx, launcher);
            }
        }


        void zero_fill(Legion::FieldID fid_dst,
                       const std::vector<Legion::LogicalRegion> &workspace,
                       Legion::Context ctx,
                       Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                LegionSolvers::zero_fill<T>(workspace[i], fid_dst, dimensions[i].second, ctx, rt);
            }
        }


        Legion::Future dot_product(Legion::FieldID fid_v,
                                   Legion::FieldID fid_w,
                                   const std::vector<Legion::LogicalRegion> &workspace,
                                   Legion::Context ctx,
                                   Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            Legion::Future result = Legion::Future::from_value<T>(rt, 0.0);
            for (std::size_t i = 0; i < workspace.size(); ++i) {
                // TODO: Implement inner reduction.
                Legion::TaskLauncher launcher{DotProductTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                                              Legion::TaskArgument{nullptr, 0}};
                launcher.add_region_requirement(
                    Legion::RegionRequirement{workspace[i], LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_v);
                launcher.add_region_requirement(
                    Legion::RegionRequirement{workspace[i], LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_w);
                result = add<T>(result, rt->execute_task(ctx, launcher), ctx, rt);
            }
            return result;
        }


        void axpy(Legion::FieldID fid_y,
                  Legion::Future alpha,
                  Legion::FieldID fid_x,
                  const std::vector<Legion::LogicalRegion> &workspace,
                  Legion::Context ctx,
                  Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{AxpyTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                                               rt->get_index_partition_color_space_name(dimensions[i].second),
                                               Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(
                    Legion::RegionRequirement{rt->get_logical_partition(ctx, workspace[i], dimensions[i].second), 0,
                                              LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(
                    Legion::RegionRequirement{rt->get_logical_partition(ctx, workspace[i], dimensions[i].second), 0,
                                              LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


        void xpay(Legion::FieldID fid_y,
                  Legion::Future alpha,
                  Legion::FieldID fid_x,
                  const std::vector<Legion::LogicalRegion> &workspace,
                  Legion::Context ctx,
                  Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{XpayTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                                               rt->get_index_partition_color_space_name(dimensions[i].second),
                                               Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(
                    Legion::RegionRequirement{rt->get_logical_partition(ctx, workspace[i], dimensions[i].second), 0,
                                              LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(
                    Legion::RegionRequirement{rt->get_logical_partition(ctx, workspace[i], dimensions[i].second), 0,
                                              LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


    }; // class Planner


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_PLANNER_HPP
