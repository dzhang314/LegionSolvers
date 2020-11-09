#ifndef LEGION_SOLVERS_PLANNER_HPP
#define LEGION_SOLVERS_PLANNER_HPP

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <legion.h>

#include "COOMatrix.hpp"


namespace LegionSolvers {


    class Planner {


      public:
        std::vector<std::pair<Legion::IndexSpace, Legion::IndexPartition>>
            dimensions;
        std::vector<std::pair<Legion::LogicalRegion, Legion::FieldID>>
            right_hand_sides;
        std::vector<std::tuple<int, int, COOMatrix>> operators;


        void add_rhs(Legion::LogicalRegion rhs_region, Legion::FieldID fid_rhs,
                     Legion::IndexPartition partition) {
            right_hand_sides.emplace_back(rhs_region, fid_rhs);
            dimensions.emplace_back(rhs_region.get_index_space(), partition);
        }


        void add_coo_matrix(int rhs_index, int sol_index,
                            Legion::LogicalRegion matrix_region,
                            Legion::FieldID fid_i, Legion::FieldID fid_j,
                            Legion::FieldID fid_entry, Legion::Context ctx,
                            Legion::Runtime *rt) {
            operators.emplace_back(
                rhs_index, sol_index,
                COOMatrix{matrix_region, fid_i, fid_j, fid_entry,
                          dimensions[sol_index].second,
                          dimensions[rhs_index].second, ctx, rt});
        }


        void matvec(Legion::FieldID fid_dst, Legion::FieldID fid_src,
                    const std::vector<Legion::LogicalRegion> &workspace,
                    Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (const auto &[dst_index, src_index, matrix] : operators) {
                matrix.launch_matvec(workspace[dst_index], fid_dst,
                                     workspace[src_index], fid_src, ctx, rt);
            }
        }


        void copy_rhs(Legion::FieldID fid_dst,
                      const std::vector<Legion::LogicalRegion> &workspace,
                      Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    COPY_TASK_ID,
                    rt->get_index_partition_color_space_name(
                        dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_dst);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, right_hand_sides[i].first,
                                              dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                    right_hand_sides[i].first});
                launcher.add_field(1, right_hand_sides[i].second);
                rt->execute_index_space(ctx, launcher);
            }
        }


        void zero_fill(Legion::FieldID fid_dst,
                       const std::vector<Legion::LogicalRegion> &workspace,
                       Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    ZERO_FILL_TASK_ID,
                    rt->get_index_partition_color_space_name(
                        dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_dst);
                rt->execute_index_space(ctx, launcher);
            }
        }


        Legion::Future
        dot_product(Legion::FieldID fid_v, Legion::FieldID fid_w,
                    const std::vector<Legion::LogicalRegion> &workspace,
                    Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            Legion::Future result = Legion::Future::from_value<double>(rt, 0.0);
            for (std::size_t i = 0; i < workspace.size(); ++i) {
                // TODO: Implement inner reduction.
                Legion::TaskLauncher launcher{DOT_PRODUCT_TASK_ID,
                                              Legion::TaskArgument{nullptr, 0}};
                launcher.add_region_requirement(
                    Legion::RegionRequirement{workspace[i], LEGION_READ_ONLY,
                                              LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_v);
                launcher.add_region_requirement(
                    Legion::RegionRequirement{workspace[i], LEGION_READ_ONLY,
                                              LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_w);
                result = add(result, rt->execute_task(ctx, launcher), ctx, rt);
            }
            return result;
        }


        void axpy(Legion::FieldID fid_y, Legion::Future alpha,
                  Legion::FieldID fid_x,
                  const std::vector<Legion::LogicalRegion> &workspace,
                  Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    AXPY_TASK_ID,
                    rt->get_index_partition_color_space_name(
                        dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


        void xpay(Legion::FieldID fid_y, Legion::Future alpha,
                  Legion::FieldID fid_x,
                  const std::vector<Legion::LogicalRegion> &workspace,
                  Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == right_hand_sides.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    XPAY_TASK_ID,
                    rt->get_index_partition_color_space_name(
                        dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}};
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i],
                                              dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


        Legion::Future add(Legion::Future a, Legion::Future b,
                           Legion::Context ctx, Legion::Runtime *rt) const {
            Legion::TaskLauncher launcher{ADDITION_TASK_ID,
                                          Legion::TaskArgument{nullptr, 0}};
            launcher.add_future(a);
            launcher.add_future(b);
            return rt->execute_task(ctx, launcher);
        }


        Legion::Future divide(Legion::Future numer, Legion::Future denom,
                              Legion::Context ctx, Legion::Runtime *rt) const {
            Legion::TaskLauncher launcher{DIVISION_TASK_ID,
                                          Legion::TaskArgument{nullptr, 0}};
            launcher.add_future(numer);
            launcher.add_future(denom);
            return rt->execute_task(ctx, launcher);
        }


        Legion::Future negate(Legion::Future x, Legion::Context ctx,
                              Legion::Runtime *rt) const {
            Legion::TaskLauncher launcher{NEGATION_TASK_ID,
                                          Legion::TaskArgument{nullptr, 0}};
            launcher.add_future(x);
            return rt->execute_task(ctx, launcher);
        }


    }; // class Planner


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_PLANNER_HPP
