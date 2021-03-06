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
        std::vector<std::pair<Legion::LogicalRegion, Legion::FieldID>> solution_vectors;
        std::vector<std::pair<Legion::LogicalRegion, Legion::FieldID>> rhs_vectors;


      public:
        std::vector<std::tuple<int, int, std::unique_ptr<LinearOperator>>> operators;


        const std::vector<std::pair<Legion::IndexSpace, Legion::IndexPartition>> &
        get_dimensions() const noexcept { return dimensions; }


        std::size_t add_rhs_vector(Legion::LogicalRegion rhs_region,
                                   Legion::FieldID fid_rhs,
                                   Legion::IndexPartition partition) {
            rhs_vectors.emplace_back(rhs_region, fid_rhs);
            dimensions.emplace_back(rhs_region.get_index_space(), partition);
            return rhs_vectors.size() - 1;
        }


        std::size_t add_solution_vector(Legion::LogicalRegion sol_region,
                                        Legion::FieldID fid_sol,
                                        Legion::IndexPartition partition) {
            solution_vectors.emplace_back(sol_region, fid_sol);
            // TODO: Ensure consistent dimensions of RHS and solution vectors
            // dimensions.emplace_back(sol_region.get_index_space(), partition);
            return solution_vectors.size() - 1;
        }


        template <int KERNEL_DIM, int DOMAIN_DIM, int RANGE_DIM>
        void add_coo_matrix(
            int rhs_index, int sol_index,
            Legion::LogicalRegionT<KERNEL_DIM> matrix_region,
            Legion::IndexPartitionT<KERNEL_DIM> matrix_partition,
            Legion::FieldID fid_i, Legion::FieldID fid_j, Legion::FieldID fid_entry,
            Legion::Context ctx, Legion::Runtime *rt
        ) {
            const Legion::IndexPartition domain_partition = dimensions[sol_index].second;
            assert(domain_partition.get_dim() == DOMAIN_DIM);

            const Legion::IndexPartition range_partition = dimensions[rhs_index].second;
            assert(range_partition.get_dim() == RANGE_DIM);

            operators.emplace_back(
                rhs_index, sol_index,
                std::make_unique<COOMatrix<T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>>(
                    matrix_region, fid_i, fid_j, fid_entry, matrix_partition,
                    Legion::IndexPartitionT<DOMAIN_DIM>{domain_partition},
                    Legion::IndexPartitionT<RANGE_DIM>{range_partition},
                    ctx, rt
                )
            );

            const auto matrix = dynamic_cast<
                COOMatrix<T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM> *
            >(std::get<2>(operators.back()).get());
            assert(matrix != nullptr);
        }


        void dummy_task_sol(Legion::Context ctx, Legion::Runtime *rt) const {

            assert(dimensions.size() == solution_vectors.size());

            for (std::size_t i = 0; i < dimensions.size(); ++i) {
                dummy_task<T>(
                    solution_vectors[i].first, solution_vectors[i].second,
                    dimensions[i].second,
                    ctx, rt
                );
            }
        }


        void matvec(Legion::FieldID fid_dst,
                    Legion::FieldID fid_src,
                    const std::vector<Legion::LogicalRegion> &workspace,
                    Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == rhs_vectors.size());

            for (const auto &[dst_index, src_index, matrix] : operators) {
                matrix->matvec(
                    workspace[dst_index], fid_dst,
                    workspace[src_index], fid_src,
                    ctx, rt
                );
            }
        }


        void copy_rhs(Legion::FieldID fid_dst,
                      const std::vector<Legion::LogicalRegion> &workspace,
                      Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == rhs_vectors.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexCopyLauncher launcher{
                    rt->get_index_partition_color_space_name(dimensions[i].second)
                };
                launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
                launcher.add_copy_requirements(
                    Legion::RegionRequirement{
                        rt->get_logical_partition(ctx, rhs_vectors[i].first, dimensions[i].second),
                        0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, rhs_vectors[i].first
                    },
                    Legion::RegionRequirement{
                        rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                        0, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, workspace[i]
                    }
                );
                launcher.add_src_field(0, rhs_vectors[i].second);
                launcher.add_dst_field(0, fid_dst);
                rt->issue_copy_operation(ctx, launcher);
            }
        }


        void zero_fill(Legion::FieldID fid_dst,
                       const std::vector<Legion::LogicalRegion> &workspace,
                       Legion::Context ctx,
                       Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == rhs_vectors.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                LegionSolvers::zero_fill<T>(
                    workspace[i], fid_dst,
                    dimensions[i].second,
                    ctx, rt
                );
            }
        }


        Legion::Future dot_product(Legion::FieldID fid_v,
                                   Legion::FieldID fid_w,
                                   const std::vector<Legion::LogicalRegion> &workspace,
                                   Legion::Context ctx,
                                   Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == rhs_vectors.size());

            Legion::Future result = Legion::Future::from_value<T>(rt, static_cast<T>(0));
            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    DotProductTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                    rt->get_index_partition_color_space_name(dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
                };
                launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]
                });
                launcher.add_field(0, fid_v);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]
                });
                launcher.add_field(1, fid_w);
                result = add<T>(
                    result,
                    rt->execute_index_space(ctx, launcher, LEGION_REDOP_SUM<T>),
                    ctx, rt
                );
            }
            return result;
        }


        void axpy(Legion::FieldID fid_y,
                  Legion::Future alpha, Legion::FieldID fid_x,
                  const std::vector<Legion::LogicalRegion> &workspace,
                  Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == rhs_vectors.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    AxpyTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                    rt->get_index_partition_color_space_name(dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
                };
                launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]});
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


        void axpy_sol(Legion::Future alpha, Legion::FieldID fid_x,
                      const std::vector<Legion::LogicalRegion> &workspace,
                      Legion::Context ctx, Legion::Runtime *rt) const {

            assert(workspace.size() == dimensions.size());
            assert(workspace.size() == solution_vectors.size());
            assert(workspace.size() == rhs_vectors.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    AxpyTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                    rt->get_index_partition_color_space_name(dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
                };
                launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, solution_vectors[i].first, dimensions[i].second),
                    0, LEGION_READ_WRITE, LEGION_EXCLUSIVE, solution_vectors[i].first
                });
                launcher.add_field(0, solution_vectors[i].second);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]
                });
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
            assert(workspace.size() == rhs_vectors.size());

            for (std::size_t i = 0; i < workspace.size(); ++i) {
                Legion::IndexLauncher launcher{
                    XpayTask<T, 0>::task_id(dimensions[i].first.get_dim()),
                    rt->get_index_partition_color_space_name(dimensions[i].second),
                    Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
                };
                launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_WRITE, LEGION_EXCLUSIVE, workspace[i]
                });
                launcher.add_field(0, fid_y);
                launcher.add_region_requirement(Legion::RegionRequirement{
                    rt->get_logical_partition(ctx, workspace[i], dimensions[i].second),
                    0, LEGION_READ_ONLY, LEGION_EXCLUSIVE, workspace[i]
                });
                launcher.add_field(1, fid_x);
                launcher.add_future(alpha);
                rt->execute_index_space(ctx, launcher);
            }
        }


    }; // class Planner


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_PLANNER_HPP
