#ifndef LEGION_SOLVERS_DISTRIBUTED_COO_MATRIX_HPP
#define LEGION_SOLVERS_DISTRIBUTED_COO_MATRIX_HPP

#include <string>

#include <legion.h>

#include "COOMatrixTasks.hpp"
#include "DistributedVector.hpp"
#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "MaterializedLinearOperator.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T,
              int KERNEL_DIM = 1, int DOMAIN_DIM = 1,
              int RANGE_DIM = 1, int COLOR_DIM = 1,
              typename KERNEL_COORD_T = Legion::coord_t,
              typename DOMAIN_COORD_T = Legion::coord_t,
              typename RANGE_COORD_T = Legion::coord_t,
              typename COLOR_COORD_T = Legion::coord_t>
    class DistributedCOOMatrixT : public MaterializedLinearOperatorT<
        ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM,
        KERNEL_COORD_T, DOMAIN_COORD_T, RANGE_COORD_T
    > {

    public:

        Legion::Context ctx;
        Legion::Runtime *rt;
        std::string name;
        Legion::IndexSpaceT<KERNEL_DIM, KERNEL_COORD_T> kernel_space;
        Legion::IndexSpaceT<DOMAIN_DIM, DOMAIN_COORD_T> domain_space;
        Legion::IndexSpaceT<RANGE_DIM, RANGE_COORD_T> range_space;
        Legion::FieldID fid_i;     // Legion::Rect<RANGE_DIM>
        Legion::FieldID fid_j;     // Legion::Rect<DOMAIN_DIM>
        Legion::FieldID fid_entry; // ENTRY_T
        Legion::LogicalRegionT<KERNEL_DIM, KERNEL_COORD_T> kernel_region;
        Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space;
        Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_index_partition;
        Legion::LogicalPartitionT<KERNEL_DIM, KERNEL_COORD_T> kernel_logical_partition;

        DistributedCOOMatrixT() = delete;
        DistributedCOOMatrixT(const DistributedCOOMatrixT &) = delete;
        DistributedCOOMatrixT(DistributedCOOMatrixT &&) = delete;
        DistributedCOOMatrixT &operator=(const DistributedCOOMatrixT &) = delete;
        DistributedCOOMatrixT &operator=(DistributedCOOMatrixT &&) = delete;

        explicit DistributedCOOMatrixT(
            const std::string &name,
            Legion::IndexSpaceT<KERNEL_DIM, KERNEL_COORD_T> kernel_space,
            Legion::IndexSpaceT<DOMAIN_DIM, DOMAIN_COORD_T> domain_space,
            Legion::IndexSpaceT<RANGE_DIM, RANGE_COORD_T> range_space,
            Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx),
            rt(rt),
            name(name),
            kernel_space(kernel_space),
            domain_space(domain_space),
            range_space(range_space),
            fid_i(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_I),
            fid_j(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_J),
            fid_entry(LEGION_SOLVERS_DEFAULT_COO_MATRIX_FID_ENTRY),
            kernel_region(create_region(
                kernel_space,
                {
                    {sizeof(Legion::Point<RANGE_DIM, RANGE_COORD_T>), fid_i},
                    {sizeof(Legion::Point<DOMAIN_DIM, DOMAIN_COORD_T>), fid_j},
                    {sizeof(ENTRY_T), fid_entry},
                },
                ctx, rt
            )),
            color_space(color_space),
            kernel_index_partition(
                rt->create_equal_partition(ctx, kernel_space, color_space)
            ),
            kernel_logical_partition(
                rt->get_logical_partition(kernel_region, kernel_index_partition)
            ) {}

        virtual Legion::LogicalRegionT<KERNEL_DIM, KERNEL_COORD_T>
        get_kernel_region() const override { return kernel_region; }

        virtual Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>
        get_kernel_partition() const override { return kernel_index_partition; }

        Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>
        kernel_partition_from_domain_partition(
            Legion::IndexPartitionT<DOMAIN_DIM, DOMAIN_COORD_T> domain_partition
        ) const {
            const Legion::IndexSpace color_space =
                rt->get_index_partition_color_space_name(domain_partition);
            return Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>{
                rt->create_partition_by_preimage(
                    ctx, domain_partition, kernel_region, kernel_region,
                    fid_j, color_space, LEGION_DISJOINT_COMPLETE_KIND
                )
            };
        }

        virtual Legion::IndexPartition kernel_partition_from_domain_partition(
            Legion::IndexPartition domain_partition
        ) const override {
            return kernel_partition_from_domain_partition(
                Legion::IndexPartitionT<DOMAIN_DIM, DOMAIN_COORD_T>{domain_partition}
            );
        }

        Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>
        kernel_partition_from_range_partition(
            Legion::IndexPartitionT<RANGE_DIM, RANGE_COORD_T> range_partition
        ) const {
            const Legion::IndexSpace color_space =
                rt->get_index_partition_color_space_name(range_partition);
            return Legion::IndexPartitionT<KERNEL_DIM, KERNEL_COORD_T>{
                rt->create_partition_by_preimage(
                    ctx, range_partition, kernel_region, kernel_region,
                    fid_i, color_space, LEGION_DISJOINT_COMPLETE_KIND
                )
            };
        }

        virtual Legion::IndexPartition kernel_partition_from_range_partition(
            Legion::IndexPartition range_partition
        ) const override {
            return kernel_partition_from_range_partition(
                Legion::IndexPartitionT<RANGE_DIM, RANGE_COORD_T>{range_partition}
            );
        }

        void matvec(
            DistributedVectorT<
                ENTRY_T, RANGE_DIM, 1, RANGE_COORD_T, Legion::coord_t
            > &output_vector,
            const DistributedVectorT<
                ENTRY_T, RANGE_DIM, 1, RANGE_COORD_T, Legion::coord_t
            > &input_vector,
            Legion::IndexSpaceT<3> tile_index_space
        ) const {
            const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            Legion::IndexLauncher launcher{
                COOMatvecTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::task_id,
                tile_index_space,
                Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
                Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

            launcher.add_region_requirement(Legion::RegionRequirement{
                output_vector.logical_partition,
                PFID_KDR_TO_R, LEGION_REDOP_SUM<ENTRY_T>, LEGION_SIMULTANEOUS,
                output_vector.logical_region
            });
            launcher.add_field(0, output_vector.fid);

            launcher.add_region_requirement(Legion::RegionRequirement{
                kernel_logical_partition,
                PFID_KDR_TO_K, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                kernel_region
            });
            launcher.add_field(1, fid_i);
            launcher.add_field(1, fid_j);
            launcher.add_field(1, fid_entry);

            launcher.add_region_requirement(Legion::RegionRequirement{
                input_vector.logical_partition,
                PFID_KDR_TO_D, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                input_vector.logical_region
            });
            launcher.add_field(2, input_vector.fid);

            rt->execute_index_space(ctx, launcher);
        }

        virtual void matvec(
            DistributedVector<ENTRY_T> &output_vector,
            const DistributedVector<ENTRY_T> &input_vector,
            Legion::IndexSpaceT<3> tile_index_space
        ) const override {
            matvec(
                dynamic_cast<DistributedVectorT<
                    ENTRY_T, RANGE_DIM, 1, RANGE_COORD_T, Legion::coord_t
                > &>(output_vector),
                dynamic_cast<const DistributedVectorT<
                    ENTRY_T, RANGE_DIM, 1, RANGE_COORD_T, Legion::coord_t
                > &>(input_vector),
                tile_index_space
            );
        }

        virtual void rmatvec(
            DistributedVector<ENTRY_T> &output_vector,
            const DistributedVector<ENTRY_T> &input_vector,
            Legion::IndexSpaceT<3> tile_index_space
        ) const override {
            // TODO
        }

        virtual void print() const override {
            // TODO
        }

    }; // class DistributedCOOMatrixT


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DISTRIBUTED_COO_MATRIX_HPP
