#ifndef LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
#define LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP

#include <legion.h>

#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "LinearAlgebraTasks.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class DistributedVector {

    public:

        virtual void zero_fill() = 0;

        virtual void constant_fill(ENTRY_T) = 0;

        virtual void operator=(ENTRY_T) = 0;

        virtual void constant_fill(Legion::Future) = 0;

        virtual void operator=(Legion::Future) = 0;

        virtual void random_fill(
            ENTRY_T low = static_cast<ENTRY_T>(0),
            ENTRY_T high = static_cast<ENTRY_T>(1)
        ) = 0;

        virtual void print() = 0;

        virtual void operator=(const DistributedVector &) = 0;

        virtual void axpy(Legion::Future, const DistributedVector &) = 0;

        virtual void axpy(ENTRY_T, const DistributedVector &) = 0;

        virtual void xpay(Legion::Future, const DistributedVector &) = 0;

        virtual void xpay(ENTRY_T, const DistributedVector &) = 0;

    }; // class DistributedVector


    template <typename ENTRY_T,
              int DIM = 1, typename COORD_T = Legion::coord_t,
              int COLOR_DIM = 1, typename COLOR_COORD_T = Legion::coord_t>
    class DistributedVectorT: public DistributedVector<ENTRY_T> {

    public:

        Legion::Context ctx;
        Legion::Runtime *rt;
        std::string name;
        Legion::IndexSpaceT<DIM, COORD_T> index_space;
        Legion::LogicalRegionT<DIM, COORD_T> logical_region;
        Legion::FieldID fid;
        Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space;
        Legion::IndexPartitionT<DIM> index_partition;
        Legion::LogicalPartitionT<DIM, COORD_T> logical_partition;

        DistributedVectorT() = delete;
        DistributedVectorT(const DistributedVectorT &) = delete;
        DistributedVectorT(DistributedVectorT &&) = delete;
        DistributedVectorT &operator=(DistributedVectorT &&) = delete;

        explicit DistributedVectorT(
            const std::string &name,
            Legion::IndexSpaceT<DIM, COORD_T> index_space,
            Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx),
            rt(rt),
            name(name),
            index_space(index_space),
            logical_region(LegionSolvers::create_region(
                index_space,
                {{sizeof(ENTRY_T), LEGION_SOLVERS_DEFAULT_VECTOR_FID}},
                ctx, rt
            )),
            fid(LEGION_SOLVERS_DEFAULT_VECTOR_FID),
            color_space(color_space),
            index_partition(
                rt->create_equal_partition(ctx, index_space, color_space)
            ),
            logical_partition(
                rt->get_logical_partition(logical_region, index_partition)
            ) {}

        explicit DistributedVectorT(
            const std::string &name,
            Legion::IndexPartitionT<DIM, COORD_T> index_partition,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx),
            rt(rt),
            name(name),
            index_space(rt->get_parent_index_space(index_partition)),
            logical_region(LegionSolvers::create_region(
                index_space,
                {{sizeof(ENTRY_T), LEGION_SOLVERS_DEFAULT_VECTOR_FID}},
                ctx, rt
            )),
            fid(LEGION_SOLVERS_DEFAULT_VECTOR_FID),
            color_space(rt->get_index_partition_color_space_name(
                index_partition
            )),
            index_partition(index_partition),
            logical_partition(
                rt->get_logical_partition(logical_region, index_partition)
            ) {}

        virtual void zero_fill() override {
            static constexpr ENTRY_T zero = static_cast<ENTRY_T>(0);
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region,
                Legion::TaskArgument{&zero, sizeof(ENTRY_T)}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        virtual void constant_fill(ENTRY_T value) override {
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region,
                Legion::TaskArgument{&value, sizeof(ENTRY_T)}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        virtual void operator=(ENTRY_T value) override {
            constant_fill(value);
        }

        virtual void constant_fill(Legion::Future value) override {
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region, value
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        virtual void operator=(Legion::Future value) override {
            constant_fill(value);
        }

        virtual void random_fill(
            ENTRY_T low = static_cast<ENTRY_T>(0),
            ENTRY_T high = static_cast<ENTRY_T>(1)
        ) override {
            const ENTRY_T args[2] = {low, high};
            Legion::IndexLauncher launcher{
                RandomFillTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{&args, 2 * sizeof(ENTRY_T)},
                Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_index_space(ctx, launcher);
        }

        virtual void print() override {
            Legion::IndexTaskLauncher launcher{
                PrintVectorTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{name.c_str(), name.length() + 1},
                Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_index_space(ctx, launcher);
        }

        void operator=(const DistributedVectorT &x) {
            assert(index_space == x.index_space);
            assert(color_space == x.color_space);
            assert(index_partition == x.index_partition);
            Legion::IndexCopyLauncher launcher{color_space};
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_copy_requirements(
                Legion::RegionRequirement{
                    x.logical_partition, 0,
                    LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.logical_region
                },
                Legion::RegionRequirement{
                    logical_partition, 0,
                    LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, logical_region
                }
            );
            launcher.add_src_field(0, x.fid);
            launcher.add_dst_field(0, fid);
            rt->issue_copy_operation(ctx, launcher);
        }

        virtual void operator=(const DistributedVector<ENTRY_T> &x) override {
            this->operator=(dynamic_cast<const DistributedVectorT &>(x));
        }

        virtual void axpy(Legion::Future alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            const DistributedVectorT &x_ref =
                dynamic_cast<const DistributedVectorT &>(x);
            assert(index_space == x_ref.index_space);
            assert(color_space == x_ref.color_space);
            assert(index_partition == x_ref.index_partition);
            Legion::IndexLauncher launcher{
                AxpyTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x_ref.logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x_ref.logical_region
            });
            launcher.add_field(1, x_ref.fid);
            launcher.add_future(alpha);
            rt->execute_index_space(ctx, launcher);
        }

        virtual void axpy(ENTRY_T alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            axpy(Legion::Future::from_value<ENTRY_T>(rt, alpha), x);
        }

        virtual void xpay(Legion::Future alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            const DistributedVectorT &x_ref =
                dynamic_cast<const DistributedVectorT &>(x);
            assert(index_space == x_ref.index_space);
            assert(color_space == x_ref.color_space);
            assert(index_partition == x_ref.index_partition);
            Legion::IndexLauncher launcher{
                XpayTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x_ref.logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x_ref.logical_region
            });
            launcher.add_field(1, x_ref.fid);
            launcher.add_future(alpha);
            rt->execute_index_space(ctx, launcher);
        }

        virtual void xpay(ENTRY_T alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            xpay(Legion::Future::from_value<ENTRY_T>(rt, alpha), x);
        }

    }; // class DistributedVectorT


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
