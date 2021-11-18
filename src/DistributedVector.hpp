#ifndef LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
#define LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP

#include <cassert>
#include <memory>
#include <string>

#include <legion.h>

#include "LegionUtilities.hpp"
#include "LibraryOptions.hpp"
#include "LinearAlgebraTasks.hpp"
#include "Scalar.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class DistributedVector {

    public:

        virtual ~DistributedVector() = 0;

        virtual Legion::IndexSpace get_index_space() const = 0;

        virtual Legion::IndexSpace get_color_space() const = 0;

        virtual Legion::IndexPartition get_index_partition() const = 0;

        virtual std::unique_ptr<DistributedVector> similar(
            const std::string &new_name
        ) const = 0;

        virtual void zero_fill() = 0;

        virtual void constant_fill(ENTRY_T) = 0;

        virtual void operator=(ENTRY_T) = 0;

        virtual void constant_fill(const Scalar<ENTRY_T> &) = 0;

        virtual void operator=(const Scalar<ENTRY_T> &) = 0;

        virtual void random_fill(
            ENTRY_T low = static_cast<ENTRY_T>(0),
            ENTRY_T high = static_cast<ENTRY_T>(1)
        ) = 0;

        virtual void print() = 0;

        virtual void operator=(const DistributedVector &) = 0;

        virtual void axpy(const Scalar<ENTRY_T> &,
                          const DistributedVector &) = 0;

        virtual void axpy(ENTRY_T, const DistributedVector &) = 0;

        virtual void xpay(const Scalar<ENTRY_T> &,
                          const DistributedVector &) = 0;

        virtual void xpay(ENTRY_T, const DistributedVector &) = 0;

        virtual Scalar<ENTRY_T> dot(const DistributedVector &) const = 0;

    }; // class DistributedVector


    template <typename ENTRY_T>
    DistributedVector<ENTRY_T>::~DistributedVector() {}


    template <typename ENTRY_T,
              int DIM = 1, int COLOR_DIM = 1,
              typename COORD_T = Legion::coord_t,
              typename COLOR_COORD_T = Legion::coord_t>
    class DistributedVectorT: public DistributedVector<ENTRY_T> {

        const Legion::Context ctx;
        Legion::Runtime *const rt;

    public:

        const std::string name;
        const Legion::IndexSpaceT<DIM, COORD_T> index_space;
        const Legion::FieldID fid;
        const Legion::LogicalRegionT<DIM, COORD_T> logical_region;
        const Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space;
        const Legion::IndexPartitionT<DIM, COORD_T> index_partition;
        const Legion::LogicalPartitionT<DIM, COORD_T> logical_partition;

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
            fid(LEGION_SOLVERS_DEFAULT_VECTOR_FID),
            logical_region(create_region(
                index_space, {{sizeof(ENTRY_T), fid}}, ctx, rt
            )),
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
            fid(LEGION_SOLVERS_DEFAULT_VECTOR_FID),
            logical_region(create_region(
                index_space, {{sizeof(ENTRY_T), fid}}, ctx, rt
            )),
            color_space(rt->get_index_partition_color_space_name(
                index_partition
            )),
            index_partition(index_partition),
            logical_partition(
                rt->get_logical_partition(logical_region, index_partition)
            ) {}

        virtual Legion::IndexSpace get_index_space() const override {
            return index_space;
        }

        virtual Legion::IndexSpace get_color_space() const override {
            return color_space;
        }

        virtual Legion::IndexPartition get_index_partition() const override {
            return index_partition;
        }

        virtual std::unique_ptr<DistributedVector<ENTRY_T>> similar(
            const std::string &new_name
        ) const override {
            return std::make_unique<DistributedVectorT>(
                new_name, index_partition, ctx, rt
            );
        }

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

        virtual void constant_fill(const Scalar<ENTRY_T> &value) override {
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition,
                logical_region, value.get_future()
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        virtual void operator=(const Scalar<ENTRY_T> &value) override {
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

        virtual void axpy(const Scalar<ENTRY_T> &alpha,
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
            launcher.add_future(alpha.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        virtual void axpy(ENTRY_T alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            axpy(Scalar<ENTRY_T>{alpha, ctx, rt}, x);
        }

        virtual void xpay(const Scalar<ENTRY_T> &alpha,
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
            launcher.add_future(alpha.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        virtual void xpay(ENTRY_T alpha,
                          const DistributedVector<ENTRY_T> &x) override {
            xpay(Scalar<ENTRY_T>{alpha, ctx, rt}, x);
        }

        virtual Scalar<ENTRY_T> dot(
            const DistributedVector<ENTRY_T> &w
        ) const override {
            const DistributedVectorT &w_ref =
                dynamic_cast<const DistributedVectorT &>(w);
            assert(index_space == w_ref.index_space);
            assert(color_space == w_ref.color_space);
            assert(index_partition == w_ref.index_partition);
            Legion::IndexLauncher launcher{
                DotTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{nullptr, 0}, Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                w_ref.logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, w_ref.logical_region
            });
            launcher.add_field(1, w_ref.fid);
            return Scalar<ENTRY_T>{
                rt->execute_index_space(
                    ctx, launcher, LEGION_REDOP_SUM_FLOAT64 // TODO
                ),
                ctx, rt
            };
        }

    }; // class DistributedVectorT


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
