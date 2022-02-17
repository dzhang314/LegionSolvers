#ifndef LEGION_SOLVERS_DENSE_DISTRIBUTED_VECTOR_HPP
#define LEGION_SOLVERS_DENSE_DISTRIBUTED_VECTOR_HPP

#include <string>

#include <legion.h>

#include "DenseVector.hpp"
#include "LinearAlgebraTasks.hpp"
#include "Scalar.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class DenseDistributedVector {

        Legion::Context          ctx;
        Legion::Runtime *        rt;
        std::string              name;
        Legion::IndexSpace       index_space;
        Legion::FieldID          fid;
        Legion::FieldSpace       field_space;
        Legion::LogicalRegion    logical_region;
        Legion::IndexSpace       color_space;
        Legion::IndexPartition   index_partition;
        Legion::LogicalPartition logical_partition;

    public:

        DenseDistributedVector() = delete;
        DenseDistributedVector(DenseDistributedVector &&) = delete;
        DenseDistributedVector &operator=(DenseDistributedVector &&) = delete;

        explicit DenseDistributedVector(
            DenseVector<ENTRY_T> &v,
            Legion::coord_t n
        ) : ctx(v.get_ctx()), rt(v.get_rt()),
            name(v.get_name()),
            index_space(v.get_index_space()),
            fid(v.get_fid()),
            field_space(v.get_field_space()),
            logical_region(v.get_logical_region()),
            color_space(rt->create_index_space(ctx,
                Legion::Rect<1, Legion::coord_t>{0, n - 1}
            )),
            index_partition(rt->create_equal_partition(ctx,
                index_space, color_space
            )),
            logical_partition(rt->get_logical_partition(
                logical_region, index_partition
            )) {
            rt->create_shared_ownership(ctx, index_space   );
            rt->create_shared_ownership(ctx, field_space   );
            rt->create_shared_ownership(ctx, logical_region);
            const std::string color_space_name       = name + "_color_space"      ;
            const std::string index_partition_name   = name + "_index_partition"  ;
            const std::string logical_partition_name = name + "_logical_partition";
            rt->attach_name(color_space      , color_space_name      .c_str());
            rt->attach_name(index_partition  , index_partition_name  .c_str());
            rt->attach_name(logical_partition, logical_partition_name.c_str());
        }

        explicit DenseDistributedVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::IndexPartition index_partition
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(rt->get_parent_index_space(index_partition)),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )),
            color_space(rt->get_index_partition_color_space_name(
                index_partition
            )),
            index_partition(index_partition),
            logical_partition(rt->get_logical_partition(
                logical_region, index_partition
            )) {
            assert(rt->is_index_partition_disjoint(index_partition));
            assert(rt->is_index_partition_complete(index_partition));
            rt->create_shared_ownership(ctx, index_space    );
            rt->create_shared_ownership(ctx, color_space    );
            rt->create_shared_ownership(ctx, index_partition);
            const std::string field_space_name       = name + "_field_space"      ;
            const std::string logical_region_name    = name + "_logical_region"   ;
            const std::string logical_partition_name = name + "_logical_partition";
            rt->attach_name(field_space      , field_space_name      .c_str());
            rt->attach_name(logical_region   , logical_region_name   .c_str());
            rt->attach_name(logical_partition, logical_partition_name.c_str());
        }

        explicit DenseDistributedVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::LogicalPartition logical_partition,
            Legion::FieldID fid
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(rt->get_parent_index_space(
                logical_partition.get_index_partition()
            )),
            fid(fid),
            field_space(logical_partition.get_field_space()),
            logical_region(rt->get_parent_logical_region(logical_partition)),
            color_space(rt->get_index_partition_color_space_name(
                logical_partition.get_index_partition()
            )),
            index_partition(logical_partition.get_index_partition()),
            logical_partition(logical_partition) {
            assert(rt->is_index_partition_disjoint(index_partition));
            assert(rt->is_index_partition_complete(index_partition));
            rt->create_shared_ownership(ctx, index_space    );
            rt->create_shared_ownership(ctx, field_space    );
            rt->create_shared_ownership(ctx, logical_region );
            rt->create_shared_ownership(ctx, color_space    );
            rt->create_shared_ownership(ctx, index_partition);
        }

        DenseDistributedVector(
            const DenseDistributedVector &v
        ) : ctx(v.ctx), rt(v.rt),
            name(v.name),
            index_space(v.index_space),
            fid(v.fid),
            field_space(v.field_space),
            logical_region(v.logical_region),
            color_space(v.color_space),
            index_partition(v.index_partition),
            logical_partition(v.logical_partition) {
            rt->create_shared_ownership(ctx, index_space    );
            rt->create_shared_ownership(ctx, field_space    );
            rt->create_shared_ownership(ctx, logical_region );
            rt->create_shared_ownership(ctx, color_space    );
            rt->create_shared_ownership(ctx, index_partition);
        }

        ~DenseDistributedVector() {
            rt->destroy_index_partition(ctx, index_partition);
            rt->destroy_index_space    (ctx, color_space    );
            rt->destroy_logical_region (ctx, logical_region );
            rt->destroy_field_space    (ctx, field_space    );
            rt->destroy_index_space    (ctx, index_space    );
        }

        Legion::Context          get_ctx              () const { return ctx              ; }
        Legion::Runtime *        get_rt               () const { return rt               ; }
        const std::string &      get_name             () const { return name             ; }
        Legion::IndexSpace       get_index_space      () const { return index_space      ; }
        Legion::FieldID          get_fid              () const { return fid              ; }
        Legion::FieldSpace       get_field_space      () const { return field_space      ; }
        Legion::LogicalRegion    get_logical_region   () const { return logical_region   ; }
        Legion::IndexSpace       get_color_space      () const { return color_space      ; }
        Legion::IndexPartition   get_index_partition  () const { return index_partition  ; }
        Legion::LogicalPartition get_logical_partition() const { return logical_partition; }

        void constant_fill(ENTRY_T value) {
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region,
                Legion::UntypedBuffer{&value, sizeof(ENTRY_T)}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        void constant_fill(const Scalar<ENTRY_T> &value) {
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region,
                value.get_future()
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        void zero_fill() {
            constant_fill(static_cast<ENTRY_T>(0));
        }

        ENTRY_T operator=(ENTRY_T value) {
            constant_fill(value);
            return value;
        }

        const Scalar<ENTRY_T> &operator=(const Scalar<ENTRY_T> &value) {
            constant_fill(value);
            return value;
        }

        const DenseDistributedVector &operator=(
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexCopyLauncher launcher{color_space};
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_copy_requirements(
                Legion::RegionRequirement{
                    x.get_logical_partition(), 0,
                    LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
                },
                Legion::RegionRequirement{
                    logical_partition, 0,
                    LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, logical_region
                }
            );
            launcher.add_src_field(0, x.get_fid());
            launcher.add_dst_field(0, fid);
            rt->issue_copy_operation(ctx, launcher);
            return x;
        }

        void random_fill(
            ENTRY_T low = static_cast<ENTRY_T>(0),
            ENTRY_T high = static_cast<ENTRY_T>(1)
        ) {
            const ENTRY_T args[2] = {low, high};
            Legion::IndexTaskLauncher launcher{
                RandomFillTask<ENTRY_T, 0, void>::task_id(index_space),
                color_space,
                Legion::UntypedBuffer{&args, 2 * sizeof(ENTRY_T)},
                Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_index_space(ctx, launcher);
        }

        void axpy(
            const Scalar<ENTRY_T> &alpha,
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                AxpyTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(alpha.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        void axpy(ENTRY_T alpha, const DenseDistributedVector &x) {
            axpy(Scalar<ENTRY_T>{ctx, rt, alpha}, x);
        }

        void axpy(
            const Scalar<ENTRY_T> &numer,
            const Scalar<ENTRY_T> &denom,
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                AxpyTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(numer.get_future());
            launcher.add_future(denom.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        void axpy(
            const Scalar<ENTRY_T> &numer1,
            const Scalar<ENTRY_T> &numer2,
            const Scalar<ENTRY_T> &denom,
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                AxpyTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(numer1.get_future());
            launcher.add_future(numer2.get_future());
            launcher.add_future(denom.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        void xpay(
            const Scalar<ENTRY_T> &alpha,
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                XpayTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(alpha.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        void xpay(
            const Scalar<ENTRY_T> &numer,
            const Scalar<ENTRY_T> &denom,
            const DenseDistributedVector &x
        ) {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                XpayTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(numer.get_future());
            launcher.add_future(denom.get_future());
            rt->execute_index_space(ctx, launcher);
        }

        void xpay(ENTRY_T alpha, const DenseDistributedVector &x) {
            xpay(Scalar<ENTRY_T>{ctx, rt, alpha}, x);
        }

        Scalar<ENTRY_T> dot(const DenseDistributedVector &x) const {
            assert(index_space     == x.get_index_space    ());
            assert(color_space     == x.get_color_space    ());
            assert(index_partition == x.get_index_partition());
            Legion::IndexTaskLauncher launcher{
                DotTask<ENTRY_T, 0, void>::task_id(index_space), color_space,
                Legion::UntypedBuffer{}, Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_partition(), 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            return Scalar<ENTRY_T>{ctx, rt,
                rt->execute_index_space(ctx,
                    launcher, LEGION_REDOP_SUM<ENTRY_T>
                ),
            };
        }

        void print() const {
            Legion::IndexTaskLauncher launcher{
                PrintVectorTask<ENTRY_T, 0, void>::task_id(index_space),
                color_space,
                Legion::UntypedBuffer{name.c_str(), name.length() + 1},
                Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_index_space(ctx, launcher);
        }

    }; // class DenseDistributedVector


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DENSE_DISTRIBUTED_VECTOR_HPP
