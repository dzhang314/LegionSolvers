#ifndef LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
#define LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED

#include <legion.h> // for Legion::*

#include "LinearAlgebraTasks.hpp" // for ScalTask, AxpyTask, XpayTask, DotTask
#include "Scalar.hpp"             // for Scalar

namespace LegionSolvers {


template <typename ENTRY_T>
class PartitionedVector {

    const Legion::Context ctx;
    Legion::Runtime *const rt;
    const std::string name;
    const Legion::IndexSpace index_space;
    const Legion::FieldID fid;
    const Legion::FieldSpace field_space;
    const Legion::LogicalRegion logical_region;
    const Legion::IndexSpace color_space;
    const Legion::IndexPartition index_partition;
    const Legion::LogicalPartition logical_partition;

  public:

    PartitionedVector() = delete;
    PartitionedVector(PartitionedVector &&) = delete;
    PartitionedVector &operator=(PartitionedVector &&) = delete;

    explicit PartitionedVector(
        Legion::Context ctx,
        Legion::Runtime *rt,
        const std::string &name,
        Legion::IndexPartition index_partition
    )
        : ctx(ctx), rt(rt), name(name),
          index_space(rt->get_parent_index_space(index_partition)), fid(0),
          field_space(LegionSolvers::create_field_space(
              ctx, rt, {sizeof(ENTRY_T)}, {fid}
          )),
          logical_region(
              rt->create_logical_region(ctx, index_space, field_space)
          ),
          color_space(rt->get_index_partition_color_space_name(index_partition)
          ),
          index_partition(index_partition),
          logical_partition(
              rt->get_logical_partition(logical_region, index_partition)
          ) {
        assert(rt->is_index_partition_disjoint(index_partition));
        assert(rt->is_index_partition_complete(index_partition));
        rt->create_shared_ownership(ctx, index_space);
        rt->create_shared_ownership(ctx, color_space);
        rt->create_shared_ownership(ctx, index_partition);
        const std::string field_space_name = name + "_field_space";
        const std::string logical_region_name = name + "_logical_region";
        const std::string logical_partition_name = name + "_logical_partition";
        rt->attach_name(field_space, field_space_name.c_str());
        rt->attach_name(logical_region, logical_region_name.c_str());
        rt->attach_name(logical_partition, logical_partition_name.c_str());
    }

    explicit PartitionedVector(
        Legion::Context ctx,
        Legion::Runtime *rt,
        const std::string &name,
        Legion::LogicalPartition logical_partition,
        Legion::FieldID fid
    )
        : ctx(ctx), rt(rt), name(name),
          index_space(rt->get_parent_index_space(
              logical_partition.get_index_partition()
          )),
          fid(fid), field_space(logical_partition.get_field_space()),
          logical_region(rt->get_parent_logical_region(logical_partition)),
          color_space(rt->get_index_partition_color_space_name(
              logical_partition.get_index_partition()
          )),
          index_partition(logical_partition.get_index_partition()),
          logical_partition(logical_partition) {
        assert(rt->is_index_partition_disjoint(index_partition));
        assert(rt->is_index_partition_complete(index_partition));
        rt->create_shared_ownership(ctx, index_space);
        rt->create_shared_ownership(ctx, field_space);
        rt->create_shared_ownership(ctx, logical_region);
        rt->create_shared_ownership(ctx, color_space);
        rt->create_shared_ownership(ctx, index_partition);
    }

    PartitionedVector(const PartitionedVector &v)
        : ctx(v.ctx), rt(v.rt), name(v.name), index_space(v.index_space),
          fid(v.fid), field_space(v.field_space),
          logical_region(v.logical_region), color_space(v.color_space),
          index_partition(v.index_partition),
          logical_partition(v.logical_partition) {
        rt->create_shared_ownership(ctx, index_space);
        rt->create_shared_ownership(ctx, field_space);
        rt->create_shared_ownership(ctx, logical_region);
        rt->create_shared_ownership(ctx, color_space);
        rt->create_shared_ownership(ctx, index_partition);
    }

    ~PartitionedVector() {
        rt->destroy_index_partition(ctx, index_partition);
        rt->destroy_index_space(ctx, color_space);
        rt->destroy_logical_region(ctx, logical_region);
        rt->destroy_field_space(ctx, field_space);
        rt->destroy_index_space(ctx, index_space);
    }

    Legion::Context get_ctx() const { return ctx; }

    Legion::Runtime *get_rt() const { return rt; }

    const std::string &get_name() const { return name; }

    Legion::IndexSpace get_index_space() const { return index_space; }

    int get_dim() const { return index_space.get_dim(); }

    Legion::FieldID get_fid() const { return fid; }

    Legion::FieldSpace get_field_space() const { return field_space; }

    Legion::LogicalRegion get_logical_region() const { return logical_region; }

    Legion::IndexSpace get_color_space() const { return color_space; }

    Legion::IndexPartition get_index_partition() const {
        return index_partition;
    }

    Legion::LogicalPartition get_logical_partition() const {
        return logical_partition;
    }

    void constant_fill(ENTRY_T value) {
        Legion::IndexFillLauncher launcher(
            color_space,
            logical_partition,
            logical_region,
            Legion::UntypedBuffer(&value, sizeof(ENTRY_T))
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_field(fid);
        rt->fill_fields(ctx, launcher);
    }

    void constant_fill(const Scalar<ENTRY_T> &value) {
        Legion::IndexFillLauncher launcher(
            color_space, logical_partition, logical_region, value.get_future()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_field(fid);
        rt->fill_fields(ctx, launcher);
    }

    void zero_fill() { constant_fill(static_cast<ENTRY_T>(0)); }

    ENTRY_T operator=(ENTRY_T value) {
        constant_fill(value);
        return value;
    }

    const Scalar<ENTRY_T> &operator=(const Scalar<ENTRY_T> &value) {
        constant_fill(value);
        return value;
    }

    const PartitionedVector &operator=(const PartitionedVector &x) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexCopyLauncher launcher(color_space);
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_copy_requirements(
            Legion::RegionRequirement(
                x.get_logical_partition(),
                0,
                LEGION_READ_ONLY,
                LEGION_EXCLUSIVE,
                x.get_logical_region()
            ),
            Legion::RegionRequirement(
                logical_partition,
                0,
                LEGION_WRITE_DISCARD,
                LEGION_EXCLUSIVE,
                logical_region
            )
        );
        launcher.add_src_field(0, x.get_fid());
        launcher.add_dst_field(0, fid);
        rt->issue_copy_operation(ctx, launcher);
        return x;
    }

    // TODO: operator*=(Scalar) should call ScalTask

    void axpy(const Scalar<ENTRY_T> &alpha, const PartitionedVector &x) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            AxpyTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        launcher.add_future(alpha.get_future());
        rt->execute_index_space(ctx, launcher);
    }

    void axpy(ENTRY_T alpha, const PartitionedVector &x) {
        axpy(Scalar<ENTRY_T>(ctx, rt, alpha), x);
    }

    void axpy(
        const Scalar<ENTRY_T> &numer,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    ) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            AxpyTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        launcher.add_future(numer.get_future());
        launcher.add_future(denom.get_future());
        rt->execute_index_space(ctx, launcher);
    }

    void axpy(
        const Scalar<ENTRY_T> &numer1,
        const Scalar<ENTRY_T> &numer2,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    ) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            AxpyTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        launcher.add_future(numer1.get_future());
        launcher.add_future(numer2.get_future());
        launcher.add_future(denom.get_future());
        rt->execute_index_space(ctx, launcher);
    }

    void xpay(const Scalar<ENTRY_T> &alpha, const PartitionedVector &x) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            XpayTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        launcher.add_future(alpha.get_future());
        rt->execute_index_space(ctx, launcher);
    }

    void xpay(
        const Scalar<ENTRY_T> &numer,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    ) {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            XpayTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_WRITE,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        launcher.add_future(numer.get_future());
        launcher.add_future(denom.get_future());
        rt->execute_index_space(ctx, launcher);
    }

    void xpay(ENTRY_T alpha, const PartitionedVector &x) {
        xpay(Scalar<ENTRY_T>(ctx, rt, alpha), x);
    }

    Scalar<ENTRY_T> dot(const PartitionedVector &x) const {
        assert(index_space == x.get_index_space());
        assert(color_space == x.get_color_space());
        assert(index_partition == x.get_index_partition());
        Legion::IndexTaskLauncher launcher(
            DotTask<ENTRY_T, 0, void>::task_id(index_space),
            color_space,
            Legion::UntypedBuffer(),
            Legion::ArgumentMap()
        );
        launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_region_requirement(Legion::RegionRequirement(
            logical_partition,
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            logical_region
        ));
        launcher.add_field(0, fid);
        launcher.add_region_requirement(Legion::RegionRequirement(
            x.get_logical_partition(),
            0,
            LEGION_READ_ONLY,
            LEGION_EXCLUSIVE,
            x.get_logical_region()
        ));
        launcher.add_field(1, x.get_fid());
        return Scalar<ENTRY_T>{
            ctx,
            rt,
            rt->execute_index_space(ctx, launcher, LEGION_REDOP_SUM<ENTRY_T>),
        };
    }

}; // class PartitionedVector


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
