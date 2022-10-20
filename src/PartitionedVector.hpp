#ifndef LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
#define LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED

#include <legion.h> // for Legion::*

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

}; // class PartitionedVector


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
