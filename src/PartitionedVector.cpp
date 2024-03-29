#include "PartitionedVector.hpp"

#include <set>    // for std::set
#include <vector> // for std::vector

#include "LinearAlgebraTasks.hpp" // for ScalTask, AxpyTask, XpayTask, DotTask

using LegionSolvers::PartitionedVector;
using LegionSolvers::Scalar;


// Every PartitionedVector contains five members that need to be explicitly
// freed using Legion runtime calls: an index space, a field space, a logical
// region, a color space, and an index partition. The PartitionedVector
// destructor explicitly calls rt->destroy_*() on each of these members.

// Thus, in every PartitionedVector constructor, each of these five members
// should either be created using an rt->create_*() call, or have its reference
// count incremented by explicitly calling rt->create_shared_ownership().

// Also, each member which is created using an rt->create_*() call should have
// a name attached using rt->attach_name().


template <typename ENTRY_T>
PartitionedVector<ENTRY_T>::PartitionedVector(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::string &name,
    Legion::IndexPartition index_partition
)
    : ctx(ctx)
    , rt(rt)
    , name(name)
    , index_space(rt->get_parent_index_space(index_partition))
    , fid(0)
    , field_space(
          LegionSolvers::create_field_space(ctx, rt, {sizeof(ENTRY_T)}, {fid})
      )
    , logical_region(rt->create_logical_region(ctx, index_space, field_space))
    , color_space(rt->get_index_partition_color_space_name(index_partition))
    , index_partition(index_partition)
    , logical_partition(
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


template <typename ENTRY_T>
PartitionedVector<ENTRY_T>::PartitionedVector(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::string &name,
    Legion::LogicalPartition logical_partition,
    Legion::FieldID fid
)
    : ctx(ctx)
    , rt(rt)
    , name(name)
    , index_space(
          rt->get_parent_index_space(logical_partition.get_index_partition())
      )
    , fid(fid)
    , field_space(logical_partition.get_field_space())
    , logical_region(rt->get_parent_logical_region(logical_partition))
    , color_space(rt->get_index_partition_color_space_name(
          logical_partition.get_index_partition()
      ))
    , index_partition(logical_partition.get_index_partition())
    , logical_partition(logical_partition) {

    assert(rt->is_index_partition_disjoint(index_partition));
    assert(rt->is_index_partition_complete(index_partition));

    rt->create_shared_ownership(ctx, index_space);
    rt->create_shared_ownership(ctx, field_space);
    rt->create_shared_ownership(ctx, logical_region);
    rt->create_shared_ownership(ctx, color_space);
    rt->create_shared_ownership(ctx, index_partition);
}


template <typename ENTRY_T>
PartitionedVector<ENTRY_T>::PartitionedVector(const PartitionedVector &v)
    : ctx(v.ctx)
    , rt(v.rt)
    , name(v.name)
    , index_space(v.index_space)
    , fid(v.fid)
    , field_space(v.field_space)
    , logical_region(v.logical_region)
    , color_space(v.color_space)
    , index_partition(v.index_partition)
    , logical_partition(v.logical_partition) {

    rt->create_shared_ownership(ctx, index_space);
    rt->create_shared_ownership(ctx, field_space);
    rt->create_shared_ownership(ctx, logical_region);
    rt->create_shared_ownership(ctx, color_space);
    rt->create_shared_ownership(ctx, index_partition);
}


template <typename ENTRY_T>
PartitionedVector<ENTRY_T>::~PartitionedVector() {
#ifndef LEGION_SOLVERS_DISABLE_CLEANUP
    rt->destroy_index_partition(ctx, index_partition);
    rt->destroy_index_space(ctx, color_space);
    rt->destroy_logical_region(ctx, logical_region);
    rt->destroy_field_space(ctx, field_space);
    rt->destroy_index_space(ctx, index_space);
#endif // LEGION_SOLVERS_DISABLE_CLEANUP
}


template <typename ENTRY_T>
Legion::RegionRequirement PartitionedVector<ENTRY_T>::get_requirement(
    Legion::PrivilegeMode privileges, Legion::CoherenceProperty coherence
) const {
    constexpr Legion::ProjectionID IDENTITY_PROJECTION = 0;
    const std::set<Legion::FieldID> privilege_fields = {fid};
    const std::vector<Legion::FieldID> instance_fields = {fid};
    return Legion::RegionRequirement(
        logical_partition,
        IDENTITY_PROJECTION,
        privilege_fields,
        instance_fields,
        privileges,
        coherence,
        logical_region
    );
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::constant_fill(ENTRY_T value) {
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


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::constant_fill(const Scalar<ENTRY_T> &value) {
    Legion::IndexFillLauncher launcher(
        color_space, logical_partition, logical_region, value.get_future()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_field(fid);
    rt->fill_fields(ctx, launcher);
}


template <typename ENTRY_T>
const PartitionedVector<ENTRY_T> &
PartitionedVector<ENTRY_T>::operator=(const PartitionedVector &x) {

    assert(index_space == x.get_index_space());
    assert(color_space == x.get_color_space());
    assert(index_partition == x.get_index_partition());

    Legion::IndexCopyLauncher launcher(color_space);
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_copy_requirements(
        x.get_requirement(LEGION_READ_ONLY),
        get_requirement(LEGION_WRITE_DISCARD)
    );
    rt->issue_copy_operation(ctx, launcher);
    return x;
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::scal(const Scalar<ENTRY_T> &alpha) {
    Legion::IndexTaskLauncher launcher(
        ScalTask<ENTRY_T, 0, void>::task_id(index_space),
        color_space,
        Legion::UntypedBuffer(),
        Legion::ArgumentMap()
    );
    launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
    launcher.add_future(alpha.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::axpy(
    const Scalar<ENTRY_T> &alpha, const PartitionedVector &x
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
    launcher.add_future(alpha.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::axpy(
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
    launcher.add_future(numer.get_future());
    launcher.add_future(denom.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::axpy(
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
    launcher.add_future(numer1.get_future());
    launcher.add_future(numer2.get_future());
    launcher.add_future(denom.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::xpay(
    const Scalar<ENTRY_T> &alpha, const PartitionedVector &x
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
    launcher.add_future(alpha.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
void PartitionedVector<ENTRY_T>::xpay(
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
    launcher.add_future(numer.get_future());
    launcher.add_future(denom.get_future());
    launcher.add_region_requirement(get_requirement(LEGION_READ_WRITE));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    rt->execute_index_space(ctx, launcher);
}


template <typename ENTRY_T>
Scalar<ENTRY_T> PartitionedVector<ENTRY_T>::dot(const PartitionedVector &x
) const {
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
    launcher.add_region_requirement(get_requirement(LEGION_READ_ONLY));
    launcher.add_region_requirement(x.get_requirement(LEGION_READ_ONLY));

    return Scalar<ENTRY_T>{
        ctx,
        rt,
        rt->execute_index_space(ctx, launcher, LEGION_REDOP_SUM<ENTRY_T>),
    };
}


// clang-format off
#if LEGION_SOLVERS_USE_F32
    template PartitionedVector<float>::PartitionedVector(Legion::Context, Legion::Runtime *, const std::string &, Legion::IndexPartition);
    template PartitionedVector<float>::PartitionedVector(Legion::Context, Legion::Runtime *, const std::string &, Legion::LogicalPartition, Legion::FieldID);
    template PartitionedVector<float>::PartitionedVector(const PartitionedVector<float> &);
    template PartitionedVector<float>::~PartitionedVector();
    template Legion::RegionRequirement PartitionedVector<float>::get_requirement(Legion::PrivilegeMode, Legion::CoherenceProperty) const;
    template void PartitionedVector<float>::constant_fill(float);
    template void PartitionedVector<float>::constant_fill(const Scalar<float> &);
    template const PartitionedVector<float> &PartitionedVector<float>::operator=(const PartitionedVector<float> &);
    template void PartitionedVector<float>::scal(const Scalar<float> &);
    template void PartitionedVector<float>::axpy(const Scalar<float> &, const PartitionedVector<float> &);
    template void PartitionedVector<float>::axpy(const Scalar<float> &, const Scalar<float> &, const PartitionedVector<float> &);
    template void PartitionedVector<float>::axpy(const Scalar<float> &, const Scalar<float> &, const Scalar<float> &, const PartitionedVector<float> &);
    template void PartitionedVector<float>::xpay(const Scalar<float> &, const PartitionedVector<float> &);
    template void PartitionedVector<float>::xpay(const Scalar<float> &, const Scalar<float> &, const PartitionedVector<float> &);
    template Scalar<float> PartitionedVector<float>::dot(const PartitionedVector<float> &) const;
#endif // LEGION_SOLVERS_USE_F32
#if LEGION_SOLVERS_USE_F64
    template PartitionedVector<double>::PartitionedVector(Legion::Context, Legion::Runtime *, const std::string &, Legion::IndexPartition);
    template PartitionedVector<double>::PartitionedVector(Legion::Context, Legion::Runtime *, const std::string &, Legion::LogicalPartition, Legion::FieldID);
    template PartitionedVector<double>::PartitionedVector(const PartitionedVector<double> &);
    template PartitionedVector<double>::~PartitionedVector();
    template Legion::RegionRequirement PartitionedVector<double>::get_requirement(Legion::PrivilegeMode, Legion::CoherenceProperty) const;
    template void PartitionedVector<double>::constant_fill(double);
    template void PartitionedVector<double>::constant_fill(const Scalar<double> &);
    template const PartitionedVector<double> &PartitionedVector<double>::operator=(const PartitionedVector<double> &);
    template void PartitionedVector<double>::scal(const Scalar<double> &);
    template void PartitionedVector<double>::axpy(const Scalar<double> &, const PartitionedVector<double> &);
    template void PartitionedVector<double>::axpy(const Scalar<double> &, const Scalar<double> &, const PartitionedVector<double> &);
    template void PartitionedVector<double>::axpy(const Scalar<double> &, const Scalar<double> &, const Scalar<double> &, const PartitionedVector<double> &);
    template void PartitionedVector<double>::xpay(const Scalar<double> &, const PartitionedVector<double> &);
    template void PartitionedVector<double>::xpay(const Scalar<double> &, const Scalar<double> &, const PartitionedVector<double> &);
    template Scalar<double> PartitionedVector<double>::dot(const PartitionedVector<double> &) const;
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
