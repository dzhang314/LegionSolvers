#ifndef LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
#define LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED

#include <string> // for std::string

#include <legion.h> // for Legion::*

#include "Scalar.hpp" // for Scalar

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

    explicit PartitionedVector(
        Legion::Context ctx,
        Legion::Runtime *rt,
        const std::string &name,
        Legion::IndexPartition index_partition
    );

    explicit PartitionedVector(
        Legion::Context ctx,
        Legion::Runtime *rt,
        const std::string &name,
        Legion::LogicalPartition logical_partition,
        Legion::FieldID fid
    );

    PartitionedVector(const PartitionedVector &v);

    PartitionedVector(PartitionedVector &&) = delete;

    ~PartitionedVector();

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

    Legion::RegionRequirement get_requirement(
        Legion::PrivilegeMode privileges,
        Legion::CoherenceProperty coherence = LEGION_EXCLUSIVE
    ) const;

    void constant_fill(ENTRY_T value);

    void constant_fill(const Scalar<ENTRY_T> &value);

    void zero_fill() { constant_fill(static_cast<ENTRY_T>(0)); }

    ENTRY_T operator=(ENTRY_T value) {
        constant_fill(value);
        return value;
    }

    const Scalar<ENTRY_T> &operator=(const Scalar<ENTRY_T> &value) {
        constant_fill(value);
        return value;
    }

    const PartitionedVector &operator=(const PartitionedVector &x);

    PartitionedVector &operator=(PartitionedVector &&) = delete;

    void scal(const Scalar<ENTRY_T> &alpha);

    void axpy(const Scalar<ENTRY_T> &alpha, const PartitionedVector &x);

    void axpy(ENTRY_T alpha, const PartitionedVector &x) {
        axpy(Scalar<ENTRY_T>(ctx, rt, alpha), x);
    }

    void axpy(
        const Scalar<ENTRY_T> &numer,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    );

    void axpy(
        const Scalar<ENTRY_T> &numer1,
        const Scalar<ENTRY_T> &numer2,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    );

    void xpay(const Scalar<ENTRY_T> &alpha, const PartitionedVector &x);

    void xpay(ENTRY_T alpha, const PartitionedVector &x) {
        xpay(Scalar<ENTRY_T>(ctx, rt, alpha), x);
    }

    void xpay(
        const Scalar<ENTRY_T> &numer,
        const Scalar<ENTRY_T> &denom,
        const PartitionedVector &x
    );

    Scalar<ENTRY_T> dot(const PartitionedVector &x) const;

}; // class PartitionedVector


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_PARTITIONED_VECTOR_HPP_INCLUDED
