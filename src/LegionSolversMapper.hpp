#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED

#include <map>    // for std::map
#include <vector> // for std::vector

#include <legion.h>                 // for Legion::*
#include <mappers/default_mapper.h> // for Legion::Mapping::DefaultMapper


namespace LegionSolvers {


class LegionSolversMapper : public Legion::Mapping::DefaultMapper {

    // std::vector<Legion::AddressSpace> address_spaces;
    // std::map<Legion::AddressSpace, std::vector<Legion::Processor>> cpus;
    // std::map<Legion::AddressSpace, std::vector<Legion::Processor>> gpus;

public:

    LegionSolversMapper(
        Legion::Mapping::MapperRuntime *rt,
        Legion::Machine machine,
        Legion::Processor local_proc
    );

    virtual const char *get_mapper_name() const override;

    virtual void memoize_operation(
        const Legion::Mapping::MapperContext ctx,
        const Legion::Mappable &mappable,
        const MemoizeInput &input,
        MemoizeOutput &output
    ) override;

    virtual void slice_task(
        const Legion::Mapping::MapperContext ctx,
        const Legion::Task &task,
        const SliceTaskInput &input,
        SliceTaskOutput &output
    ) override;

    virtual void default_policy_select_constraints(
        Legion::Mapping::MapperContext ctx,
        Legion::LayoutConstraintSet &constraints,
        Legion::Memory target_memory,
        const Legion::RegionRequirement &req
    ) override;

#ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION
    virtual void select_sharding_functor(
        const Legion::Mapping::MapperContext ctx,
        const Legion::Task &task,
        const SelectShardingFunctorInput &input,
        SelectShardingFunctorOutput &output
    ) override;
#endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION

    // Legion::Processor get_gpu(Legion::coord_t i);

    static bool is_task(Legion::TaskID task_id, Legion::TaskID block_id);

}; // class LegionSolversMapper


void mapper_registration_callback(
    Legion::Machine machine,
    Legion::Runtime *rt,
    const std::set<Legion::Processor> &local_procs
);


#ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION

struct BlockingShardingFunctor : public Legion::ShardingFunctor {

    virtual Legion::ShardID shard(
        const Legion::DomainPoint &point,
        const Legion::Domain &domain,
        std::size_t total_shards
    ) override;

}; // struct BlockingShardingFunctor

#endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED
