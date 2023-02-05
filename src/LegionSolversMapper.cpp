#include "LegionSolversMapper.hpp"

#include <cassert> // for assert
#include <set>     // for std::set

#include <mappers/logging_wrapper.h> // for Legion::Mapping::LoggingWrapper

#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_MAPPER_ID,
                               //     LEGION_SOLVERS_TASK_ID_ORIGIN
#include "TaskBaseClasses.hpp" // for LEGION_SOLVERS_TASK_BLOCK_SIZE
#include "TaskIDs.hpp"         // for NUM_META_TASK_IDS

using LegionSolvers::LegionSolversMapper;

#ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION
using LegionSolvers::BlockingShardingFunctor;
#endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION


LegionSolversMapper::LegionSolversMapper(
    Legion::Mapping::MapperRuntime *rt,
    Legion::Machine machine,
    Legion::Processor local_proc
)
    : Legion::Mapping::DefaultMapper(rt, machine, local_proc) {
    // std::set<Legion::Processor> all_procs;
    // Legion::Machine::get_machine().get_all_processors(all_procs);
    // for (const Legion::Processor &proc : all_procs) {
    //     const Legion::AddressSpace addr = proc.address_space();
    //     const Legion::Processor::Kind kind = proc.kind();
    //     address_spaces.push_back(addr);
    //     if (kind == Legion::Processor::LOC_PROC) {
    //         cpus[addr].push_back(proc);
    //     } else if (kind == Legion::Processor::TOC_PROC) {
    //         gpus[addr].push_back(proc);
    //     }
    // }
    // std::sort(address_spaces.begin(), address_spaces.end());
    // const auto last = std::unique(address_spaces.begin(),
    // address_spaces.end()); address_spaces.erase(last, address_spaces.end());
    // assert(address_spaces.size() == cpus.size());
    // assert(address_spaces.size() == gpus.size());
    // for (const auto &addr : address_spaces) {
    //     std::sort(cpus[addr].begin(), cpus[addr].end());
    //     std::sort(gpus[addr].begin(), gpus[addr].end());
    // }
}


const char *LegionSolversMapper::get_mapper_name() const {
    return "LegionSolversMapper";
}


void LegionSolversMapper::memoize_operation(
    const Legion::Mapping::MapperContext,
    const Legion::Mappable &,
    const MemoizeInput &,
    MemoizeOutput &output
) {
    output.memoize = true;
}


void LegionSolversMapper::slice_task(
    const Legion::Mapping::MapperContext ctx,
    const Legion::Task &task,
    const SliceTaskInput &input,
    SliceTaskOutput &output
) {
    Legion::Mapping::DefaultMapper::slice_task(ctx, task, input, output);
}


#ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION
void LegionSolversMapper::select_sharding_functor(
    const Legion::Mapping::MapperContext,
    const Legion::Task &,
    const SelectShardingFunctorInput &,
    SelectShardingFunctorOutput &output
) {
    output.chosen_functor = LEGION_SOLVERS_SHARDING_FUNCTOR_ID;
}
#endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION


// Legion::Processor LegionSolversMapper::get_gpu(Legion::coord_t i) {
//     while (true) {
//         for (const auto &addr : address_spaces) {
//             // cast to avoid signed/unsigned comparison
//             if (i < static_cast<Legion::coord_t>(gpus[addr].size())) {
//                 return gpus[addr][i];
//             }
//             i -= gpus[addr].size();
//         }
//     }
// }


bool LegionSolversMapper::is_task(
    const Legion::TaskID task_id, const Legion::TaskID block_id
) {
    const Legion::TaskID block_origin =
        LEGION_SOLVERS_TASK_ID_ORIGIN + NUM_META_TASK_IDS +
        LEGION_SOLVERS_TASK_BLOCK_SIZE * block_id;
    return (block_origin <= task_id) &&
           (task_id < block_origin + LEGION_SOLVERS_TASK_BLOCK_SIZE);
}


void LegionSolvers::mapper_registration_callback(
    Legion::Machine machine,
    Legion::Runtime *rt,
    const std::set<Legion::Processor> &local_procs
) {
    for (const Legion::Processor &proc : local_procs) {
        rt->add_mapper(
            LEGION_SOLVERS_MAPPER_ID,
            new LegionSolversMapper(rt->get_mapper_runtime(), machine, proc),
            proc
        );
    }
}


#ifdef LEGION_SOLVERS_USE_CONTROL_REPLICATION

Legion::ShardID BlockingShardingFunctor::shard(
    const Legion::DomainPoint &point,
    const Legion::Domain &domain,
    std::size_t total_shards
) {
    assert(domain.get_dim() == 1);
    assert(domain.dense());
    assert(domain.lo()[0] == 0);
    const std::size_t points_per_shard =
        (domain.get_volume() + total_shards - 1) / total_shards;
    return static_cast<Legion::ShardID>(point[0] / points_per_shard);
}

#endif // LEGION_SOLVERS_USE_CONTROL_REPLICATION
