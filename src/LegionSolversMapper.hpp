#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP

#include <map>

#include <legion.h>
#include <mappers/default_mapper.h>
#include <mappers/logging_wrapper.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    class LegionSolversMapper : public Legion::Mapping::DefaultMapper {


        std::map<Legion::DomainPoint, Legion::Memory> memory_map;
        const Legion::AddressSpace num_address_spaces;


        static Legion::AddressSpace get_num_address_spaces() {
            std::set<Legion::Processor> all_procs{};
            Legion::Machine::get_machine().get_all_processors(all_procs);
            Legion::AddressSpace result = 0;
            for (const Legion::Processor &proc : all_procs) {
                result = std::max(result, proc.address_space());
            }
            return result + 1;
        }


      public:
        LegionSolversMapper(Legion::Mapping::MapperRuntime *rt,
                            Legion::Machine machine,
                            Legion::Processor local)
            : Legion::Mapping::DefaultMapper(rt, machine, local),
              memory_map(),
              num_address_spaces(get_num_address_spaces()) {}


        virtual Legion::Mapping::Mapper::MapperSyncModel
        get_mapper_sync_model(void) const override {
            return CONCURRENT_MAPPER_MODEL;
        }


        virtual const char *get_mapper_name(void) const override {
            return "legion_solvers_mapper";
        }


        //   struct MapTaskInput {
        //     std::vector<std::vector<PhysicalInstance> >     valid_instances;
        //     std::vector<unsigned>                           premapped_regions;
        //   };

        //   struct MapTaskOutput {
        //     std::vector<std::vector<PhysicalInstance> >     chosen_instances;
        //     std::set<unsigned>                              untracked_valid_regions;
        //     std::vector<Processor>                          target_procs;
        //     VariantID                                       chosen_variant; // = 0
        //     TaskPriority                                    task_priority;  // = 0
        //     TaskPriority                                    profiling_priority;
        //     ProfilingRequest                                task_prof_requests;
        //     ProfilingRequest                                copy_prof_requests;
        //     bool                                            postmap_task; // = false
        //   };

        //   struct TaskOptions {
        //     Processor                              initial_proc; // = current
        //     bool                                   inline_task;  // = false
        //     bool                                   stealable;   // = false
        //     bool                                   map_locally;  // = false
        //     bool                                   valid_instances; // = true
        //     bool                                   replicate; // = false
        //     TaskPriority                           parent_priority; // = current
        //   };


        virtual void select_task_options(
            const Legion::Mapping::MapperContext ctx,
            const Legion::Task &task,
            TaskOptions &output
        ) override {
            Legion::Mapping::DefaultMapper::select_task_options(ctx, task, output);
            output.valid_instances = true;
        }


        virtual void memoize_operation(
            const Legion::Mapping::MapperContext ctx,
            const Legion::Mappable &mappable,
            const MemoizeInput &input,
            MemoizeOutput &output
        ) override {
            output.memoize = true;
        }


        static constexpr bool is_task(
            Legion::TaskID task_id, Legion::TaskID block_id
        ) noexcept {
            const Legion::TaskID block_origin =
                LEGION_SOLVERS_TASK_ID_ORIGIN +
                LEGION_SOLVERS_TASK_BLOCK_SIZE * block_id;
            return (block_origin <= task_id) &&
                   (task_id < block_origin + LEGION_SOLVERS_TASK_BLOCK_SIZE);
        }


        virtual void slice_task(const Legion::Mapping::MapperContext ctx,
                                const Legion::Task &task,
                                const SliceTaskInput& input,
                                SliceTaskOutput& output) override {
            if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {
                std::set<Legion::Processor> all_procs{};
                Legion::Machine::get_machine().get_all_processors(all_procs);
                Legion::Processor cpus[num_address_spaces];
                Legion::Processor gpus[num_address_spaces];
                for (const Legion::Processor &proc : all_procs) {
                    if (proc.kind() == Legion::Processor::LOC_PROC) {
                        cpus[proc.address_space()] = proc;
                    } else if (proc.kind() == Legion::Processor::TOC_PROC) {
                        gpus[proc.address_space()] = proc;
                    }
                }
                const Legion::DomainPoint lo = input.domain.lo();
                const Legion::DomainPoint hi = input.domain.hi();
                for (Legion::coord_t range_index = lo[2]; range_index <= hi[2]; range_index++) {
                    const Legion::coord_t rank = range_index % num_address_spaces;
                    const Legion::Rect<3> rect{
                        Legion::Point<3>{lo[0], lo[1], range_index},
                        Legion::Point<3>{hi[0], hi[1], range_index}
                    };
                    output.slices.emplace_back(
                        input.domain.intersection(Legion::Domain{rect}),
                        gpus[rank], false, true
                    );
                }
                // for (auto &slice : output.slices) {
                //     std::cout << "Assigning " << slice.domain << " to rank "
                //               << slice.proc.address_space() << std::endl;
                // }
                // output.verify_correctness = true;
            } else {
                Legion::Mapping::DefaultMapper::slice_task(ctx, task, input, output);
            }
        }


        virtual void map_task(const Legion::Mapping::MapperContext ctx,
                              const Legion::Task &task,
                              const MapTaskInput &input,
                              MapTaskOutput &output) override {

            if (is_task(task.task_id, DUMMY_TASK_BLOCK_ID)) {

                Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

            } else if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {

                assert(input.valid_instances.size() == 3);
                assert(task.is_index_space);
                Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

            } else {
                Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);
            }
        }


    }; // class LegionSolversMapper


    void mapper_registration_callback(Legion::Machine machine,
                                      Legion::Runtime *rt,
                                      const std::set<Legion::Processor> &local_procs) {
        for (const Legion::Processor &proc : local_procs) {
            rt->add_mapper(
                LEGION_SOLVERS_MAPPER_ID,
                new Legion::Mapping::LoggingWrapper(
                    new LegionSolversMapper(rt->get_mapper_runtime(), machine, proc)
                ),
                proc
            );
        }
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
