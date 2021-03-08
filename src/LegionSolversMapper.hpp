#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP

#include <legion.h>
#include <mappers/default_mapper.h>
#include <mappers/logging_wrapper.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    class LegionSolversMapper : public Legion::Mapping::DefaultMapper {

      public:
        LegionSolversMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine machine, Legion::Processor local)
            : Legion::Mapping::DefaultMapper(rt, machine, local) {}

        virtual Legion::Mapping::Mapper::MapperSyncModel get_mapper_sync_model(void) const override {
            return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
        }

        virtual const char *get_mapper_name(void) const override { return "legion_solvers_mapper"; }

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
        //     bool                                   memoize;  // = false
        //     bool                                   replicate; // = false
        //     TaskPriority                           parent_priority; // = current
        //   };

        virtual void select_task_options(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task &task,
                                         TaskOptions &output) override {
            Legion::Mapping::DefaultMapper::select_task_options(ctx, task, output);
            output.valid_instances = true;
            // output.initial_proc = Legion::Processor(0);
        }

        static bool is_task(Legion::TaskID task_id, Legion::TaskID block_id) {
            const Legion::TaskID block_origin =
                LEGION_SOLVERS_TASK_ID_ORIGIN +
                LEGION_SOLVERS_TASK_BLOCK_SIZE * block_id;
            return (block_origin <= task_id) &&
                   (task_id < block_origin + LEGION_SOLVERS_TASK_BLOCK_SIZE);
        }

        virtual void map_task(const Legion::Mapping::MapperContext ctx,
                              const Legion::Task &task,
                              const MapTaskInput &input,
                              MapTaskOutput &output) override {
            if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {

                // assert(input.valid_instances.size() == 3);
                // std::cout << "MAPPING TASK: " << task.get_task_name() << " (id " << task.task_id << ")" << std::endl;
                // for (std::size_t i = 0; i < 3; ++i) {
                //     std::cout << "    valid instances for " << i << ":" << std::endl;
                //     for (const Legion::Mapping::PhysicalInstance &instance : input.valid_instances[i]) {
                //         std::cout << "        " << instance.get_location() << std::endl;
                //     }
                // }

                // bool create_physical_instance(
                //   MapperContext ctx, Memory target_memory,
                //   const LayoutConstraintSet &constraints,
                //   const std::vector<LogicalRegion> &regions,
                //   PhysicalInstance &result) const;

                Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

                // assert(output.chosen_instances.size() == 3);
                // for (std::size_t i = 0; i < 3; ++i) {
                //     std::cout << "    chosen instances for " << i << ":" << std::endl;
                //     for (const Legion::Mapping::PhysicalInstance &instance : output.chosen_instances[i]) {
                //         std::cout << "        " << instance.get_location() << std::endl;
                //     }
                // }

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
