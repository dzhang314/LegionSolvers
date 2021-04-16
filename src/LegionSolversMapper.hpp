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
            return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
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


        virtual void slice_task(const Legion::Mapping::MapperContext ctx,
                                const Legion::Task &task,
                                const SliceTaskInput& input,
                                SliceTaskOutput& output) override {
            Legion::Mapping::DefaultMapper::slice_task(ctx, task, input, output);
            if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {

                output.slices.clear();

                std::set<Legion::Processor> all_procs{};
                Legion::Machine::get_machine().get_all_processors(all_procs);
                Legion::Processor cpus[num_address_spaces];
                for (const Legion::Processor &proc : all_procs) {
                    if (proc.kind() == Legion::Processor::LOC_PROC) {
                        cpus[proc.address_space()] = proc;
                    }
                }

                const Legion::DomainPoint lo = input.domain.lo();
                const Legion::DomainPoint hi = input.domain.hi();
                for (Legion::coord_t tile_index = lo[1]; tile_index <= hi[1]; tile_index++) {
                    const Legion::coord_t rank = tile_index % num_address_spaces;
                    const Legion::Rect<2> rect{
                        Legion::Point<2>{input.domain.lo()[0], tile_index},
                        Legion::Point<2>{input.domain.hi()[0], tile_index}
                    };
                    output.slices.emplace_back(
                        input.domain.intersection(Legion::Domain{rect}),
                        cpus[rank], false, true
                    );
                }

                // for (auto &slice : output.slices) {
                //     std::cout << "Assigning " << slice.domain << " to rank "
                //               << slice.proc.address_space() << std::endl;
                // }
                // output.verify_correctness = true;

            }
        }


        virtual void map_task(const Legion::Mapping::MapperContext ctx,
                              const Legion::Task &task,
                              const MapTaskInput &input,
                              MapTaskOutput &output) override {

            if (is_task(task.task_id, DUMMY_TASK_BLOCK_ID)) {

                assert(input.valid_instances.size() == 1);
                assert(input.valid_instances[0].size() == 1);
                assert(task.is_index_space);

                Legion::Mapping::PhysicalInstance instance = *input.valid_instances[0].begin();
                memory_map[task.index_point] = instance.get_location();

                Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

            } else if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {

                assert(input.valid_instances.size() == 3);
                assert(task.is_index_space);
                const auto key = Legion::DomainPoint{task.index_point[0]};

                // std::cout << "MEMORY_MAP CONTENTS:" << std::endl;
                // for (auto it = memory_map.begin(); it != memory_map.end(); ++it) {
                //     std::cout << "    " << it->first << " : " << it->second << std::endl;
                // }

                // const auto it = memory_map.find(key);
                // if (it != memory_map.end()) {
                //     std::cout << "FOUND: " << it->second << std::endl;
                // } else {
                //     std::cout << "NOT FOUND" << std::endl;
                // }

                // std::cout << "MAPPING TASK: " << task.get_task_name() << " : " << task.index_point << " / " << task.index_domain << " (id " << task.task_id << ")" << std::endl;
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

            // } else if (is_task(task.task_id, AXPY_TASK_BLOCK_ID)) {

                // assert(input.valid_instances.size() == 2);
                // assert(task.is_index_space);
                // const auto key = std::make_pair(task.index_domain, task.index_point);

                // const auto it = memory_map.find(key);
                // if (it != memory_map.end()) {
                //     std::cout << it->second << std::endl;
                // }

                // std::cout << "MAPPING TASK: " << task.get_task_name() << " : " << task.index_point << " / " << task.index_domain << " (id " << task.task_id << ")" << std::endl;
                // for (std::size_t i = 0; i < 2; ++i) {
                //     std::cout << "    valid instances for " << i << ":" << std::endl;
                //     for (const Legion::Mapping::PhysicalInstance &instance : input.valid_instances[i]) {
                //         std::cout << "        " << instance.get_location() << std::endl;
                //     }
                // }

                // Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

                // assert(output.chosen_instances.size() == 2);
                // for (std::size_t i = 0; i < 2; ++i) {
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
