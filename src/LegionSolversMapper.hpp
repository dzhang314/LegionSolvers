#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP

#include <legion.h>
#include <mappers/default_mapper.h>

#include "TaskIDs.hpp"


namespace LegionSolvers {


    class LegionSolversMapper : public Legion::Mapping::DefaultMapper {

      public:
        LegionSolversMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine machine, Legion::Processor local)
            : Legion::Mapping::DefaultMapper(rt, machine, local) {}

        virtual const char *get_mapper_name(void) const override { return "legion_solvers_mapper"; }

        virtual void map_task(const Legion::Mapping::MapperContext ctx,
                              const Legion::Task &task,
                              const MapTaskInput &input,
                              MapTaskOutput &output) {
            std::cout << "MAPPING TASK: " << task.get_task_name() << std::endl;
            for (std::size_t i = 0; i < input.valid_instances.size(); ++i) {
                for (const Legion::Mapping::PhysicalInstance &instance : input.valid_instances[i]) {
                    std::cout << instance.get_location() << std::endl;
                }
            }
            Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);
        }

    }; // class LegionSolversMapper


    void mapper_registration_callback(Legion::Machine machine,
                                      Legion::Runtime *rt,
                                      const std::set<Legion::Processor> &local_procs) {
        for (const Legion::Processor &proc : local_procs) {
            rt->add_mapper(LEGION_SOLVERS_MAPPER_ID, new LegionSolversMapper(rt->get_mapper_runtime(), machine, proc),
                           proc);
        }
    }


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
