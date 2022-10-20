#include "LegionSolversMapper.hpp"

#include <mappers/logging_wrapper.h> // for Legion::Mapping::LoggingWrapper

#include "LibraryOptions.hpp" // for LEGION_SOLVERS_MAPPER_ID


LegionSolvers::LegionSolversMapper::LegionSolversMapper(
    Legion::Mapping::MapperRuntime *rt,
    Legion::Machine machine,
    Legion::Processor local_proc
)
    : Legion::Mapping::DefaultMapper(rt, machine, local_proc) {}


void LegionSolvers::mapper_registration_callback(
    Legion::Machine machine,
    Legion::Runtime *rt,
    const std::set<Legion::Processor> &local_procs
) {
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
