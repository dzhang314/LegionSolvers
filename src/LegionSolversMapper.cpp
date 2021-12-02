#include "LegionSolversMapper.hpp"

#include <algorithm>
#include <cassert>

#include "LibraryOptions.hpp"


LegionSolvers::LegionSolversMapper::LegionSolversMapper(
    Legion::Mapping::MapperRuntime *rt,
    Legion::Machine machine,
    Legion::Processor local
) : Legion::Mapping::DefaultMapper(rt, machine, local) {
    std::set<Legion::Processor> all_procs{};
    Legion::Machine::get_machine().get_all_processors(all_procs);
    for (const Legion::Processor &proc : all_procs) {
        const Legion::AddressSpace addr = proc.address_space();
        const Legion::Processor::Kind kind = proc.kind();
        address_spaces.push_back(addr);
        if (kind == Legion::Processor::LOC_PROC) {
            assert(cpus.find(addr) == cpus.end());
            cpus[addr] = proc;
        } else if (kind == Legion::Processor::TOC_PROC) {
            assert(gpus.find(addr) == gpus.end());
            gpus[addr] = proc;
        }
    }
    std::sort(address_spaces.begin(), address_spaces.end());
    const auto last = std::unique(address_spaces.begin(),
                                  address_spaces.end());
    address_spaces.erase(last, address_spaces.end());
    assert(address_spaces.size() == cpus.size());
    assert(address_spaces.size() == gpus.size());
}


void LegionSolvers::mapper_registration_callback(
    Legion::Machine machine,
    Legion::Runtime *rt,
    const std::set<Legion::Processor> &local_procs
) {
    for (const Legion::Processor &proc : local_procs) {
        rt->add_mapper(
            LEGION_SOLVERS_MAPPER_ID,
            new Legion::Mapping::LoggingWrapper(
                new LegionSolversMapper(
                    rt->get_mapper_runtime(), machine, proc
                )
            ),
            proc
        );
    }
}
