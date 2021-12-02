#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP

#include <map>
#include <set>
#include <vector>

#include <legion.h>
#include <mappers/default_mapper.h>
#include <mappers/logging_wrapper.h>


namespace LegionSolvers {


    class LegionSolversMapper : public Legion::Mapping::DefaultMapper {

    public:

        std::vector<Legion::AddressSpace> address_spaces;
        std::map<Legion::AddressSpace, Legion::Processor> cpus;
        std::map<Legion::AddressSpace, Legion::Processor> gpus;

        LegionSolversMapper(
            Legion::Mapping::MapperRuntime *rt,
            Legion::Machine machine,
            Legion::Processor local
        );

        virtual Legion::Mapping::Mapper::MapperSyncModel
        get_mapper_sync_model() const override {
            return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
        }

        virtual const char *get_mapper_name() const override {
            return "legion_solvers_mapper";
        }

    }; // class LegionSolversMapper


    void mapper_registration_callback(
        Legion::Machine machine,
        Legion::Runtime *rt,
        const std::set<Legion::Processor> &local_procs
    );


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
