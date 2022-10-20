#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED

#include <legion.h>                 // for Legion::*
#include <mappers/default_mapper.h> // for Legion::Mapping::DefaultMapper

namespace LegionSolvers {


class LegionSolversMapper : public Legion::Mapping::DefaultMapper {

  public:

    LegionSolversMapper(
        Legion::Mapping::MapperRuntime *rt,
        Legion::Machine machine,
        Legion::Processor local_proc
    );

    // TODO: add rest of LegionSolversMapper + thermodynamic mapping strategy

}; // class LegionSolversMapper


void mapper_registration_callback(
    Legion::Machine machine,
    Legion::Runtime *rt,
    const std::set<Legion::Processor> &local_procs
);


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP_INCLUDED
