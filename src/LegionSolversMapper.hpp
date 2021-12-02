#ifndef LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
#define LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP

#include <map>
#include <set>
#include <vector>

#include <legion.h>
#include <mappers/default_mapper.h>
#include <mappers/logging_wrapper.h>

#include "LibraryOptions.hpp"
#include "TaskBaseClasses.hpp"
#include "TaskIDs.hpp"


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
            // TODO: What is the correct model?
            return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
        }

        virtual const char *get_mapper_name() const override {
            return "legion_solvers_mapper";
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
                const Legion::DomainPoint lo = input.domain.lo();
                const Legion::DomainPoint hi = input.domain.hi();
                for (Legion::coord_t i = lo[2]; i <= hi[2]; i++) {
                    const Legion::Rect<3> rect{
                        Legion::Point<3>{lo[0], lo[1], i},
                        Legion::Point<3>{hi[0], hi[1], i}
                    };
                    output.slices.emplace_back(
                        input.domain.intersection(Legion::Domain{rect}),
                        gpus[address_spaces[i % address_spaces.size()]],
                        false, // do not recursively call slice_task
                        false // TODO: should this be stealable?
                    );
                }
                for (auto &slice : output.slices) {
                    std::cout << "Assigning " << slice.domain << " to rank "
                              << slice.proc.address_space() << std::endl;
                }
                output.verify_correctness = true;
            } else {
                Legion::Mapping::DefaultMapper::slice_task(
                    ctx, task, input, output
                );
            }
        }

    }; // class LegionSolversMapper


    void mapper_registration_callback(
        Legion::Machine machine,
        Legion::Runtime *rt,
        const std::set<Legion::Processor> &local_procs
    );


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_LEGION_SOLVERS_MAPPER_HPP
