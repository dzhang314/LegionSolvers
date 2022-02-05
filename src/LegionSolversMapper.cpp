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


void LegionSolvers::LegionSolversMapper::slice_task(
    const Legion::Mapping::MapperContext ctx,
    const Legion::Task &task,
    const SliceTaskInput& input,
    SliceTaskOutput& output
) {
    if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {
        const Legion::DomainPoint lo = input.domain.lo();
        const Legion::DomainPoint hi = input.domain.hi();
        if (lo.get_dim() == 3) {
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
            // for (auto &slice : output.slices) {
            //     std::cout << "Assigning " << slice.domain << " to rank "
            //                 << slice.proc.address_space() << std::endl;
            // }
            output.verify_correctness = true;
        } else {
            for (Legion::coord_t i = lo[0]; i <= hi[0]; i++) {
                const Legion::Rect<1> rect{
                    Legion::Point<1>{i},
                    Legion::Point<1>{i}
                };
                output.slices.emplace_back(
                    input.domain.intersection(Legion::Domain{rect}),
                    gpus[address_spaces[i % address_spaces.size()]],
                    false, // do not recursively call slice_task
                    false // TODO: should this be stealable?
                );
            }
            // for (auto &slice : output.slices) {
            //     std::cout << "Assigning " << slice.domain << " to rank "
            //                 << slice.proc.address_space() << std::endl;
            // }
            output.verify_correctness = true;
        }
    } else if (is_task(task.task_id, AXPY_TASK_BLOCK_ID) ||
               is_task(task.task_id, XPAY_TASK_BLOCK_ID) ||
               is_task(task.task_id, DOT_TASK_BLOCK_ID) ||
               is_task(task.task_id, FILL_COO_NEGATIVE_LAPLACIAN_1D_TASK_BLOCK_ID)) {
        const Legion::DomainPoint lo = input.domain.lo();
        const Legion::DomainPoint hi = input.domain.hi();
        for (Legion::coord_t i = lo[0]; i <= hi[0]; i++) {
            const Legion::Rect<1> rect{
                Legion::Point<1>{i},
                Legion::Point<1>{i}
            };
            output.slices.emplace_back(
                input.domain.intersection(Legion::Domain{rect}),
                gpus[address_spaces[i % address_spaces.size()]],
                false, // do not recursively call slice_task
                false // TODO: should this be stealable?
            );
        }
    } else {
        Legion::Mapping::DefaultMapper::slice_task(
            ctx, task, input, output
        );
    }
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
