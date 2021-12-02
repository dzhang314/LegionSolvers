


#include <mappers/logging_wrapper.h>

#include "TaskIDs.hpp"


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


        // virtual void select_task_options(
        //     const Legion::Mapping::MapperContext ctx,
        //     const Legion::Task &task,
        //     TaskOptions &output
        // ) override {
        //     Legion::Mapping::DefaultMapper::select_task_options(ctx, task, output);
        //     output.valid_instances = true;
        // }


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
                // for (auto &slice : output.slices) {
                //     std::cout << "Assigning " << slice.domain << " to rank "
                //               << slice.proc.address_space() << std::endl;
                // }
                // output.verify_correctness = true;
            } else {
                Legion::Mapping::DefaultMapper::slice_task(ctx, task, input, output);
            }
        }


        // virtual void map_task(const Legion::Mapping::MapperContext ctx,
        //                       const Legion::Task &task,
        //                       const MapTaskInput &input,
        //                       MapTaskOutput &output) override {

        //     if (is_task(task.task_id, DUMMY_TASK_BLOCK_ID)) {

        //         Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

        //     } else if (is_task(task.task_id, COO_MATVEC_TASK_BLOCK_ID)) {

        //         assert(input.valid_instances.size() == 3);
        //         assert(task.is_index_space);
        //         Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);

        //     } else {
        //         Legion::Mapping::DefaultMapper::map_task(ctx, task, input, output);
        //     }
        // }
