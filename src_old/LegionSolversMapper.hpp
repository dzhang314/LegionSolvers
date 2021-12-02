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
