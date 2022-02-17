
        void matvec(
            DistributedVectorT<
                ENTRY_T, RANGE_DIM, 1, RANGE_COORD_T, Legion::coord_t
            > &output_vector,
            const DistributedVectorT<
                ENTRY_T, DOMAIN_DIM, 1, DOMAIN_COORD_T, Legion::coord_t
            > &input_vector,
            Legion::IndexSpaceT<3> tile_index_space
        ) const {
            const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            Legion::IndexLauncher launcher{
                COOMatvecTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::task_id,
                tile_index_space,
                Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
                Legion::ArgumentMap{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;

            launcher.add_region_requirement(Legion::RegionRequirement{
                output_vector.logical_partition,
                PFID_KDR_TO_R, LEGION_REDOP_SUM<ENTRY_T>, LEGION_SIMULTANEOUS,
                output_vector.logical_region
            });
            launcher.add_field(0, output_vector.fid);

            launcher.add_region_requirement(Legion::RegionRequirement{
                kernel_logical_partition,
                PFID_KDR_TO_K, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                kernel_region
            });
            launcher.add_field(1, fid_i);
            launcher.add_field(1, fid_j);
            launcher.add_field(1, fid_entry);

            launcher.add_region_requirement(Legion::RegionRequirement{
                input_vector.logical_partition,
                PFID_KDR_TO_D, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                input_vector.logical_region
            });
            launcher.add_field(2, input_vector.fid);

            rt->execute_index_space(ctx, launcher);
        }

        virtual void print() const override {
            // TODO: index print?
            const Legion::FieldID fids[3] = {fid_i, fid_j, fid_entry};
            Legion::TaskLauncher launcher{
                COOPrintTask<ENTRY_T, KERNEL_DIM, DOMAIN_DIM, RANGE_DIM>::task_id,
                Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])}
            };
            // Legion::IndexLauncher launcher{
            //     COOPrintTask<ENTRY_T, KERNEL_DIM,
            //                  DOMAIN_DIM, RANGE_DIM>::task_id,
            //     this->tile_index_space,
            //     Legion::TaskArgument{&fids, sizeof(Legion::FieldID[3])},
            //     Legion::ArgumentMap{}
            // };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                kernel_region, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
                kernel_region
            });
            // launcher.add_region_requirement(Legion::RegionRequirement{
            //     this->column_logical_partition, PFID_IJ_TO_IJ,
            //     LEGION_READ_ONLY, LEGION_EXCLUSIVE, this->matrix_region
            // });
            launcher.add_field(0, fid_i);
            launcher.add_field(0, fid_j);
            launcher.add_field(0, fid_entry);
            // rt->execute_index_space(ctx, launcher);
            rt->execute_task(ctx, launcher);
        }
