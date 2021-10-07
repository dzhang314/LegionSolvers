#ifndef LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
#define LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP

#include <legion.h>

#include "LegionUtilities.hpp"
#include "UtilityTasks.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T,
              int DIM = 1, typename COORD_T = Legion::coord_t,
              int COLOR_DIM = 1, typename COLOR_COORD_T = Legion::coord_t>
    struct DistributedVector {

        static constexpr Legion::FieldID DEFAULT_FID = 101;

    public:

        Legion::Context ctx;
        Legion::Runtime *rt;
        Legion::IndexSpaceT<DIM, COORD_T> index_space;
        Legion::LogicalRegionT<DIM, COORD_T> logical_region;
        Legion::FieldID fid;
        Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space;
        Legion::IndexPartitionT<DIM> index_partition;
        Legion::LogicalPartitionT<DIM, COORD_T> logical_partition;

        DistributedVector(Legion::IndexSpaceT<DIM, COORD_T> index_space,
                          Legion::IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
                          Legion::Context ctx, Legion::Runtime *rt) :
            ctx(ctx),
            rt(rt),
            index_space(index_space),
            logical_region(LegionSolvers::create_region(
                index_space, {{sizeof(ENTRY_T), DEFAULT_FID}}, ctx, rt
            )),
            fid(DEFAULT_FID),
            color_space(color_space),
            index_partition(
                rt->create_equal_partition(ctx, index_space, color_space)
            ),
            logical_partition(
                rt->get_logical_partition(logical_region, index_partition)
            ) {}

        void zero_fill() {
            static constexpr ENTRY_T zero = static_cast<ENTRY_T>(0);
            Legion::IndexFillLauncher launcher{
                color_space, logical_partition, logical_region,
                Legion::TaskArgument{&zero, sizeof(ENTRY_T)}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        void print() {
            static const std::string name{"hello"};
            Legion::IndexTaskLauncher launcher{
                PrintVectorTask<ENTRY_T, DIM>::task_id, color_space,
                Legion::TaskArgument{name.c_str(), name.length() + 1},
                Legion::ArgumentMap{}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_partition, 0,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_index_space(ctx, launcher);
        }

    }; // class DistributedVector


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DISTRIBUTED_VECTOR_HPP
