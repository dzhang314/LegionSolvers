#ifndef LEGION_SOLVERS_DENSE_VECTOR_HPP
#define LEGION_SOLVERS_DENSE_VECTOR_HPP

#include <string>

#include <legion.h>

#include "LinearAlgebraTasks.hpp"
#include "Scalar.hpp"


namespace LegionSolvers {


    template <typename ENTRY_T>
    class DenseVector {

        Legion::Context ctx;
        Legion::Runtime *rt;
        std::string name;
        Legion::IndexSpace index_space;
        Legion::FieldID fid;
        Legion::FieldSpace field_space;
        Legion::LogicalRegion logical_region;

    public:

        DenseVector(DenseVector &&) = delete;
        DenseVector(const DenseVector &) = delete;
        DenseVector &operator=(DenseVector &&) = delete;

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::coord_t n
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<1>{0, n - 1}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {
            const std::string index_space_name    = name + "_index_space"   ;
            const std::string field_space_name    = name + "_field_space"   ;
            const std::string logical_region_name = name + "_logical_region";
            rt->attach_name(index_space   , index_space_name   .c_str());
            rt->attach_name(field_space   , field_space_name   .c_str());
            rt->attach_name(logical_region, logical_region_name.c_str());
        }

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::coord_t m, Legion::coord_t n
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<2>{{0, 0}, {m - 1, n - 1}}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {
            const std::string index_space_name    = name + "_index_space"   ;
            const std::string field_space_name    = name + "_field_space"   ;
            const std::string logical_region_name = name + "_logical_region";
            rt->attach_name(index_space   , index_space_name   .c_str());
            rt->attach_name(field_space   , field_space_name   .c_str());
            rt->attach_name(logical_region, logical_region_name.c_str());
        }

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::coord_t x, Legion::coord_t y, Legion::coord_t z
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<3>{{0, 0, 0}, {x - 1, y - 1, z - 1}}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {
            const std::string index_space_name    = name + "_index_space"   ;
            const std::string field_space_name    = name + "_field_space"   ;
            const std::string logical_region_name = name + "_logical_region";
            rt->attach_name(index_space   , index_space_name   .c_str());
            rt->attach_name(field_space   , field_space_name   .c_str());
            rt->attach_name(logical_region, logical_region_name.c_str());
        }

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            const std::string &name,
            Legion::IndexSpace index_space
        ) : ctx(ctx), rt(rt),
            name(name),
            index_space(index_space),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {
            rt->create_shared_ownership(ctx, index_space);
            const std::string field_space_name    = name + "_field_space"   ;
            const std::string logical_region_name = name + "_logical_region";
            rt->attach_name(field_space   , field_space_name   .c_str());
            rt->attach_name(logical_region, logical_region_name.c_str());
        }

        ~DenseVector() {
            rt->destroy_logical_region(ctx, logical_region);
            rt->destroy_field_space   (ctx, field_space   );
            rt->destroy_index_space   (ctx, index_space   );
        }

        Legion::IndexSpace    get_index_space   () const { return index_space   ; }
        Legion::FieldID       get_fid           () const { return fid           ; }
        Legion::FieldSpace    get_field_space   () const { return field_space   ; }
        Legion::LogicalRegion get_logical_region() const { return logical_region; }

        void constant_fill(ENTRY_T value) {
            Legion::FillLauncher launcher{
                logical_region, logical_region,
                Legion::UntypedBuffer{&value, sizeof(ENTRY_T)}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        void constant_fill(const Scalar<ENTRY_T> &value) {
            Legion::FillLauncher launcher{
                logical_region, logical_region, value.get_future()
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_field(fid);
            rt->fill_fields(ctx, launcher);
        }

        void zero_fill() {
            constant_fill(static_cast<ENTRY_T>(0));
        }

        ENTRY_T operator=(ENTRY_T value) {
            constant_fill(value);
            return value;
        }

        const Scalar<ENTRY_T> &operator=(const Scalar<ENTRY_T> &value) {
            constant_fill(value);
            return value;
        }

        DenseVector &operator=(const DenseVector &x) {
            assert(index_space == x.get_index_space());
            Legion::CopyLauncher launcher{};
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_copy_requirements(
                Legion::RegionRequirement{
                    x.get_logical_region(),
                    LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
                },
                Legion::RegionRequirement{
                    logical_region,
                    LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
                }
            );
            launcher.add_src_field(0, x.get_fid());
            launcher.add_dst_field(0, fid);
            rt->issue_copy_operation(ctx, launcher);
            return *this;
        }

        void random_fill(
            ENTRY_T low = static_cast<ENTRY_T>(0),
            ENTRY_T high = static_cast<ENTRY_T>(1)
        ) {
            const ENTRY_T args[2] = {low, high};
            Legion::TaskLauncher launcher{
                RandomFillTask<ENTRY_T, 0, void>::task_id(index_space),
                Legion::UntypedBuffer{&args, 2 * sizeof(ENTRY_T)}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_region,
                LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_task(ctx, launcher);
        }

        void axpy(const Scalar<ENTRY_T> &alpha, const DenseVector &x) {
            assert(index_space == x.get_index_space());
            Legion::TaskLauncher launcher{
                AxpyTask<ENTRY_T, 0, void>::task_id(index_space),
                Legion::UntypedBuffer{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_region,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_region(),
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(alpha.get_future());
            rt->execute_task(ctx, launcher);
        }

        void axpy(ENTRY_T alpha, const DenseVector &x) {
            axpy(Scalar<ENTRY_T>{ctx, rt, alpha}, x);
        }

        void xpay(const Scalar<ENTRY_T> &alpha, const DenseVector &x) {
            assert(index_space == x.get_index_space());
            Legion::TaskLauncher launcher{
                XpayTask<ENTRY_T, 0, void>::task_id(index_space),
                Legion::UntypedBuffer{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_region,
                LEGION_READ_WRITE, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_region(),
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            launcher.add_future(alpha.get_future());
            rt->execute_task(ctx, launcher);
        }

        void xpay(ENTRY_T alpha, const DenseVector &x) {
            xpay(Scalar<ENTRY_T>{ctx, rt, alpha}, x);
        }

        Scalar<ENTRY_T> dot(const DenseVector &x) const {
            assert(index_space == x.get_index_space());
            Legion::TaskLauncher launcher{
                DotTask<ENTRY_T, 0, void>::task_id(index_space),
                Legion::UntypedBuffer{}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_region,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            launcher.add_region_requirement(Legion::RegionRequirement{
                x.get_logical_region(),
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, x.get_logical_region()
            });
            launcher.add_field(1, x.get_fid());
            return Scalar<ENTRY_T>{ctx, rt, rt->execute_task(ctx, launcher)};
        }

        void print() const {
            Legion::TaskLauncher launcher{
                PrintVectorTask<ENTRY_T, 0, void>::task_id(index_space),
                Legion::UntypedBuffer{name.c_str(), name.length() + 1}
            };
            launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_region_requirement(Legion::RegionRequirement{
                logical_region,
                LEGION_READ_ONLY, LEGION_EXCLUSIVE, logical_region
            });
            launcher.add_field(0, fid);
            rt->execute_task(ctx, launcher);
        }

    }; // class DenseVector


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DENSE_VECTOR_HPP
