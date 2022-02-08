#ifndef LEGION_SOLVERS_DENSE_VECTOR_HPP
#define LEGION_SOLVERS_DENSE_VECTOR_HPP

#include <legion.h>


namespace LegionSolvers {


    template <typename ENTRY_T>
    class DenseVector {

        Legion::Context ctx;
        Legion::Runtime *rt;
        Legion::IndexSpace index_space;
        Legion::FieldID fid;
        Legion::FieldSpace field_space;
        Legion::LogicalRegion logical_region;

    public:

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            Legion::coord_t n
        ) : ctx(ctx), rt(rt),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<1>{0, n - 1}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {}

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            Legion::coord_t m, Legion::coord_t n
        ) : ctx(ctx), rt(rt),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<2>{{0, 0}, {m - 1, n - 1}}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {}

        explicit DenseVector(
            Legion::Context ctx, Legion::Runtime *rt,
            Legion::coord_t x, Legion::coord_t y, Legion::coord_t z
        ) : ctx(ctx), rt(rt),
            index_space(rt->create_index_space(ctx,
                Legion::Rect<3>{{0, 0, 0}, {x - 1, y - 1, z - 1}}
            )),
            fid(0),
            field_space(LegionSolvers::create_field_space(ctx, rt,
                {sizeof(ENTRY_T)}, {fid}
            )),
            logical_region(rt->create_logical_region(ctx,
                index_space, field_space
            )) {}

        ~DenseVector() {
            rt->destroy_logical_region(ctx, logical_region);
            rt->destroy_field_space   (ctx, field_space   );
            rt->destroy_index_space   (ctx, index_space   );
        }

        Legion::IndexSpace    get_index_space   () const { return index_space   ; }
        Legion::FieldSpace    get_field_space   () const { return field_space   ; }
        Legion::LogicalRegion get_logical_region() const { return logical_region; }
        Legion::FieldID       get_fid           () const { return fid           ; }

    }; // class DenseVector


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_DENSE_VECTOR_HPP
