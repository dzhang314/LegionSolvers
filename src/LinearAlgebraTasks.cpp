#include "LinearAlgebraTasks.hpp"

#include <cassert> // for assert
#include <cmath>   // for std::fma

#include "LegionUtilities.hpp" // for AffineReader, AffineWriter, ...
#include "LibraryOptions.hpp"  // for LEGION_SOLVERS_USE_*

using LegionSolvers::AxpyTask;
using LegionSolvers::DotTask;
using LegionSolvers::ScalTask;
using LegionSolvers::XpayTask;


template <typename ENTRY_T, int DIM, typename COORD_T>
void ScalTask<ENTRY_T, DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    assert(regions.size() == 1);
    const auto &x = regions[0];

    assert(task->regions.size() == 1);
    const auto &x_req = task->regions[0];

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> x_reader_writer(x, x_fid);

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    using RectIterator = Legion::RectInDomainIterator<DIM, COORD_T>;
    using PointIterator = Legion::PointInRectIterator<DIM, COORD_T>;

    for (RectIterator rect_iter(x_domain); rect_iter(); ++rect_iter) {
        const Legion::Rect<DIM, COORD_T> rect = *rect_iter;
        for (PointIterator point_iter(rect); point_iter(); ++point_iter) {
            const Legion::Point<DIM, COORD_T> point = *point_iter;
            x_reader_writer[point] = alpha * x_reader_writer[point];
        }
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void AxpyTask<ENTRY_T, DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    assert(regions.size() == 2);
    const auto &y = regions[0];
    const auto &x = regions[1];

    assert(task->regions.size() == 2);
    const auto &y_req = task->regions[0];
    const auto &x_req = task->regions[1];

    assert(y_req.privilege_fields.size() == 1);
    const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y_reader_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain =
        rt->get_index_space_domain(ctx, y_req.region.get_index_space());

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(y_domain == x_domain);

    using RectIterator = Legion::RectInDomainIterator<DIM, COORD_T>;
    using PointIterator = Legion::PointInRectIterator<DIM, COORD_T>;

    for (RectIterator rect_iter(x_domain); rect_iter(); ++rect_iter) {
        const Legion::Rect<DIM, COORD_T> rect = *rect_iter;
        for (PointIterator point_iter(rect); point_iter(); ++point_iter) {
            const Legion::Point<DIM, COORD_T> point = *point_iter;
            y_reader_writer[point] =
                std::fma(alpha, x_reader[point], y_reader_writer[point]);
        }
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
void XpayTask<ENTRY_T, DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    assert(regions.size() == 2);
    const auto &y = regions[0];
    const auto &x = regions[1];

    assert(task->regions.size() == 2);
    const auto &y_req = task->regions[0];
    const auto &x_req = task->regions[1];

    assert(y_req.privilege_fields.size() == 1);
    const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

    assert(x_req.privilege_fields.size() == 1);
    const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

    const ENTRY_T alpha = get_alpha<ENTRY_T>(task->futures);

    AffineReaderWriter<ENTRY_T, DIM, COORD_T> y_reader_writer{y, y_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> x_reader{x, x_fid};

    const Legion::Domain y_domain =
        rt->get_index_space_domain(ctx, y_req.region.get_index_space());

    const Legion::Domain x_domain =
        rt->get_index_space_domain(ctx, x_req.region.get_index_space());

    assert(y_domain == x_domain);

    using RectIterator = Legion::RectInDomainIterator<DIM, COORD_T>;
    using PointIterator = Legion::PointInRectIterator<DIM, COORD_T>;

    for (RectIterator rect_iter(x_domain); rect_iter(); ++rect_iter) {
        const Legion::Rect<DIM, COORD_T> rect = *rect_iter;
        for (PointIterator point_iter(rect); point_iter(); ++point_iter) {
            const Legion::Point<DIM, COORD_T> point = *point_iter;
            y_reader_writer[point] =
                std::fma(alpha, y_reader_writer[point], x_reader[point]);
        }
    }
}


template <typename ENTRY_T, int DIM, typename COORD_T>
ENTRY_T DotTask<ENTRY_T, DIM, COORD_T>::task_body(LEGION_SOLVERS_TASK_ARGS) {

    assert(regions.size() == 2);
    const auto &v = regions[0];
    const auto &w = regions[1];

    assert(task->regions.size() == 2);
    const auto &v_req = task->regions[0];
    const auto &w_req = task->regions[1];

    assert(v_req.privilege_fields.size() == 1);
    const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

    assert(w_req.privilege_fields.size() == 1);
    const Legion::FieldID w_fid = *w_req.privilege_fields.begin();

    AffineReader<ENTRY_T, DIM, COORD_T> v_reader{v, v_fid};
    AffineReader<ENTRY_T, DIM, COORD_T> w_reader{w, w_fid};

    const Legion::Domain v_domain =
        rt->get_index_space_domain(ctx, v_req.region.get_index_space());

    const Legion::Domain w_domain =
        rt->get_index_space_domain(ctx, w_req.region.get_index_space());

    assert(v_domain == w_domain);

    using RectIterator = Legion::RectInDomainIterator<DIM, COORD_T>;
    using PointIterator = Legion::PointInRectIterator<DIM, COORD_T>;

    ENTRY_T result = static_cast<ENTRY_T>(0);
    for (RectIterator rect_iter(v_domain); rect_iter(); ++rect_iter) {
        const Legion::Rect<DIM, COORD_T> rect = *rect_iter;
        for (PointIterator point_iter(rect); point_iter(); ++point_iter) {
            const Legion::Point<DIM, COORD_T> point = *point_iter;
            result += v_reader[point] * w_reader[point];
        }
    }
    return result;
}


// clang-format off
#ifdef LEGION_SOLVERS_USE_F32
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template float DotTask<float, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F32
#ifdef LEGION_SOLVERS_USE_F64
    #ifdef LEGION_SOLVERS_USE_S32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, int>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S32_INDICES
    #ifdef LEGION_SOLVERS_USE_U32_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, unsigned>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_U32_INDICES
    #ifdef LEGION_SOLVERS_USE_S64_INDICES
        #if LEGION_SOLVERS_MAX_DIM >= 1
            template void ScalTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 1, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 1
        #if LEGION_SOLVERS_MAX_DIM >= 2
            template void ScalTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 2, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 2
        #if LEGION_SOLVERS_MAX_DIM >= 3
            template void ScalTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void AxpyTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template void XpayTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
            template double DotTask<double, 3, long long>::task_body(LEGION_SOLVERS_TASK_ARGS);
        #endif // LEGION_SOLVERS_MAX_DIM >= 3
    #endif // LEGION_SOLVERS_USE_S64_INDICES
#endif // LEGION_SOLVERS_USE_F64
// clang-format on
