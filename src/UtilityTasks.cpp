#include "UtilityTasks.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>


template <typename T>
T LegionSolvers::AdditionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() + b.get_result<T>();
}


template <typename T>
T LegionSolvers::SubtractionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() - b.get_result<T>();
}


template <typename T>
T LegionSolvers::NegationTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    return -x.get_result<T>();
}


template <typename T>
T LegionSolvers::MultiplicationTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() * b.get_result<T>();
}


template <typename T>
T LegionSolvers::DivisionTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 2);
    Legion::Future a = task->futures[0];
    Legion::Future b = task->futures[1];
    return a.get_result<T>() / b.get_result<T>();
}


template <typename T>
void LegionSolvers::AssertSmallTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert(task->futures.size() == 1);
    Legion::Future x = task->futures[0];
    assert(std::abs(x.get_result<T>()) <= std::numeric_limits<T>::epsilon());
}


template <typename T, int DIM>
void LegionSolvers::RandomFillTask<T, DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // RandomFillTask::announce_cpu(task->index_point, ctx, rt);

    assert(regions.size() == 1);
    const auto &region = regions[0];

    assert(task->regions.size() == 1);
    const auto &region_req = task->regions[0];

    assert(region_req.privilege_fields.size() == 1);
    const Legion::FieldID fid = *region_req.privilege_fields.begin();

    assert(task->arglen == 2 * sizeof(T));
    const T *arg_ptr = reinterpret_cast<const T *>(task->args);
    const T low = arg_ptr[0];
    const T high = arg_ptr[1];

    std::random_device rng{};
    std::uniform_real_distribution<T> entry_dist{low, high};

    const Legion::FieldAccessor<LEGION_WRITE_DISCARD, T, DIM>
    entry_writer{region, fid};

    for (Legion::PointInDomainIterator<DIM> iter{region}; iter(); ++iter) {
        entry_writer[*iter] = entry_dist(rng);
    }
}


template <typename T, int DIM>
void LegionSolvers::PrintVectorTask<T, DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    // PrintVectorTask::announce_cpu(task->index_point, ctx, rt);

    assert(regions.size() == 1);
    const auto &vector = regions[0];

    assert(task->regions.size() == 1);
    const auto &vector_req = task->regions[0];

    assert(vector_req.privilege_fields.size() == 1);
    const Legion::FieldID vector_fid =
        *vector_req.privilege_fields.begin();

    const Legion::FieldAccessor<LEGION_READ_ONLY, T, DIM>
    entry_reader{vector, vector_fid};

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
            std::cout << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    } else {
        const std::string name{reinterpret_cast<const char *>(task->args)};
        for (Legion::PointInDomainIterator<DIM> iter{vector}; iter(); ++iter) {
            std::cout << name << ' ' << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    }
    std::cout << std::flush;
}


// template __half LegionSolvers::AdditionTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::AdditionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::AdditionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template long double LegionSolvers::AdditionTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template __float128 LegionSolvers::AdditionTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

// template __half LegionSolvers::SubtractionTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::SubtractionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::SubtractionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template long double LegionSolvers::SubtractionTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template __float128 LegionSolvers::SubtractionTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

// template __half LegionSolvers::NegationTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::NegationTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::NegationTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template long double LegionSolvers::NegationTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template __float128 LegionSolvers::NegationTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

// template __half LegionSolvers::MultiplicationTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::MultiplicationTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::MultiplicationTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template long double LegionSolvers::MultiplicationTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template __float128 LegionSolvers::MultiplicationTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

// template __half LegionSolvers::DivisionTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template float LegionSolvers::DivisionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DivisionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template long double LegionSolvers::DivisionTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template __float128 LegionSolvers::DivisionTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

// template void LegionSolvers::AssertSmallTask<__half>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AssertSmallTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AssertSmallTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AssertSmallTask<long double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
// template void LegionSolvers::AssertSmallTask<__float128>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::RandomFillTask<float, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<long double, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<long double, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<long double, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::PrintVectorTask<float, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<long double, 1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<long double, 2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<long double, 3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
