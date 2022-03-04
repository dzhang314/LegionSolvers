#include "UtilityTasks.hpp"

#include <algorithm>  // for std::generate
#include <cmath>      // for std::abs
#include <functional> // for std::ref
#include <iostream>   // for std::cout
#include <iterator>   // for std::begin, std::end
#include <limits>     // for std::numeric_limits
#include <random>     // for std::mt19937, std::random_device, etc.

#include "LegionUtilities.hpp"


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
int LegionSolvers::PrintScalarTask<T>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    assert((task->futures.size() == 1) || (task->futures.size() == 2));
    Legion::Future x = task->futures[0];
    std::cout << x.get_result<T>() << std::endl;
    return 0;
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


template <typename T, int DIM, typename COORD_T>
void LegionSolvers::RandomFillTask<T, DIM, COORD_T>::task_body(
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

    std::random_device system_entropy{};
    unsigned int seed_data[624];
    std::generate(std::begin(seed_data), std::end(seed_data),
                  std::ref(system_entropy));
    std::seed_seq seed_seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 rng{seed_seq};

    std::uniform_real_distribution<T> entry_dist{low, high};

    const AffineWriter<T, DIM, COORD_T> entry_writer{region, fid};

    for (Legion::PointInDomainIterator<DIM, COORD_T> iter{region}; iter(); ++iter) {
        entry_writer[*iter] = entry_dist(rng);
    }
}


template <typename T, int DIM, typename COORD_T>
void LegionSolvers::PrintVectorTask<T, DIM, COORD_T>::task_body(
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

    const AffineReader<T, DIM, COORD_T> entry_reader{vector, vector_fid};

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM, COORD_T> iter{vector}; iter(); ++iter) {
            std::cout << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    } else {
        const std::string name{reinterpret_cast<const char *>(task->args)};
        for (Legion::PointInDomainIterator<DIM, COORD_T> iter{vector}; iter(); ++iter) {
            std::cout << name << ' ' << task->index_point << ' '
                      << *iter << ": " << entry_reader[*iter] << '\n';
        }
    }
    std::cout << std::flush;
}


template <int DIM>
void LegionSolvers::PrintIndexTask<DIM>::task_body(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx, Legion::Runtime *rt
) {
    const Legion::DomainPoint index_point = task->index_point;

    // PrintIndexTask::announce_cpu(index_point, ctx, rt);

    assert(regions.size() == 1);
    const auto &dummy = regions[0];

    assert(task->regions.size() == 1);
    const auto &dummy_req = task->regions[0];

    if (task->arglen == 0) {
        for (Legion::PointInDomainIterator<DIM> iter{dummy}; iter(); ++iter) {
            std::cout << index_point << ' ' << *iter << '\n';
        }
    } else {
        const std::string name{reinterpret_cast<const char *>(task->args)};
        for (Legion::PointInDomainIterator<DIM> iter{dummy}; iter(); ++iter) {
            std::cout << name << ' ' << index_point << ' ' << *iter << '\n';
        }
    }
    std::cout << std::flush;
}


template float LegionSolvers::AdditionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::AdditionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template float LegionSolvers::SubtractionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::SubtractionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template float LegionSolvers::NegationTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::NegationTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template float LegionSolvers::MultiplicationTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::MultiplicationTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template float LegionSolvers::DivisionTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template double LegionSolvers::DivisionTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template int LegionSolvers::PrintScalarTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template int LegionSolvers::PrintScalarTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::AssertSmallTask<float>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::AssertSmallTask<double>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::RandomFillTask<float , 1, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 1, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 1, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 2, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 2, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 2, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 3, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 3, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<float , 3, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 1, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 1, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 1, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 2, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 2, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 2, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 3, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 3, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::RandomFillTask<double, 3, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::PrintVectorTask<float , 1, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 1, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 1, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 2, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 2, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 2, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 3, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 3, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<float , 3, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 1, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 1, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 1, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 2, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 2, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 2, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 3, int      >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 3, unsigned >::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintVectorTask<double, 3, long long>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);

template void LegionSolvers::PrintIndexTask<1>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintIndexTask<2>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
template void LegionSolvers::PrintIndexTask<3>::task_body(const Legion::Task *, const std::vector<Legion::PhysicalRegion> &, Legion::Context, Legion::Runtime *);
