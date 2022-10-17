#ifndef LEGION_SOLVERS_SCALAR_HPP_INCLUDED
#define LEGION_SOLVERS_SCALAR_HPP_INCLUDED

#include <legion.h>

#include "UtilityTasks.hpp"

namespace LegionSolvers {


template <typename T>
class Scalar {

    const Legion::Context ctx;
    Legion::Runtime *const rt;
    Legion::Future future;

  public:

    explicit Scalar(
        Legion::Context ctx, Legion::Runtime *rt, const Legion::Future &future
    )
        : ctx(ctx), rt(rt), future(future) {}

    explicit Scalar(Legion::Context ctx, Legion::Runtime *rt, const T &value)
        : ctx(ctx), rt(rt), future(Legion::Future::from_value(rt, value)) {}

    // NOTE: Copy constructor should not be explicit.
    Scalar(const Scalar &) = default;

    Scalar &operator=(const Scalar &rhs) {
        future = rhs.future;
        return *this; // No need to overwrite ctx or rt.
    }

    Legion::Future get_future() const { return future; }

    T get_value() const { return future.get_result<T>(); }

    Scalar operator+() const { return *this; }

    Scalar operator-() const {
        Legion::TaskLauncher launcher(
            NegateScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        return Scalar{ctx, rt, rt->execute_task(ctx, launcher)};
    }

    Scalar operator+(const Scalar &rhs) const {
        Legion::TaskLauncher launcher(
            AddScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        launcher.add_future(rhs.future);
        return Scalar{ctx, rt, rt->execute_task(ctx, launcher)};
    }

    Scalar operator-(const Scalar &rhs) const {
        Legion::TaskLauncher launcher(
            SubtractScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        launcher.add_future(rhs.future);
        return Scalar{ctx, rt, rt->execute_task(ctx, launcher)};
    }

    Scalar operator*(const Scalar &rhs) const {
        Legion::TaskLauncher launcher(
            MultiplyScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        launcher.add_future(rhs.future);
        return Scalar{ctx, rt, rt->execute_task(ctx, launcher)};
    }

    Scalar operator/(const Scalar &rhs) const {
        Legion::TaskLauncher launcher(
            DivideScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        launcher.add_future(rhs.future);
        return Scalar{ctx, rt, rt->execute_task(ctx, launcher)};
    }

    Legion::Future print() const {
        Legion::TaskLauncher launcher(
            PrintScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        return rt->execute_task(ctx, launcher);
    }

    Legion::Future print(Legion::Future dummy) const {
        Legion::TaskLauncher launcher(
            PrintScalarTask<T>::task_id, Legion::TaskArgument{nullptr, 0}
        );
        // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
        launcher.add_future(future);
        launcher.add_future(dummy);
        return rt->execute_task(ctx, launcher);
    }

}; // class Scalar


} // namespace LegionSolvers

#endif // LEGION_SOLVERS_SCALAR_HPP_INCLUDED
