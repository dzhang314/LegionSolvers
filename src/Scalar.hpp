#ifndef LEGION_SOLVERS_SCALAR_HPP
#define LEGION_SOLVERS_SCALAR_HPP

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
            const Legion::Future &future,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx), rt(rt), future(future) {}

        explicit Scalar(
            const T &value,
            Legion::Context ctx, Legion::Runtime *rt
        ) : ctx(ctx), rt(rt), future(Legion::Future::from_value(rt, value)) {}

        Legion::Future get_future() const { return future; }

        T get_value() const { return future.get_result<T>(); }

        Scalar operator+(const Scalar &rhs) const {
            Legion::TaskLauncher launcher{
                AdditionTask<T>::task_id,
                Legion::TaskArgument{nullptr, 0}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_future(future);
            launcher.add_future(rhs.future);
            return Scalar{rt->execute_task(ctx, launcher), ctx, rt};
        }

        Scalar operator-(const Scalar &rhs) const {
            Legion::TaskLauncher launcher{
                SubtractionTask<T>::task_id,
                Legion::TaskArgument{nullptr, 0}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_future(future);
            launcher.add_future(rhs.future);
            return Scalar{rt->execute_task(ctx, launcher), ctx, rt};
        }

        Scalar operator-() const {
            Legion::TaskLauncher launcher{
                NegationTask<T>::task_id,
                Legion::TaskArgument{nullptr, 0}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_future(future);
            return Scalar{rt->execute_task(ctx, launcher), ctx, rt};
        }

        Scalar operator*(const Scalar &rhs) const {
            Legion::TaskLauncher launcher{
                MultiplicationTask<T>::task_id,
                Legion::TaskArgument{nullptr, 0}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_future(future);
            launcher.add_future(rhs.future);
            return Scalar{rt->execute_task(ctx, launcher), ctx, rt};
        }

        Scalar operator/(const Scalar &rhs) const {
            Legion::TaskLauncher launcher{
                DivisionTask<T>::task_id,
                Legion::TaskArgument{nullptr, 0}
            };
            // launcher.map_id = LEGION_SOLVERS_MAPPER_ID;
            launcher.add_future(future);
            launcher.add_future(rhs.future);
            return Scalar{rt->execute_task(ctx, launcher), ctx, rt};
        }

    }; // class Scalar


} // namespace LegionSolvers


#endif // LEGION_SOLVERS_SCALAR_HPP
