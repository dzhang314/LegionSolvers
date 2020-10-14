#include <vector>

#include <Kokkos_Core.hpp>
#include <legion.h>

namespace LegionSolvers {

  template <typename TFloat>
  using AccessorRO =
      Legion::FieldAccessor<READ_ONLY, TFloat, 1, Legion::coord_t,
                            Realm::AffineAccessor<TFloat, 1, Legion::coord_t>>;

  template <typename TFloat>
  using AccessorRW =
      Legion::FieldAccessor<READ_WRITE, TFloat, 1, Legion::coord_t,
                            Realm::AffineAccessor<TFloat, 1, Legion::coord_t>>;

  template <typename KokkosExecutionSpace, typename TFloat>
  Kokkos::View<TFloat *, typename KokkosExecutionSpace::memory_space>
  get_view(const Legion::PhysicalRegion &region, Legion::FieldID field_id) {
    AccessorRW<TFloat> acc{region, field_id};
    return acc.accessor;
  }

  template <typename KokkosExecutionSpace, typename TFloat>
  Kokkos::View<const TFloat *, typename KokkosExecutionSpace::memory_space>
  get_const_view(const Legion::PhysicalRegion &region,
                 Legion::FieldID field_id) {
    AccessorRO<TFloat> acc{region, field_id};
    return acc.accessor;
  }

  template <typename KokkosExecutionSpace>
  Kokkos::RangePolicy<KokkosExecutionSpace>
  get_range(const Legion::Task *task, const Legion::Context &ctx,
            Legion::Runtime *runtime) {
    Legion::Rect<1> subspace = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());
    Kokkos::RangePolicy<KokkosExecutionSpace, Legion::coord_t> range{
        runtime->get_executing_processor(ctx).kokkos_work_space(),
        subspace.lo.x, subspace.hi.x + 1};
    return range;
  }

  template <typename KokkosExecutionSpace, typename TFloat>
  struct AxpyTask {
    static void task_body(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context ctx, Legion::Runtime *runtime) {
      assert(regions.size() == 2);
      const auto &y_reg = regions[0];
      const auto &x_reg = regions[1];

      assert(task->regions.size() == 2);
      const auto &y_req = task->regions[0];
      const auto &x_req = task->regions[1];

      assert(y_req.privilege_fields.size() == 1);
      const Legion::FieldID y_fid = *y_req.privilege_fields.begin();

      assert(x_req.privilege_fields.size() == 1);
      const Legion::FieldID x_fid = *x_req.privilege_fields.begin();

      assert(task->arglen == 0);
      assert(task->futures.size() == 1);
      const TFloat alpha = task->futures[0].get_result<TFloat>();

      auto y = get_view<KokkosExecutionSpace, TFloat>(y_reg, y_fid);
      auto x = get_const_view<KokkosExecutionSpace, TFloat>(x_reg, x_fid);
      Kokkos::parallel_for(
          get_range<KokkosExecutionSpace>(task, ctx, runtime),
          KOKKOS_LAMBDA(int i) { y(i) += alpha * x(i); });
    }
  }; // struct AxpyTask

} // namespace LegionSolvers

template <typename TFloat>
void negative_second_derivative_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
    Legion::Runtime *runtime) {

  assert(regions.size() == 2);
  const auto &output = regions[0];
  const auto &input = regions[1];

  assert(task->regions.size() == 2);
  const auto &output_req = task->regions[0];
  const auto &input_req = task->regions[1];

  assert(output_req.privilege_fields.size() == 1);
  const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

  assert(input_req.privilege_fields.size() == 1);
  const Legion::FieldID input_fid = *input_req.privilege_fields.begin();

  assert(task->arglen == sizeof(int));
  const int num_elements = *reinterpret_cast<int *>(task->args);

  const Legion::FieldAccessor<WRITE_DISCARD, TFloat, 1> writer{output,
                                                               output_fid};
  const Legion::FieldAccessor<READ_ONLY, TFloat, 1> reader{input, input_fid};

  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, output_req.region.get_index_space());
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    const TFloat l = (iter[0] == 0) ? 0.0 : reader[*iter - 1];
    const TFloat c = reader[*iter];
    const TFloat r = (iter[0] == num_elements - 1) ? 0.0 : reader[*iter + 1];
    writer[*iter] = (c - l) + (c - r);
  }
}

template <typename TFloat>
void zero_fill_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime) {
  assert(regions.size() == 1);
  const auto &output = regions[0];

  assert(task->regions.size() == 1);
  const auto &output_req = task->regions[0];

  assert(output_req.privilege_fields.size() == 1);
  const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

  assert(task->arglen == 0);

  const Legion::FieldAccessor<WRITE_DISCARD, TFloat, 1> writer{output,
                                                               output_fid};
  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, output_req.region.get_index_space());
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    writer[*iter] = static_cast<TFloat>(0);
  }
}

template <typename TFloat>
struct BoundaryFillTaskArgs {
  double a;
  double b;
  int num_elements;
};

template <typename TFloat>
void boundary_fill_task(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx, Legion::Runtime *runtime) {
  assert(regions.size() == 1);
  const auto &output = regions[0];

  assert(task->regions.size() == 1);
  const auto &output_req = task->regions[0];

  assert(output_req.privilege_fields.size() == 1);
  const Legion::FieldID output_fid = *output_req.privilege_fields.begin();

  assert(task->arglen == sizeof(BoundaryFillTaskArgs<TFloat>));
  const BoundaryFillTaskArgs<TFloat> args =
      *reinterpret_cast<BoundaryFillTaskArgs<TFloat> *>(task->args);

  const Legion::FieldAccessor<WRITE_DISCARD, TFloat, 1> writer{output,
                                                               output_fid};
  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, output_req.region.get_index_space());
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    if (iter[0] == 0) {
      writer[*iter] = args.a;
    } else if (iter[0] == args.num_elements - 1) {
      writer[*iter] = args.b;
    } else {
      writer[*iter] = static_cast<TFloat>(0);
    }
  }
}

template <typename TFloat>
TFloat dot_product_task(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx, Legion::Runtime *runtime) {
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

  assert(task->arglen == 0);

  const Legion::FieldAccessor<READ_ONLY, TFloat, 1> v_reader{v, v_fid};
  const Legion::FieldAccessor<READ_ONLY, TFloat, 1> w_reader{w, w_fid};
  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, v_req.region.get_index_space());
  TFloat result = static_cast<TFloat>(0);
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    result += v_reader[*iter] * w_reader[*iter];
  }
  return result;
}

template <typename TFloat>
TFloat norm_squared_task(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *runtime) {
  assert(regions.size() == 1);
  const auto &v = regions[0];

  assert(task->regions.size() == 1);
  const auto &v_req = task->regions[0];

  assert(v_req.privilege_fields.size() == 1);
  const Legion::FieldID v_fid = *v_req.privilege_fields.begin();

  assert(task->arglen == 0);

  const Legion::FieldAccessor<READ_ONLY, TFloat, 1> v_reader{v, v_fid};
  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, v_req.region.get_index_space());
  TFloat result = static_cast<TFloat>(0);
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    const TFloat temp = v_reader[*iter];
    result += temp * temp;
  }
  return result;
}

template <typename TFloat>
void aypx_task(const Legion::Task *task,
               const std::vector<Legion::PhysicalRegion> &regions,
               Legion::Context ctx, Legion::Runtime *runtime) {
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

  assert(task->arglen == 0);
  assert(task->futures.size() == 1);
  const TFloat alpha = task->futures[0].get_result<TFloat>();

  const Legion::FieldAccessor<READ_WRITE, TFloat, 1> y_writer{y, y_fid};
  const Legion::FieldAccessor<READ_ONLY, TFloat, 1> x_reader{x, x_fid};
  Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, y_req.region.get_index_space());
  for (Legion::PointInRectIterator<1> iter{rect}; iter(); ++iter) {
    y_writer[*iter] = alpha * y_writer[*iter] + x_reader[*iter];
  }
}

template <typename TFloat>
TFloat division_task(const Legion::Task *task,
                     const std::vector<Legion::PhysicalRegion> &regions,
                     Legion::Context, Legion::Runtime *) {
  assert(regions.size() == 0);
  assert(task->regions.size() == 0);
  assert(task->arglen == 0);
  assert(task->futures.size() == 2);
  Legion::Future numerator = task->futures[0];
  Legion::Future denominator = task->futures[1];
  return numerator.get_result<TFloat>() / denominator.get_result<TFloat>();
}

template <typename TFloat>
TFloat negation_task(const Legion::Task *task,
                     const std::vector<Legion::PhysicalRegion> &regions,
                     Legion::Context, Legion::Runtime *) {
  assert(regions.size() == 0);
  assert(task->regions.size() == 0);
  assert(task->arglen == 0);
  assert(task->futures.size() == 1);
  Legion::Future x = task->futures[0];
  return -x.get_result<TFloat>();
}
