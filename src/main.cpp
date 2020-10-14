#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <legion.h>

#include <Kokkos_Core.hpp>

#include "LegionSolvers.hpp"
#include "TaskUtils.hpp"

enum TaskID : Legion::TaskID {
  TOP_LEVEL_TASK_ID,
  ZERO_FILL_TASK_ID,
  BOUNDARY_FILL_TASK_ID,
  DOT_PRODUCT_TASK_ID,
  NEGATIVE_SECOND_DERIVATIVE_TASK_ID,
  AXPY_TASK_ID,
  AYPX_TASK_ID,
  DIVISION_TASK_ID,
  NEGATION_TASK_ID,
  NORM_SQUARED_TASK_ID,
  CHECK_TASK_ID,
  PRINT_TASK_ID,
};

enum FieldIDs {
  FID_P,
  FID_Q,
  FID_R,
  FID_X,
};

void top_level_task(const Legion::Task *,
                    const std::vector<Legion::PhysicalRegion> &,
                    Legion::Context ctx, Legion::Runtime *rt) {
  const int num_elements = 16;
  const int num_subregions = 4;

  Legion::Rect<1> elem_rect{0, num_elements - 1};
  Legion::IndexSpaceT<1> index_space = rt->create_index_space(ctx, elem_rect);
  Legion::FieldSpace field_space = rt->create_field_space(ctx);
  {
    Legion::FieldAllocator allocator =
        rt->create_field_allocator(ctx, field_space);
    allocator.allocate_field(sizeof(double), FID_P);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_R);
    allocator.allocate_field(sizeof(double), FID_X);
  }
  Legion::LogicalRegion stencil_lr =
      rt->create_logical_region(ctx, index_space, field_space);

  Legion::Rect<1> color_bounds(0, num_subregions - 1);
  Legion::IndexSpaceT<1> color_is = rt->create_index_space(ctx, color_bounds);
  Legion::IndexPartition disjoint_ip =
      rt->create_equal_partition(ctx, index_space, color_is);
  const int block_size = (num_elements + num_subregions - 1) / num_subregions;
  Legion::Transform<1, 1> transform;
  transform[0][0] = block_size;
  Legion::Rect<1> extent(-1, block_size);
  Legion::IndexPartition ghost_ip = rt->create_partition_by_restriction(
      ctx, index_space, color_is, transform, extent);

  Legion::LogicalPartition disjoint_partition =
      rt->get_logical_partition(ctx, stencil_lr, disjoint_ip);
  Legion::LogicalPartition ghost_partition =
      rt->get_logical_partition(ctx, stencil_lr, ghost_ip);

  Legion::ArgumentMap arg_map;

  const BoundaryFillTaskArgs<double> boundary_values = {-1.0, 2.0,
                                                        num_elements};

  { // mov x, 0
    Legion::IndexLauncher init_launcher{
        ZERO_FILL_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0}, arg_map};
    init_launcher.add_region_requirement(Legion::RegionRequirement{
        disjoint_partition, 0, WRITE_DISCARD, EXCLUSIVE, stencil_lr});
    init_launcher.add_field(0, FID_X);
    rt->execute_index_space(ctx, init_launcher);
  }

  { // mov r, b
    Legion::IndexLauncher init_launcher{
        BOUNDARY_FILL_TASK_ID, color_is,
        Legion::TaskArgument{&boundary_values,
                             sizeof(decltype(boundary_values))},
        arg_map};
    init_launcher.add_region_requirement(Legion::RegionRequirement{
        disjoint_partition, 0, WRITE_DISCARD, EXCLUSIVE, stencil_lr});
    init_launcher.add_field(0, FID_R);
    rt->execute_index_space(ctx, init_launcher);
  }

  { // mov p, b
    Legion::IndexLauncher init_launcher{
        BOUNDARY_FILL_TASK_ID, color_is,
        Legion::TaskArgument{&boundary_values,
                             sizeof(decltype(boundary_values))},
        arg_map};
    init_launcher.add_region_requirement(Legion::RegionRequirement{
        disjoint_partition, 0, WRITE_DISCARD, EXCLUSIVE, stencil_lr});
    init_launcher.add_field(0, FID_P);
    rt->execute_index_space(ctx, init_launcher);
  }

  // dot r_norm2, r, r
  Legion::TaskLauncher dot_product_launcher_1{DOT_PRODUCT_TASK_ID,
                                              Legion::TaskArgument{nullptr, 0}};
  dot_product_launcher_1.add_region_requirement(
      Legion::RegionRequirement{stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr});
  dot_product_launcher_1.add_field(0, FID_R);
  dot_product_launcher_1.add_region_requirement(
      Legion::RegionRequirement{stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr});
  dot_product_launcher_1.add_field(1, FID_R);
  Legion::Future r_norm2 = rt->execute_task(ctx, dot_product_launcher_1);

  for (int i = 0; i < 16; ++i) {

    { // matmul q, A, p
      Legion::IndexLauncher stencil_launcher{
          NEGATIVE_SECOND_DERIVATIVE_TASK_ID, color_is,
          Legion::TaskArgument{&num_elements, sizeof(num_elements)}, arg_map};
      stencil_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, WRITE_DISCARD, EXCLUSIVE, stencil_lr});
      stencil_launcher.add_field(0, FID_Q);
      stencil_launcher.add_region_requirement(Legion::RegionRequirement{
          ghost_partition, 0, READ_ONLY, EXCLUSIVE, stencil_lr});
      stencil_launcher.add_field(1, FID_P);
      rt->execute_index_space(ctx, stencil_launcher);
    }

    // { // verify matmul
    //     Legion::TaskLauncher check_launcher(
    //         CHECK_TASK_ID,
    //         Legion::TaskArgument(&num_elements, sizeof(num_elements)));
    //     check_launcher.add_region_requirement(Legion::RegionRequirement(
    //         stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
    //     check_launcher.add_field(0, FID_P);
    //     check_launcher.add_region_requirement(Legion::RegionRequirement(
    //         stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
    //     check_launcher.add_field(1, FID_Q);
    //     rt->execute_task(ctx, check_launcher);
    // }

    // dot p_norm, p, q
    Legion::TaskLauncher dot_product_launcher_2{
        DOT_PRODUCT_TASK_ID, Legion::TaskArgument{nullptr, 0}};
    dot_product_launcher_2.add_region_requirement(Legion::RegionRequirement{
        stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr});
    dot_product_launcher_2.add_field(0, FID_P);
    dot_product_launcher_2.add_region_requirement(Legion::RegionRequirement{
        stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr});
    dot_product_launcher_2.add_field(1, FID_Q);
    Legion::Future p_norm = rt->execute_task(ctx, dot_product_launcher_2);

    // div alpha, r_norm2, p_norm
    Legion::TaskLauncher division_launcher_1{DIVISION_TASK_ID,
                                             Legion::TaskArgument{nullptr, 0}};
    division_launcher_1.add_future(r_norm2);
    division_launcher_1.add_future(p_norm);
    Legion::Future alpha = rt->execute_task(ctx, division_launcher_1);

    { // axpy x, +alpha, p
      Legion::IndexLauncher axpy_launcher{
          AXPY_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0}, arg_map};
      axpy_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_WRITE, EXCLUSIVE, stencil_lr});
      axpy_launcher.add_field(0, FID_X);
      axpy_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_ONLY, EXCLUSIVE, stencil_lr});
      axpy_launcher.add_field(1, FID_P);
      axpy_launcher.add_future(alpha);
      rt->execute_index_space(ctx, axpy_launcher);
    }

    // axpy r, -alpha, q
    Legion::TaskLauncher negation_launcher{NEGATION_TASK_ID,
                                           Legion::TaskArgument{nullptr, 0}};
    negation_launcher.add_future(alpha);
    Legion::Future neg_alpha = rt->execute_task(ctx, negation_launcher);
    {
      Legion::IndexLauncher axpy_launcher{
          AXPY_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0}, arg_map};
      axpy_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_WRITE, EXCLUSIVE, stencil_lr});
      axpy_launcher.add_field(0, FID_R);
      axpy_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_ONLY, EXCLUSIVE, stencil_lr});
      axpy_launcher.add_field(1, FID_Q);
      axpy_launcher.add_future(neg_alpha);
      rt->execute_index_space(ctx, axpy_launcher);
    }

    // dot r_norm2_new, r, r
    Legion::TaskLauncher norm_squared_launcher{
        NORM_SQUARED_TASK_ID, Legion::TaskArgument{nullptr, 0}};
    norm_squared_launcher.add_region_requirement(Legion::RegionRequirement{
        stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr});
    norm_squared_launcher.add_field(0, FID_R);
    Legion::Future r_norm2_new = rt->execute_task(ctx, norm_squared_launcher);

    // div beta, r_norm2_new, r_norm2
    Legion::TaskLauncher division_launcher_2{DIVISION_TASK_ID,
                                             Legion::TaskArgument{nullptr, 0}};
    division_launcher_2.add_future(r_norm2_new);
    division_launcher_2.add_future(r_norm2);
    Legion::Future beta = rt->execute_task(ctx, division_launcher_2);

    // mov r_norm2, r_norm2_new
    r_norm2 = r_norm2_new;

    { // aypx p, +beta, r
      Legion::IndexLauncher aypx_launcher{
          AYPX_TASK_ID, color_is, Legion::TaskArgument{nullptr, 0}, arg_map};
      aypx_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_WRITE, EXCLUSIVE, stencil_lr});
      aypx_launcher.add_field(0, FID_P);
      aypx_launcher.add_region_requirement(Legion::RegionRequirement{
          disjoint_partition, 0, READ_ONLY, EXCLUSIVE, stencil_lr});
      aypx_launcher.add_field(1, FID_R);
      aypx_launcher.add_future(beta);
      rt->execute_index_space(ctx, aypx_launcher);
    }
  }

  { // print x
    Legion::TaskLauncher print_launcher(PRINT_TASK_ID,
                                        Legion::TaskArgument{nullptr, 0});
    print_launcher.add_region_requirement(Legion::RegionRequirement(
        stencil_lr, READ_WRITE, EXCLUSIVE, stencil_lr));
    print_launcher.add_field(0, FID_X);
    rt->execute_task(ctx, print_launcher);
  }

  rt->destroy_logical_region(ctx, stencil_lr);
  rt->destroy_field_space(ctx, field_space);
  rt->destroy_index_space(ctx, index_space);
}

void check_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  const int max_elements = *((const int *)task->args);

  Legion::FieldID src_fid = *(task->regions[0].privilege_fields.begin());
  Legion::FieldID dst_fid = *(task->regions[1].privilege_fields.begin());

  const Legion::FieldAccessor<READ_ONLY, double, 1> src_acc(regions[0],
                                                            src_fid);
  const Legion::FieldAccessor<READ_ONLY, double, 1> dst_acc(regions[1],
                                                            dst_fid);

  Legion::Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  bool all_passed = true;
  for (Legion::PointInRectIterator<1> iter(rect); iter(); iter++) {
    const double l = (iter[0] == 0) ? 0.0 : src_acc[*iter - 1];
    const double c = src_acc[*iter];
    const double r = (iter[0] == max_elements - 1) ? 0.0 : src_acc[*iter + 1];
    const double expected = (c - l) + (c - r);
    const double received = dst_acc[*iter];
    if (expected != received) all_passed = false;
  }
  if (all_passed) printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

void print_task(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions,
                Legion::Context ctx, Legion::Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == 0);

  Legion::FieldID src_fid = *(task->regions[0].privilege_fields.begin());
  const Legion::FieldAccessor<READ_WRITE, double, 1> src_acc{regions[0],
                                                             src_fid};

  Legion::Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  for (Legion::PointInRectIterator<1> iter(rect); iter(); iter++) {
    std::cout << iter[0] << ": " << src_acc[*iter] << std::endl;
  }
}

template <void (*TaskPtr)(const Legion::Task *,
                          const std::vector<Legion::PhysicalRegion> &,
                          Legion::Context, Legion::Runtime *)>
void register_cpu_task(Legion::TaskID task_id, const char *task_name) {
  Legion::TaskVariantRegistrar registrar(task_id, task_name);
  registrar.add_constraint(
      Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
  Legion::Runtime::preregister_task_variant<TaskPtr>(registrar, task_name);
}

template <typename ReturnType,
          ReturnType (*TaskPtr)(const Legion::Task *,
                                const std::vector<Legion::PhysicalRegion> &,
                                Legion::Context, Legion::Runtime *)>
void register_cpu_task(Legion::TaskID task_id, const char *task_name) {
  Legion::TaskVariantRegistrar registrar(task_id, task_name);
  registrar.add_constraint(
      Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
  Legion::Runtime::preregister_task_variant<ReturnType, TaskPtr>(registrar,
                                                                 task_name);
}

int main(int argc, char **argv) {
  Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  register_cpu_task<top_level_task>(TOP_LEVEL_TASK_ID, "top_level");
  register_cpu_task<zero_fill_task<double>>(ZERO_FILL_TASK_ID, "zero_fill");
  register_cpu_task<boundary_fill_task<double>>(BOUNDARY_FILL_TASK_ID,
                                                "boundary_fill");
  register_cpu_task<double, dot_product_task<double>>(DOT_PRODUCT_TASK_ID,
                                                      "dot_product");
  TaskUtils::preregister_kokkos_task<LegionSolvers::AxpyTask, double>(
      AXPY_TASK_ID, "axpy");
  register_cpu_task<aypx_task<double>>(AYPX_TASK_ID, "aypx");
  register_cpu_task<negative_second_derivative_task<double>>(
      NEGATIVE_SECOND_DERIVATIVE_TASK_ID, "negative_second_derivative");
  register_cpu_task<check_task>(CHECK_TASK_ID, "check");
  register_cpu_task<print_task>(PRINT_TASK_ID, "print");
  register_cpu_task<double, division_task<double>>(DIVISION_TASK_ID,
                                                   "division");
  register_cpu_task<double, negation_task<double>>(NEGATION_TASK_ID,
                                                   "negation");
  register_cpu_task<double, norm_squared_task<double>>(NORM_SQUARED_TASK_ID,
                                                       "norm_squared");
  return Legion::Runtime::start(argc, argv);
}
