#include <Kokkos_Core.hpp>
#include <legion.h>

namespace TaskUtils {

  template <template <typename, typename...> class KokkosTaskTemplate,
            typename... TaskTemplateArgs>
  void preregister_kokkos_task(Legion::TaskID task_id, const char *name) {
#ifdef KOKKOS_ENABLE_SERIAL
    {
      Legion::TaskVariantRegistrar registrar{task_id, name};
      registrar.add_constraint(
          Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
      Legion::Runtime::preregister_task_variant<
          KokkosTaskTemplate<Kokkos::Serial, TaskTemplateArgs...>::task_body>(
          registrar, name);
    }
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    {
      Legion::TaskVariantRegistrar registrar{task_id, name};
      registrar.add_constraint(Legion::ProcessorConstraint{
#ifdef REALM_USE_OPENMP
          Legion::Processor::OMP_PROC
#else
          Legion::Processor::LOC_PROC
#endif // REALM_USE_OPENMP
      });
      Legion::Runtime::preregister_task_variant<
          KokkosTaskTemplate<Kokkos::OpenMP, TaskTemplateArgs...>::task_body>(
          registrar, name);
    }
#endif // KOKKOS_ENABLE_OPENMP
  }

  template <typename ReturnType,
            template <typename, typename...> class KokkosTaskTemplate,
            typename... TaskTemplateArgs>
  void preregister_kokkos_task(Legion::TaskID task_id, const char *name) {
#ifdef KOKKOS_ENABLE_SERIAL
    {
      Legion::TaskVariantRegistrar registrar{task_id, name};
      registrar.add_constraint(
          Legion::ProcessorConstraint{Legion::Processor::LOC_PROC});
      Legion::Runtime::preregister_task_variant<
          ReturnType,
          KokkosTaskTemplate<Kokkos::Serial, TaskTemplateArgs...>::task_body>(
          registrar, name);
    }
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    {
      Legion::TaskVariantRegistrar registrar{task_id, name};
      registrar.add_constraint(Legion::ProcessorConstraint{
#ifdef REALM_USE_OPENMP
          Legion::Processor::OMP_PROC
#else
          Legion::Processor::LOC_PROC
#endif // REALM_USE_OPENMP
      });
      Legion::Runtime::preregister_task_variant<
          ReturnType,
          KokkosTaskTemplate<Kokkos::OpenMP, TaskTemplateArgs...>::task_body>(
          registrar, name);
    }
#endif // KOKKOS_ENABLE_OPENMP
  }

}; // namespace TaskUtils
