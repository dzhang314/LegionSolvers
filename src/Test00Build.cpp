#include <iostream> // for std::cout, std::endl

#include <legion.h> // for LEGION_USE_CUDA, REALM_USE_KOKKOS

int main() {

#ifdef LEGION_USE_CUDA
    std::cout << "CUDA enabled" << std::endl;
#else
    std::cout << "CUDA disabled" << std::endl;
#endif

#ifdef REALM_USE_KOKKOS
    std::cout << "Kokkos enabled" << std::endl;
#else
    std::cout << "Kokkos disabled" << std::endl;
#endif

    return 0;
}
