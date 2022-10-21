#pragma once

#include "CudaLibs.hpp"

namespace LegionSolvers {

// Template dispatch for AXPY.
template <typename ENTRY_T>
void cublasAXPY(cublasHandle_t handle,
                int n,
                const ENTRY_T* alpha,
                const ENTRY_T* x,
                int incx,
                ENTRY_T* y,
                int incy) { assert(false); }

template <>
void cublasAXPY<float>(cublasHandle_t handle,
                       int n,
                       const float* alpha,
                       const float* x,
                       int incx,
                       float* y,
                       int incy) {
  CHECK_CUBLAS(cublasSaxpy(
      handle,
      n,
      alpha,
      x,
      incx,
      y,
      incy
  ));
}

template <>
void cublasAXPY<double>(cublasHandle_t handle,
                        int n,
                        const double* alpha,
                        const double* x,
                        int incx,
                        double* y,
                        int incy) {
  CHECK_CUBLAS(cublasDaxpy(
      handle,
      n,
      alpha,
      x,
      incx,
      y,
      incy
  ));
}


} // namespace LegionSolvers