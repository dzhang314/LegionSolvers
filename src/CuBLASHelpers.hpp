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

// Template dispatch for DOT.
template <typename ENTRY_T>
void cublasDOT(cublasHandle_t handle,
               int n,
               const ENTRY_T* x,
               int incx,
               const ENTRY_T* y,
               int incy,
               ENTRY_T* result) { assert(false); }

template <>
void cublasDOT<float>(cublasHandle_t handle,
                      int n,
                      const float* x,
                      int incx,
                      const float* y,
                      int incy,
                      float* result) {
  CHECK_CUBLAS(cublasSdot(
    handle,
    n,
    x,
    incx,
    y,
    incy,
    result
  ));
}

template <>
void cublasDOT<double>(cublasHandle_t handle,
                       int n,
                       const double* x,
                       int incx,
                       const double* y,
                       int incy,
                       double* result) {
  CHECK_CUBLAS(cublasDdot(
      handle,
      n,
      x,
      incx,
      y,
      incy,
      result
  ));
}

// Template dispatch for SCAL.
template <typename ENTRY_T>
void cublasSCAL(cublasHandle_t handle,
                int n,
                const ENTRY_T* alpha,
                const ENTRY_T* x,
                int incx) { assert(false); }

template <>
void cublasSCAL<float>(cublasHandle_t handle,
                       int n,
                       const float* alpha,
                       const float* x,
                       int incx) {
  CHECK_CUBLAS(cublasSscal(
      handle,
      n,
      alpha,
      x,
      incx
  ));
}

template <>
void cublasSCAL<double>(cublasHandle_t handle,
                        int n,
                        const double* alpha,
                        const double* x,
                        int incx) {
  CHECK_CUBLAS(cublasDscal(
      handle,
      n,
      alpha,
      x,
      incx
  ));
}

} // namespace LegionSolvers