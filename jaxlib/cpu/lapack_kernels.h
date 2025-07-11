/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_CPU_LAPACK_KERNELS_H_
#define JAXLIB_CPU_LAPACK_KERNELS_H_

#include <complex>
#include <cstdint>
#include <type_traits>

#include "absl/status/statusor.h"
#include "xla/ffi/api/ffi.h"

// Underlying function pointers (i.e., KERNEL_CLASS::Fn) are initialized either
// by the nanobind wrapper that links them to an existing SciPy lapack instance,
// or using the lapack_kernels_strong.cc static initialization to link them
// directly to lapack for use in a pure C++ context.

namespace jax {

struct MatrixParams {
  enum class Side : char { kLeft = 'L', kRight = 'R' };
  enum class UpLo : char { kLower = 'L', kUpper = 'U' };
  enum class Diag : char { kNonUnit = 'N', kUnit = 'U' };
  enum class Transpose : char {
    kNoTrans = 'N',
    kTrans = 'T',
    kConjTrans = 'C'
  };
};

namespace svd {

enum class ComputationMode : char {
  kComputeFullUVt = 'A',  // Compute U and VT
  kComputeMinUVt = 'S',   // Compute min(M, N) columns of U and rows of VT
  kComputeVtOverwriteXPartialU = 'O',  // Compute VT, overwrite X
                                       // with partial U
  kNoComputeUVt = 'N',                 // Do not compute U or VT
};

inline bool ComputesUV(ComputationMode mode) {
  return mode == ComputationMode::kComputeFullUVt ||
         mode == ComputationMode::kComputeMinUVt;
}

}  // namespace svd

namespace eig {

enum class ComputationMode : char {
  kNoEigenvectors = 'N',
  kComputeEigenvectors = 'V',
};

}  // namespace eig

namespace schur {

enum class ComputationMode : char {
  kNoComputeSchurVectors = 'N',
  kComputeSchurVectors = 'V',
};

enum class Sort : char { kNoSortEigenvalues = 'N', kSortEigenvalues = 'S' };

}  // namespace schur

template <typename KernelType>
void AssignKernelFn(void* func) {
  KernelType::fn = reinterpret_cast<typename KernelType::FnType*>(func);
}

template <typename KernelType>
void AssignKernelFn(typename KernelType::FnType* func) {
  KernelType::fn = func;
}

}  // namespace jax

namespace jax {

using lapack_int = int;
inline constexpr auto LapackIntDtype = ::xla::ffi::DataType::S32;
static_assert(
    std::is_same_v<::xla::ffi::NativeType<LapackIntDtype>, lapack_int>);

//== Triangular System Solver ==//

template <::xla::ffi::DataType dtype>
struct TriMatrixEquationSolver {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* side, char* uplo, char* transa, char* diag,
                      lapack_int* m, lapack_int* n, ValueType* alpha,
                      ValueType* a, lapack_int* lda, ValueType* b,
                      lapack_int* ldb);

  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<dtype> y,
      ::xla::ffi::RemainingArgs, ::xla::ffi::ResultBuffer<dtype> y_out,
      MatrixParams::Side side, MatrixParams::UpLo uplo,
      MatrixParams::Transpose trans_x, MatrixParams::Diag diag);
};

//== LU Decomposition ==//

template <::xla::ffi::DataType dtype>
struct LuDecomposition {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, lapack_int* ipiv, lapack_int* info);

  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> ipiv,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

//== QR Factorization ==//

template <::xla::ffi::DataType dtype>
struct QrFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::ResultBuffer<dtype> x_out,
                                  ::xla::ffi::ResultBuffer<dtype> tau);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols);
};

//== Column Pivoting QR Factorization ==//

// lapack geqp3
template <::xla::ffi::DataType dtype>
struct PivotingQrFactorization {
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = std::conditional_t<
      ::xla::ffi::IsComplexType<dtype>(),
      void(lapack_int* m, lapack_int* n, ValueType* a, lapack_int* lda,
           lapack_int* jpvt, ValueType* tau, ValueType* work, lapack_int* lwork,
           RealType* rwork, lapack_int* info),
      void(lapack_int* m, lapack_int* n, ValueType* a, lapack_int* lda,
           lapack_int* jpvt, ValueType* tau, ValueType* work, lapack_int* lwork,
           lapack_int* info)>;

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<LapackIntDtype> jpvt,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> jpvt_out,
      ::xla::ffi::ResultBuffer<dtype> tau);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols);
};

//== Orthogonal QR ==//

template <::xla::ffi::DataType dtype>
struct OrthogonalQr {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, lapack_int* k, ValueType* a,
                      lapack_int* lda, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::Buffer<dtype> tau,
                                  ::xla::ffi::ResultBuffer<dtype> x_out);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                  lapack_int tau_size);
};

//== Cholesky Factorization ==//

template <::xla::ffi::DataType dtype>
struct CholeskyFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* uplo, lapack_int* n, ValueType* a, lapack_int* lda,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

//== Singular Value Decomposition (SVD) ==//

template <::xla::ffi::DataType dtype>
struct SingularValueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ValueType;
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, lapack_int* iwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(lapack_int x_rows,
                                                  lapack_int x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct SingularValueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, RealType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, RealType* rwork,
                      lapack_int* iwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(lapack_int x_rows,
                                                  lapack_int x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct SingularValueDecompositionQR {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ValueType;
  using FnType = void(char* jobu, char* jobvt, lapack_int* m, lapack_int* n,
                      ValueType* a, lapack_int* lda, ValueType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<lapack_int> GetWorkspaceSize(lapack_int x_rows,
                                                     lapack_int x_cols,
                                                     svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct SingularValueDecompositionQRComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobu, char* jobvt, lapack_int* m, lapack_int* n,
                      ValueType* a, lapack_int* lda, RealType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, RealType* rwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<lapack_int> GetWorkspaceSize(lapack_int x_rows,
                                                     lapack_int x_cols,
                                                     svd::ComputationMode mode);
};

namespace svd {

template <::xla::ffi::DataType dtype>
using SVDType = std::conditional_t<::xla::ffi::IsComplexType<dtype>(),
                                   SingularValueDecompositionComplex<dtype>,
                                   SingularValueDecomposition<dtype>>;

template <::xla::ffi::DataType dtype>
using SVDQRType = std::conditional_t<::xla::ffi::IsComplexType<dtype>(),
                                     SingularValueDecompositionQRComplex<dtype>,
                                     SingularValueDecompositionQR<dtype>>;

absl::StatusOr<lapack_int> GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols);
absl::StatusOr<lapack_int> GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                                                ComputationMode mode);
absl::StatusOr<lapack_int> GetRealWorkspaceSizeQR(int64_t x_rows,
                                                  int64_t x_cols);

}  // namespace svd

//== Eigenvalues and eigenvectors ==//

namespace eig {

// Eigenvalue Decomposition
absl::StatusOr<lapack_int> GetWorkspaceSize(int64_t x_cols,
                                            ComputationMode mode);
absl::StatusOr<lapack_int> GetIntWorkspaceSize(int64_t x_cols,
                                               ComputationMode mode);

// Hermitian Eigenvalue Decomposition
absl::StatusOr<lapack_int> GetComplexWorkspaceSize(int64_t x_cols,
                                                   ComputationMode mode);
absl::StatusOr<lapack_int> GetRealWorkspaceSize(int64_t x_cols,
                                                ComputationMode mode);

}  // namespace eig

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionSymmetric {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobz, char* uplo, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* w, ValueType* work,
                      lapack_int* lwork, lapack_int* iwork, lapack_int* liwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  MatrixParams::UpLo uplo,
                                  ::xla::ffi::ResultBuffer<dtype> x_out,
                                  ::xla::ffi::ResultBuffer<dtype> eigenvalues,
                                  ::xla::ffi::ResultBuffer<LapackIntDtype> info,
                                  eig::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionHermitian {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, char* uplo, lapack_int* n, ValueType* a,
                      lapack_int* lda, RealType* w, ValueType* work,
                      lapack_int* lwork, RealType* rwork, lapack_int* lrwork,
                      lapack_int* iwork, lapack_int* liwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> eigenvalues,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode);
};

// LAPACK uses a packed representation to represent a mixture of real
// eigenvectors and complex conjugate pairs. This helper unpacks the
// representation into regular complex matrices.
template <typename T, typename Int = lapack_int>
static void UnpackEigenvectors(Int n, const T* eigenvals_imag, const T* packed,
                               std::complex<T>* unpacked) {
  for (int j = 0; j < n;) {
    if (eigenvals_imag[j] == 0. || std::isnan(eigenvals_imag[j])) {
      // Real values in each row without imaginary part
      // Second row of the imaginary part is not provided
      for (int i = 0; i < n; ++i) {
        unpacked[j * n + i] = {packed[j * n + i], 0.};
      }
      ++j;
    } else {
      // Complex values where the real part is in the jth row
      // and the imaginary part is in the next row (j + 1)
      for (int i = 0; i < n; ++i) {
        const T real_part = packed[j * n + i];
        const T imag_part = packed[(j + 1) * n + i];
        unpacked[j * n + i] = {real_part, imag_part};
        unpacked[(j + 1) * n + i] = {real_part, -imag_part};
      }
      j += 2;
    }
  }
}

template <::xla::ffi::DataType dtype>
struct EigenvalueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* wr, ValueType* wi,
                      ValueType* vl, lapack_int* ldvl, ValueType* vr,
                      lapack_int* ldvr, ValueType* work, lapack_int* lwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals_real,
      ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_left,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* w, ValueType* vl,
                      lapack_int* ldvl, ValueType* vr, lapack_int* ldvr,
                      ValueType* work, lapack_int* lwork, RealType* rwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_left,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

//== Schur Decomposition ==//

template <::xla::ffi::DataType dtype>
struct SchurDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobvs, char* sort,
                      bool (*select)(ValueType, ValueType), lapack_int* n,
                      ValueType* a, lapack_int* lda, lapack_int* sdim,
                      ValueType* wr, ValueType* wi, ValueType* vs,
                      lapack_int* ldvs, ValueType* work, lapack_int* lwork,
                      bool* bwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode,
      schur::Sort sort, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> schur_vectors,
      ::xla::ffi::ResultBuffer<dtype> eigvals_real,
      ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
      ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  schur::ComputationMode mode,
                                  schur::Sort sort);
};

template <::xla::ffi::DataType dtype>
struct SchurDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobvs, char* sort, bool (*select)(ValueType),
                      lapack_int* n, ValueType* a, lapack_int* lda,
                      lapack_int* sdim, ValueType* w, ValueType* vs,
                      lapack_int* ldvs, ValueType* work, lapack_int* lwork,
                      RealType* rwork, bool* bwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode,
      schur::Sort sort, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> schur_vectors,
      ::xla::ffi::ResultBuffer<dtype> eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  schur::ComputationMode mode,
                                  schur::Sort sort);
};

//== Hessenberg Decomposition                                       ==//
//== Reduces a non-symmetric square matrix to upper Hessenberg form ==//

template <::xla::ffi::DataType dtype>
struct HessenbergDecomposition {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* n, lapack_int* ilo, lapack_int* ihi,
                      ValueType* a, lapack_int* lda, ValueType* tau,
                      ValueType* work, lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, lapack_int low, lapack_int high,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> tau,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                  lapack_int low, lapack_int high);
};

//== Tridiagonal Reduction                                           ==//
//== Reduces a Symmetric/Hermitian square matrix to tridiagonal form ==//

template <::xla::ffi::DataType dtype>
struct TridiagonalReduction {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* uplo, lapack_int* n, ValueType* a, lapack_int* lda,
                      RealType* d, RealType* e, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> diagonal,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> off_diagonal,
      ::xla::ffi::ResultBuffer<dtype> tau,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols);
};

//== General Tridiagonal System Solver ==//

template <::xla::ffi::DataType dtype>
struct TridiagonalSolver {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* n, lapack_int* nrhs, ValueType* dl,
                      ValueType* d, ValueType* du, ValueType* b,
                      lapack_int* ldb, lapack_int* info);

  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> dl, ::xla::ffi::Buffer<dtype> d,
      ::xla::ffi::Buffer<dtype> du, ::xla::ffi::Buffer<dtype> b,
      ::xla::ffi::ResultBuffer<dtype> dl_out,
      ::xla::ffi::ResultBuffer<dtype> d_out,
      ::xla::ffi::ResultBuffer<dtype> du_out,
      ::xla::ffi::ResultBuffer<dtype> b_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

// Declare all the handler symbols
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_strsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dtrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ctrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ztrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_spotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_chetrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zhetrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgtsv_ffi);

}  // namespace jax

#endif  // JAXLIB_CPU_LAPACK_KERNELS_H_
