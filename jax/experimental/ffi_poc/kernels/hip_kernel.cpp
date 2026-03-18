// file: hip_axpb1_mul.cpp
// Computes elementwise: c[i] = a[i] * (b[i] + 1)
//
// Build (ROCm/HIP):
//   hipcc -O2 hip_axpb1_mul.cpp -o hip_axpb1_mul
//
// Run:
//   ./hip_axpb1_mul

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err)                     \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void axpb1_mul_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * (b[idx] + 1.0f);
  }
}

int main() {
  const int n = 1 << 20;  // 1M elements
  const size_t bytes = n * sizeof(float);

  // Host data
  std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);

  for (int i = 0; i < n; ++i) {
    h_a[i] = 0.5f * i;
    h_b[i] = 0.25f * i;
    h_ref[i] = h_a[i] * (h_b[i] + 1.0f);
  }

  // Device data
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  // Launch
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  hipLaunchKernelGGL(axpb1_mul_kernel, dim3(blocks), dim3(threads), 0, 0,
                     d_a, d_b, d_c, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Copy back
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

  // Verify
  int errors = 0;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(h_c[i] - h_ref[i]) > 1e-5f) {
      if (errors < 5) {
        std::cerr << "Mismatch at " << i
                  << ": got " << h_c[i]
                  << ", expected " << h_ref[i] << "\n";
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << "Success! c[i] = a[i] * (b[i] + 1) computed correctly.\n";
    std::cout << "Example: c[10] = " << h_c[10] << "\n";
  } else {
    std::cerr << "Verification failed with " << errors << " errors.\n";
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
