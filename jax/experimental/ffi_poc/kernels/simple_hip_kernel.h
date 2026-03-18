
#pragma once
#include <hip/hip_runtime.h>



__global__ void neuron_kernel_fwd(
    const float *a,
    const float *b,
    float *c,
    float *b_plus_1,  // intermediate output b+1
    size_t n
);


__global__ void neuron_kernel_bwd(
    const float *c_grad,    // incoming gradient wrt c
    const float *a,         // original input a
    const float *b_plus_1,  // intermediate output b+1
    float *a_grad,          // outgoing gradient wrt a
    float *b_grad,          // outgoing gradient wrt b
    size_t n
);


