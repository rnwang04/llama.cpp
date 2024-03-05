#pragma once
#include <sycl/sycl.hpp>

// memory
 void ggml_format_convert_to_xpu(const void * src, void * dst, size_t n, int qtype);

// kernel
extern void mul_mat_q4_0_sycl(
    const uint8_t* weight,
    const float* input,
    float* output,
    const int state_size,
    const int output_size,
    sycl::queue & queue);

extern void mul_mat_q8_0_sycl(
    const uint8_t* weight,
    const float* input,
    float* output,
    const int state_size,
    const int output_size,
    sycl::queue & queue);


extern void mul_mat_q4_1_sycl(
    const uint8_t* weight,
    const float* input,
    float* output,
    const int state_size,
    const int output_size,
    sycl::queue & queue);
