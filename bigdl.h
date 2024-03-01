#pragma once
#include <sycl/sycl.hpp>

// memory
 void ggml_q4_0_format_convert_to_xpu(const void * src, void * dst, size_t n);

// kernel
extern void mul_mat_q4_0_sycl(
    const uint8_t* weight,
    const float* input, // TODO: consider fp16 later
    float* output,
    const int state_size,
    const int output_size,
    sycl::queue & queue);
