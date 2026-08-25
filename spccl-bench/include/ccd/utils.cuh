#ifndef UTILS_CUH
#define UTILS_CUH

#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <bit>
#include <cuda.h>

#define CC(code) { cuda_check((code), __FILE__, __LINE__); }

inline void cuda_check(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__host__ __device__ __forceinline__
size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

__host__ __device__ __forceinline__
size_t alignment_offset(
    size_t n, size_t k
) {
    return (k - (n % k)) % k;
}

// rounds up x to nearest integer evenly divisible by
// k, where k is a power of two
__host__ __device__ __forceinline__
size_t round_up(size_t x, size_t k) {
    return (x + k - 1) & ~(k - 1);
}

double benchmark(std::function<void(void)> cb) {
    auto start = std::chrono::high_resolution_clock::now();
    cb();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
}

template<typename T>
T random_real(T a, T b, size_t seed = 42) {
    static std::random_device dev;
    static std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(a, b);
    return dist(rng);
}

bool keep_with_probability(double p, size_t seed = 42) {
    static std::mt19937 rng(seed);
    std::bernoulli_distribution dist(p);
    return dist(rng);
}

template<typename T = std::mt19937::result_type>
T random_integer(T a, T b, size_t seed = 42) {
    static std::random_device dev;
    static std::mt19937 rng(seed);//dev());
    std::uniform_int_distribution<T> dist(a, b);
    return dist(rng);
}

template<typename T>
void print_bits(T x, size_t max = sizeof(T) * 8) {
    for (int i = max - 1; i >= 0; i--) {
        std::cout << ((x & (T(1) << i)) ? "1" : "0");
    }
}

#endif /* UTILS_CUH */
