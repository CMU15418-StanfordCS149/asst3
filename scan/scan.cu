#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// 排除前缀和的核函数 (N 假设是2的幂)
__global__ void gpu_exclusive_scan(int* input, int N, int* result, int* debug) {
    // 计算 CUDA 线程ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread_id = %d\n", threadIdx.x);

    // cudaScan 函数已经让 result/debug 数组完全等同 input 数组，这里不需要再 memmove

    // // upsweep阶段
    // for (int two_d = 1; two_d <= N/2; two_d*=2) {
    //     int two_dplus1 = 2*two_d;
    //     // parallel_for (int i = 0; i < N; i += two_dplus1) {
    //     for (int i = 0; i < N; i += two_dplus1) {
    //         result[i+two_dplus1-1] += result[i+two_d-1];
    //     }
    // }
    // thread 0，检测看 result 是否完全等于 debug
    if(0 == thread_id) {
        for(int i = 0; i < N; i++) {
            if(result[i] != debug[i]) {
                printf("Error at initialization: i = %d, debug = %d, result = %d\n", i, debug[i], result[i]);
                return;
            }
        }
    }

    // 同步所有线程
    __syncthreads();
    
    // upsweep阶段
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        for (int i = 0; i < N; i += two_dplus1 * THREADS_PER_BLOCK) {
            // 防止越界
            if(i + (thread_id) * two_dplus1 + two_dplus1 - 1 < N) {
                if(i + (thread_id) * two_dplus1 + two_dplus1 - 1 == 131) {
                    printf("thread %d, two_d %d: original result = %d, added result = %d\n", thread_id, two_d, result[i + (thread_id) * two_dplus1 + two_dplus1 - 1], result[i + (thread_id) * two_dplus1 + two_d - 1]);
                }
                result[i + (thread_id) * two_dplus1 + two_dplus1 - 1] += result[i + (thread_id) * two_dplus1 + two_d - 1];
            }
        }
        // 同步所有线程
        __syncthreads();
        // 运行一遍 thread 0 单线程运算，检查看是否出错
        if(0 == thread_id) {
            for (int i = 0; i < N; i += two_dplus1) {
                int tmp = debug[i+two_dplus1-1];
                debug[i+two_dplus1-1] += debug[i+two_d-1];
                if(debug[i+two_dplus1-1] != result[i+two_dplus1-1]) {
                    printf("Error at upsweep: i = %d, two_d = %d, debug = %d, result = %d\n", i, two_d, debug[i+two_dplus1-1], result[i+two_dplus1-1]);
                    printf("Error at upsweep: original debug[i+two_dplus1-1] = %d, debug[i+two_d-1] = %d\n", tmp, debug[i+two_d-1]);
                    return;
                }
            }
        }
    }

    // 下面保持单线程，方便调试
    if(thread_id > 0)
        return;

    // 这里仅需一个线程执行即可
    if(thread_id == 0)
        result[N-1] = 0;

    // 同步所有线程
    __syncthreads();
    
    // downsweep阶段
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        // parallel_for (int i = 0; i < N; i += two_dplus1) {
        for (int i = 0; i < N; i += two_dplus1) {
            int t = result[i+two_d-1];
            result[i+two_d-1] = result[i+two_dplus1-1];
            result[i+two_dplus1-1] += t;
        }
        // // 同步所有线程
        // __syncthreads();
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
// void exclusive_scan(int* input, int N, int* result)
// 为了方便调试，申请一块调试用内存
void exclusive_scan(int* input, int N, int* result, int* debug)
{
    // CS149 TODO:
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    // 注意：这个函数是在CPU上运行的，但是传入的参数input/result是GPU上的数组

    // 定义线程块和网格大小
    int blockSize = THREADS_PER_BLOCK;  // 256个线程/块
    int rounded_length = nextPow2(N);
    int gridSize = (rounded_length + blockSize - 1) / blockSize; // 计算需要多少个块，根据roundup后的长度
    // 计算下一个2次幂，方便简化算法的计算 (这在最坏情况会引入2倍工作负载)
    // Launch kernel over the rounded (power-of-two) length
    gpu_exclusive_scan<<<gridSize, blockSize>>>(input, rounded_length, result, debug);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    // 为了方便调试，申请一块调试用内存
    int *device_debug;
    cudaMalloc((void **)&device_debug, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    // 为了方便调试，申请一块调试用内存
    cudaMemcpy(device_debug, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    // 为了方便调试，申请一块调试用内存
    // exclusive_scan(device_input, N, device_result);
    exclusive_scan(device_input, N, device_result, device_debug);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    // // 自己添加的调试代码：
    // for(int i = 0; i < N; i++) {
    //     printf("inarray[%d] = %d\n", i, inarray[i]);
    // }
    // for(int i = 0; i < N; i++) {
    //     printf("resultarray[%d] = %d\n", i, resultarray[i]);
    // }

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
