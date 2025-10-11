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

__global__ void gpu_print(int* result, int N) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id > 0)
        return;
    for(int i = 0; i < N; i++) {
        printf("gpu_print: result[%d] = %d\n", i, result[i]);
    }
}

__global__ void gpu_upsweep(int* result, int rounded_length, int two_d) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID（使用64位以防止乘法溢出）
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long two_d_ll = (long long)two_d;
    long long left_idx = thread_id*2*two_d_ll + two_d_ll - 1;
    long long right_idx = thread_id*2*two_d_ll + 2*two_d_ll - 1;
    // 防止越界（比较也用64位）
    if (right_idx < (long long)rounded_length) {
        result[right_idx] += result[left_idx];
    }
}

__global__ void gpu_setzero(int* result, int rounded_length) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID（使用64位以防止溢出）
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // 只让线程 0 更新这个元素
    if(thread_id == 0) {
        result[rounded_length - 1] = 0;
    }
}

__global__ void gpu_downsweep(int* result, int rounded_length, int two_d) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID（使用64位以防止乘法溢出）
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long two_d_ll = (long long)two_d;
    long long left_idx = thread_id*2*two_d_ll + two_d_ll - 1;
    long long right_idx = thread_id*2*two_d_ll + 2*two_d_ll - 1;
    // 防止越界
    if (right_idx < (long long)rounded_length) {
        int t = result[left_idx];
        result[left_idx] = result[right_idx];
        result[right_idx] += t;
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
    // 计算下一个2次幂，方便简化算法的计算 (这在最坏情况会引入2倍工作负载)
    int rounded_length = nextPow2(N);
    // // 使用 rounded_length 计算 gridSize
    // int gridSize = (rounded_length + blockSize - 1) / blockSize;

    // cudaScan 函数已经让 result/debug 数组完全等同 input 数组，这里不需要再 memmove

    // upsweep阶段
    for (int two_d = 1; two_d <= rounded_length/2; two_d*=2) {
        // gpu_upsweep<<<gridSize, blockSize>>>(result, rounded_length, two_d);
        // 只启动需要的线程
        // 计算需要的线程
        int activeThreads = rounded_length / (2 * two_d);
        // 用这个数字计算网格大小
        int grid = (activeThreads + blockSize - 1) / blockSize;
        // 启动 CUDA 内核函数
        if (grid > 0) {
            gpu_upsweep<<<grid, blockSize>>>(result, rounded_length, two_d);
        }
    }

    // 设置最后一个元素为0
    // gpu_setzero<<<gridSize, blockSize>>>(result, rounded_length);
    // 这里只需要一个线程 0
    gpu_setzero<<<1, 1>>>(result, rounded_length);

    // downsweep阶段
    for (int two_d = rounded_length/2; two_d >= 1; two_d /= 2) {
        // gpu_downsweep<<<gridSize, blockSize>>>(result, rounded_length, two_d);
        // 只启动需要的线程
        // 计算需要的线程
        int activeThreads = rounded_length / (2 * two_d);
        // 用这个数字计算网格大小
        int grid = (activeThreads + blockSize - 1) / blockSize;
        if (grid > 0) {
            gpu_downsweep<<<grid, blockSize>>>(result, rounded_length, two_d);
        }
    }
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

// 这里对于大于 length 的位置全部标为 0
__global__ void gpu_mark1forNeighborSame(int* input, int length, int rounded_length, int* output) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID（使用64位以防止乘法溢出）
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < rounded_length) {
        if(thread_id < length-1) {
            // 只要和下一个元素相等就标1
            output[thread_id] = input[thread_id] == input[thread_id + 1] ? 1 : 0;
        } else {
            // 大于等于 length 的部分全部标0
            output[thread_id] = 0;
        }
    }
}

// 遍历输入数组，把所有 input[i] != input[i+1] 的索引写入 output
__global__ void gpu_findrepeats_helper(int* input, int length, int* output) {
    // 线程块内的 ID: threadIdx.x
    // 计算全局 CUDA 线程ID（使用64位以防止乘法溢出）
    long long thread_id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < length) {
        if(input[thread_id] != input[thread_id + 1]) {
            // 这里直接把索引写入 output
            output[input[thread_id]] = thread_id;
        }
    }
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
    // 为什么要用下面这种复杂的方式求重复数据索引？答：常规方式并行度不高，因此使用这种方式
    // 这里的思路是:
    // 1. 先计算一个辅助数组 aux，aux[i] = 1 if input[i] == input[i+1] else 0
    // 2. 对 aux 做一个 exclusive scan，得到 aux_scan
    // 3. 再对 aux_scan 做一个遍历，把所有 "aux_scan[i] != aux_scan[i+1]" 索引写入 output
    // 4. aux_scan 最后一个元素的值就是重复数据的个数

    // gpu_print<<<1,1>>>(device_input, length);

    // 定义线程块和网格大小
    int blockSize = THREADS_PER_BLOCK;  // 256个线程/块
    // 计算下一个2次幂，方便简化算法的计算 (这在最坏情况会引入2倍工作负载)
    int rounded_length = nextPow2(length);
    // 使用 rounded_length 计算 gridSize
    int gridSize = (rounded_length + blockSize - 1) / blockSize;

    // 申请临时 cuda 数组
    int* device_tmp;
    cudaMalloc((void**)&device_tmp, rounded_length * sizeof(int));

    // 1. 先计算一个辅助数组 aux，aux[i] = 1 if input[i] == input[i+1] else 0
    gpu_mark1forNeighborSame<<<gridSize, blockSize>>>(device_input, length, rounded_length, device_tmp);
    // gpu_print<<<1,1>>>(device_tmp, length);
    // 2. 对 aux 做一个 exclusive scan，得到 aux_scan
    exclusive_scan(device_tmp, length, device_tmp, nullptr);
    // gpu_print<<<1,1>>>(device_tmp, length);
    // 4. aux_scan 最后一个元素的值就是重复数据的个数
    int repeats_num;
    cudaMemcpy(&repeats_num, &device_tmp[length - 1], sizeof(int),
            cudaMemcpyDeviceToHost);
    // 3. 再对 aux_scan 做一个遍历，把所有 "aux_scan[i] != aux_scan[i+1]" 索引写入 output
    gpu_findrepeats_helper<<<gridSize, blockSize>>>(device_tmp, length, device_output);

    // 释放临时 CUDA 数组
    cudaFree(device_tmp);
    // 返回重复数据个数
    return repeats_num; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);

    // // 自己添加的调试代码：
    // for(int i = 0; i < length; i++) {
    //     printf("input[%d] = %d\n", i, input[i]);
    // }
    
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
