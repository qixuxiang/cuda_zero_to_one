// 相关 CUDA 库
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

using namespace std;

const int N = 100;

// 块数
const int BLOCK_data = 3; 
// 各块中的线程数
const int THREAD_data = 10; 

// CUDA初始化函数
bool InitCUDA()
{
    int deviceCount; 

    // 获取显示设备数
    cudaGetDeviceCount (&deviceCount);

    if (deviceCount == 0) 
    {
        cout << "找不到设备" << endl;
        return EXIT_FAILURE;
    }

    int i;
    for (i=0; i<deviceCount; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop,i)==cudaSuccess) // 获取设备属性
        {
            if (prop.major>=1) //cuda计算能力
            {
                break;
            }
        }
    }

    if (i==deviceCount)
    {
        cout << "找不到支持 CUDA 计算的设备" << endl;
        return EXIT_FAILURE;
    }

    cudaSetDevice(i); // 选定使用的显示设备

    return EXIT_SUCCESS;
}

// 此函数在主机端调用，设备端执行。
__global__ 
static void Sum (int *data,int *result)
{
    // 取得线程号
    const int tid = threadIdx.x; 
    // 获得块号
    const int bid = blockIdx.x; 
    
    int sum = 0;

    // 有点像网格计算的思路
    for (int i=bid*THREAD_data+tid; i<N; i+=BLOCK_data*THREAD_data)
    {
        sum += data[i];
    }
    
    // result 数组存放各个线程的计算结果
    result[bid*THREAD_data+tid] = sum; 
}

int main ()
{
    // 初始化 CUDA 编译环境
    if (InitCUDA()) {
        return EXIT_FAILURE;
    }
    cout << "成功建立 CUDA 计算环境" << endl << endl;

    // 建立，初始化，打印测试数组
    int *data = new int [N];
    cout << "测试矩阵: " << endl;
    for (int i=0; i<N; i++)
    {
        data[i] = rand()%10;
        cout << data[i] << " ";
        if ((i+1)%10 == 0) cout << endl;
    }
    cout << endl;

    int *gpudata, *result; 
    
    // 在显存中为计算对象开辟空间
    cudaMalloc ((void**)&gpudata, sizeof(int)*N); 
    // 在显存中为结果对象开辟空间
    cudaMalloc ((void**)&result, sizeof(int)*BLOCK_data*THREAD_data);
    
    // 将数组数据传输进显存
    cudaMemcpy (gpudata, data, sizeof(int)*N, cudaMemcpyHostToDevice); 
    // 调用 kernel 函数 - 此函数可以根据显存地址以及自身的块号，线程号处理数据。
    Sum<<<BLOCK_data,THREAD_data,0>>> (gpudata,result);
    
    // 在内存中为计算对象开辟空间
    int *sumArray = new int[THREAD_data*BLOCK_data];
    // 从显存获取处理的结果
    cudaMemcpy (sumArray, result, sizeof(int)*THREAD_data*BLOCK_data, cudaMemcpyDeviceToHost);
    
    // 释放显存
    cudaFree (gpudata); 
    cudaFree (result);

    // 计算 GPU 每个线程计算出来和的总和
    int final_sum=0;
    for (int i=0; i<THREAD_data*BLOCK_data; i++)
    {
        final_sum += sumArray[i];
    }

    cout << "GPU 求和结果为: " << final_sum << endl;

    // 使用 CPU 对矩阵进行求和并将结果对照
    final_sum = 0;
    for (int i=0; i<N; i++)
    {
        final_sum += data[i];
    }
    cout << "CPU 求和结果为: " << final_sum << endl;

    return 0;
}
