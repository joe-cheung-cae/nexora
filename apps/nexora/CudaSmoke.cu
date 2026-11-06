#include <cuda_runtime.h>

#include <iostream>

namespace {

struct Resources {
    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    cudaEvent_t produced = nullptr;
    cudaEvent_t consumed = nullptr;
    void* host_allocation = nullptr;
    void* device_allocation = nullptr;
    bool complete = false;

    ~Resources() {
        // Drain this probe's streams on error before releasing borrowed memory.
        if (!complete) {
            if (producer)
                cudaStreamSynchronize(producer);
            if (consumer)
                cudaStreamSynchronize(consumer);
        }
        if (consumed)
            cudaEventDestroy(consumed);
        if (produced)
            cudaEventDestroy(produced);
        if (consumer)
            cudaStreamDestroy(consumer);
        if (producer)
            cudaStreamDestroy(producer);
        if (device_allocation)
            cudaFree(device_allocation);
        if (host_allocation)
            cudaFreeHost(host_allocation);
    }
};

bool checked(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
    return false;
}

__global__ void twice(int* value) {
    *value *= 2;
}

} // namespace

int cuda_smoke() {
    int devices = 0;
    const auto discovery = cudaGetDeviceCount(&devices);
    if (discovery == cudaErrorNoDevice || (discovery == cudaSuccess && devices == 0)) {
        std::cerr << "CUDA smoke skipped: no visible CUDA device.\n";
        return 77;
    }
    if (!checked(discovery, "cudaGetDeviceCount")) {
        return 1;
    }

    Resources resources;
    if (!checked(cudaStreamCreateWithFlags(&resources.producer, cudaStreamNonBlocking),
                 "create producer") ||
        !checked(cudaStreamCreateWithFlags(&resources.consumer, cudaStreamNonBlocking),
                 "create consumer") ||
        !checked(cudaEventCreateWithFlags(&resources.produced, cudaEventDisableTiming),
                 "create producer event") ||
        !checked(cudaEventCreateWithFlags(&resources.consumed, cudaEventDisableTiming),
                 "create consumer event") ||
        !checked(cudaMallocHost(&resources.host_allocation, sizeof(int)), "cudaMallocHost") ||
        !checked(cudaMalloc(&resources.device_allocation, sizeof(int)), "cudaMalloc")) {
        return 1;
    }

    auto* const host = static_cast<int*>(resources.host_allocation);
    auto* const device = static_cast<int*>(resources.device_allocation);
    *host = 21;
    if (!checked(
            cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, resources.producer),
            "copy to device")) {
        return 1;
    }
    twice<<<1, 1, 0, resources.producer>>>(device);
    if (!checked(cudaGetLastError(), "launch twice") ||
        !checked(cudaEventRecord(resources.produced, resources.producer),
                 "record producer event") ||
        !checked(cudaStreamWaitEvent(resources.consumer, resources.produced, 0),
                 "wait for producer") ||
        !checked(
            cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, resources.consumer),
            "copy to host") ||
        !checked(cudaEventRecord(resources.consumed, resources.consumer),
                 "record consumer event") ||
        !checked(cudaEventSynchronize(resources.consumed), "wait for consumer")) {
        return 1;
    }

    resources.complete = true;
    if (*host != 42) {
        std::cerr << "CUDA smoke failed: expected 42, got " << *host << ".\n";
        return 1;
    }
    std::cout << "CUDA smoke passed: two non-default streams, event dependency, result 42.\n";
    return 0;
}
