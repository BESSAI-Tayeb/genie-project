// Lightweight host-side API header for bindings.cpp
// This header intentionally avoids including CUDA/OptiX headers so it can be compiled with host compilers.

#pragma once

// Minimal vector types to match CUDA float4/float3 in the interface
typedef struct { float x, y, z, w; } float4;
typedef struct { float x, y, z; } float3;
typedef struct { int x, y, z; } int3;

// Opaque handles for GPU/OptiX structures - host code treats these as pointers
typedef void* OptixDeviceContext;
typedef void* OptixModule;
typedef void* OptixProgramGroup;
typedef void* OptixPipeline;
typedef unsigned long long OptixTraversableHandle;

typedef struct S_CUDA_KNN {
    OptixDeviceContext optixContext;
    OptixModule module;
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPG;
    OptixProgramGroup hitgroupPG;
    OptixPipeline pipeline;
    void *sbt;
    void *raygenRecordsBuffer;
    void *missRecordsBuffer;
    void *hitgroupRecordsBuffer;
    float chi_square_squared_radius;
    void *gaussian_as_polygon_vertices;
    int3 *gaussian_as_polygon_indices;
    OptixTraversableHandle GAS;
    void *GASBuffer;
    float4 *means;
    int number_of_means;
    void *instancesBuffer;
    OptixTraversableHandle IAS;
    void *IASBuffer;
} S_CUDA_KNN;

extern "C" {
    bool CUDA_KNN_Init(float chi_square_squared_radius, S_CUDA_KNN *knn);
    bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn);
    bool CUDA_KNN_KNeighbors(float4 *queried_points, int number_of_queried_points, int K, float *distances, int *indices, S_CUDA_KNN* knn);
    bool CUDA_KNN_Destroy(S_CUDA_KNN* cknn);
}
