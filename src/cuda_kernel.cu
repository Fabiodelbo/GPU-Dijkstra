// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>

// Include associated header file.
#include "../include/cuda_kernel.cuh"



// TODO: Define the kernel function right here
__global__ void short_path_update_naive(short* graph, short* dist, int* u, int V){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<V){
        dist[tid] = (short)min((dist[*u] + graph[*u*V+tid])+(short)(graph[*u*V+tid] == 0)*dist[tid], dist[tid]);
        //printf("dist[%d]= %hu\r\n",tid ,dist[tid]);
    }
}

__global__ void minDistance_kernel_naive(short* dist, bool* sptSet, int n, int V, short* out, int* out_idx,int* node_u){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid;

    if (i >= n) return;
    
    short val1 = (i < n && (n != V ||!sptSet[i])) ? dist[i] : (short)0x7FFF;
    short val2 = (i + 1 < n && ( n != V || !sptSet[i + 1])) ? dist[i + 1] : (short)0x7FFF;
    int idx1, idx2;
    if(n == V){
        idx1 = i;
        idx2 =  i + 1;
    }
    else{
        idx1 = out_idx[i];
        idx2 =  out_idx[i + 1];
    }
    

    if (val1 <= val2) {
        //printf("min(%hu,%hu) = %hu\r\n", val1, val2, val1);
        out[tid] = val1;
        out_idx[tid] = idx1;
        if(n==2){
            sptSet[idx1] = true;
            *node_u=idx1;
        }
    } else {
        //printf("min(%hu,%hu) = %hu\r\n", val1, val2, val2);
        out[tid] = val2;
        out_idx[tid] = idx2;
        if(n==2){
            /*find the minimum and update the vector*/
            sptSet[idx2] = true;
            *node_u=idx2;
        }
    }
    
    
}

int minDistance(short dist[], bool sptSet[], int V)
{
    // Initialize min value
    int min = 0x7FFF, min_index;

    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}


void dijkstra_parallelize_naive(short* graph, int src, short* dist, int V)
{   
    short *graph_d, *dist_d;
    int thread = 0;
        if(V<256)
            thread = V;
        else
            thread =256;

    cudaMalloc((void **) &graph_d, V*V*sizeof(short));
    cudaMalloc((void **) &dist_d, V*sizeof(short));
    cudaMemcpy(graph_d, graph, V*V*sizeof(short), cudaMemcpyHostToDevice);

    /*var for min*/
    int n = V;
    short* in_buff;
    bool *sptSet_d;
    short* out;
    int* outIdx;
    short *tmp_out;
    int* node_u;
    
    cudaMalloc(&sptSet_d, V*sizeof(bool));
    cudaMalloc(&out, (n/2+1)*sizeof(short));
    //temporary buffer to be switched with in buffer in order to get the min in divide and conquer mode
    cudaMalloc((void **) &tmp_out, (n/2+1)*sizeof(short));
    cudaMalloc(&outIdx, (n/2+1)*sizeof(int));
    cudaMalloc(&node_u, sizeof(int));
    

    bool sptSet[V]; //Set to true all the vertexes for which the shortest path has been already found

    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < V; i++)
        dist[i] = 0x7FFF, sptSet[i] = false;

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    cudaMemcpy(dist_d, dist, V*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(sptSet_d, sptSet, V*sizeof(bool), cudaMemcpyHostToDevice);
    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        //Pick the minimum distance vertex
        n = V;
        in_buff = dist_d;
        out;

        while (n > 1) {
            //printf("============\r\n");
            dim3 blockSize_min(thread,1,1);
            dim3 gridSize_min((n + thread*2 - 1) / (thread*2),1);
            minDistance_kernel_naive<<<gridSize_min, blockSize_min>>>(in_buff, sptSet_d, n, V, out, outIdx, node_u);
            cudaDeviceSynchronize();
            // swap in/out
            in_buff = out;
            out = tmp_out;
            n = (n + 1) / 2;
        }

        //int u;
        //cudaMemcpy(&u, outIdx, sizeof(int), cudaMemcpyDeviceToHost);
        //int u = minDistance(dist, sptSet, V);
        //sptSet[u] = true;
        //printf(" u :%d -", u);

        
        dim3 blockSize(thread,1,1);
        dim3 gridSize(V/thread+1,1);

        short_path_update_naive<<<gridSize, blockSize>>>(graph_d, dist_d, node_u, V);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        /*cudaMemcpy(dist, dist_d, V*sizeof(short), cudaMemcpyDeviceToHost);
        printf("dist: ");
        for (int v = 0; v < V; v++)
            printf("%hu, ",dist[v]);
        printf("\r\n");*/
        
    }
    cudaMemcpy(dist, dist_d, V*sizeof(short), cudaMemcpyDeviceToHost);
}










