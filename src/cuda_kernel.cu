// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <vector>

#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <vector>
#include <cstdint>

// Include associated header file.
#include "../include/cuda_kernel.cuh"
//#define PRINT

constexpr int INF = 0x3f3f3f3f;

void print_gpu_properties() {

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("Printing CUDA device informations...\n");
    printf("Device name:\t\t\t\t\t\t%s\n", p.name);
    printf("Amount of Global memory (bytes):\t\t\t%lu\n", p.totalGlobalMem);
    printf("Maximum amount of Shared memory per block (bytes):\t%lu\n", p.sharedMemPerBlock);
    printf("Amount of Constant memory (bytes):\t\t\t%lu\n", p.totalConstMem);
    printf("Number of SM:\t\t\t\t\t\t%d\n", p.multiProcessorCount);
    printf("Concurrent kernels supported:\t\t\t\t%d\n", p.concurrentKernels);
    printf("\n");

}


__device__ unsigned long long global_min = 0x7FFFFFFF<<32;
__device__ int global_u = 0;
__device__ inline unsigned long long packMin(int val, int idx) {
    // Valore nei 32 bit alti, indice nei 32 bit bassi
    return (((unsigned long long)(unsigned int)val) << 32) | (unsigned int)idx;
}

__device__ inline int unpackVal(unsigned long long x) {
    return (int)(x >> 32);
}

__device__ inline int unpackIdx(unsigned long long x) {
    return (int)(x & 0xFFFFFFFF);
}

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
        out[tid] = val1;
        out_idx[tid] = idx1;
        if(n==2){
            sptSet[idx1] = true;
            *node_u=idx1;
        }
    } else {
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
            dim3 blockSize_min(thread,1,1);
            dim3 gridSize_min((n + thread*2 - 1) / (thread*2),1);
            minDistance_kernel_naive<<<gridSize_min, blockSize_min>>>(in_buff, sptSet_d, n, V, out, outIdx, node_u);
            cudaDeviceSynchronize();
            // swap in/out
            in_buff = out;
            out = tmp_out;
            n = (n + 1) / 2;
        }
        
        dim3 blockSize(thread,1,1);
        dim3 gridSize(V/thread+1,1);

        short_path_update_naive<<<gridSize, blockSize>>>(graph_d, dist_d, node_u, V);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        
    }
    cudaMemcpy(dist, dist_d, V*sizeof(short), cudaMemcpyDeviceToHost);
}

__global__ void minDistance_kernel_shared(short *dist, bool *sptSet, short *min_val_res, int* min_idx_res){
    __shared__ short min[BLOCK_DIM];
    __shared__ int idx_min[BLOCK_DIM];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < VERTEX && !sptSet[i]){
        min[tid] = dist[i];
        idx_min[tid] = i;
    }
    else{
        min[tid] = (short)0x7FFF;
        idx_min[tid] = -1;
    }
    __syncthreads();

    for(int s = BLOCK_DIM/2; s>0; s>>=1){
        if(tid<s){
            if(min[tid+s]<min[tid]){
                min[tid] = min[tid+s];
                idx_min[tid] = idx_min[tid+s];
            }
        }
        //we have to wait that all the threads update the value before searching in the inner block other min
        __syncthreads();
    }

    if(tid==0){
        min_idx_res[blockIdx.x]=idx_min[0];
        min_val_res[blockIdx.x]=min[0];
    }

}

__global__ void reduceMin_kernel_shared(short* min_val_res, int* min_idx_res, bool* sptSet, int* node_u, int numBlocks) {
    __shared__ short sVal[BLOCK_DIM];
    __shared__ int   sIdx[BLOCK_DIM];

    int tid = threadIdx.x;
    /*put in shared mem only the minimum from previous step*/
    if (tid < numBlocks) {
        sVal[tid] = min_val_res[tid];
        sIdx[tid] = min_idx_res[tid];
    } else {
        sVal[tid] = (short)0x7FFF;
        sIdx[tid] = -1;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sVal[tid + s] < sVal[tid]) {
                sVal[tid] = sVal[tid + s];
                sIdx[tid] = sIdx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *node_u = sIdx[0];
        /*update sptSet because we are going to udate this node*/
        sptSet[*node_u] = true;
    }
}

void dijkstra_parallelize_shared(short* graph, int src, short* dist)
{   
    short *graph_d, *dist_d, *relaxed_val;
    bool  *sptSet_d;
    int   *relaxed_idx_d, *node_u;
    

    cudaMalloc(&graph_d,  VERTEX*VERTEX*sizeof(short));
    cudaMalloc(&dist_d,   VERTEX*sizeof(short));
    cudaMalloc(&sptSet_d, VERTEX*sizeof(bool));

    int numBlocks = (VERTEX + BLOCK_DIM - 1) / BLOCK_DIM;
    cudaMalloc(&relaxed_val,  numBlocks * sizeof(short));
    cudaMalloc(&relaxed_idx_d,numBlocks * sizeof(int));
    cudaMalloc(&node_u, sizeof(int));

    cudaMemcpy(graph_d, graph, VERTEX*VERTEX*sizeof(short), cudaMemcpyHostToDevice);

    // init host
    for (int i=0;i<VERTEX;i++) 
    dist[i] = (short)0x7FFF;
    dist[src] = 0;

    // init device
    cudaMemcpy(dist_d, dist, VERTEX*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemset(sptSet_d, 0, VERTEX*sizeof(bool)); // set all false

    dim3 block(BLOCK_DIM,1,1);
    dim3 grid(numBlocks,1,1);

    /*Assure that the number of thread is a power of 2 in order to correctly apply the reduction in successive halves*/
    int reduceBlock = 1;
    while (reduceBlock < numBlocks) reduceBlock <<= 1;
    reduceBlock = min(reduceBlock, BLOCK_DIM);

    for (int it = 0; it < VERTEX - 1; ++it) {
         /*find 1 minimum per block */
         minDistance_kernel_shared<<<grid, block>>>(dist_d, sptSet_d, relaxed_val, relaxed_idx_d);
         cudaDeviceSynchronize();
         /*Relax all block's minimum and find the global minimum*/
         reduceMin_kernel_shared<<<1, reduceBlock>>>(relaxed_val, relaxed_idx_d, sptSet_d, node_u, numBlocks);
         cudaDeviceSynchronize();
         /*Relax all the neighbour of the minimum vertex*/

        short_path_update_naive<<<grid, block>>>(graph_d, dist_d, node_u, VERTEX);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(dist, dist_d, VERTEX*sizeof(short), cudaMemcpyDeviceToHost);
    cudaFree(graph_d); cudaFree(dist_d); cudaFree(sptSet_d);
    cudaFree(relaxed_val); cudaFree(relaxed_idx_d); cudaFree(node_u);
}

__global__ void fusedMinKernel(short* dist, unsigned char* sptSet, int V, int* node_u) {
    __shared__ short localMinVal[BLOCK_DIM];
    __shared__ int localMinIdx[BLOCK_DIM];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Put all non visited node int shared mem
    if (i < V && !sptSet[i]) {
        localMinVal[tid] = dist[i];
        localMinIdx[tid] = i;
    } else {
        localMinVal[tid] = (short)0x7FFF;
        localMinIdx[tid] = -1;
        //printf("Useless thread\r\n");
    }
    __syncthreads();

    // min reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (localMinVal[tid+s] < localMinVal[tid]) {
                localMinVal[tid] = localMinVal[tid+s];
                localMinIdx[tid] = localMinIdx[tid+s];
            }
        }
        __syncthreads();
    }

    // thread 0 of each block check if the shared minimal of the block is lower than global minimal
    if (tid == 0) {
        // checks and set global minimal

        unsigned long long pack = packMin((int)localMinVal[0], localMinIdx[0]);
        unsigned long long old = atomicMin(&global_min, pack);
    }
}

// TODO: Define the kernel function right here
__global__ void short_path_update_atomic(short* graph, short* dist, unsigned char* sptSet, int* u, int V){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int min_g = unpackIdx(global_min);
    short dist_c = dist[tid];
    short graph_elem = graph[min_g*V+tid];
    
    if(min_g == -1){
        //printf("Error -1\r\n");
    }
    if(tid<V){
        dist[tid] = (short)min(((int)dist[min_g] + (int)graph_elem)+(int)(graph_elem == 0)*(int)dist_c, (int)dist_c);
    }
    if(tid == 0){
    //only one thread, the first for convenience update the visited node and re-initilize the global minimum
    __threadfence();
    sptSet[min_g] = 1;
    __threadfence();
    }
}

auto checkCuda = [](const char* tag){
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
};

void dijkstra_parallelize_shared_atomic(short* graph, int src, short* dist)
{   
    short *graph_d, *dist_d;
    unsigned char  *sptSet_d;
    int   *node_u;
    
    cudaMalloc(&graph_d,  VERTEX*VERTEX*sizeof(short));
    cudaMalloc(&dist_d,   VERTEX*sizeof(short));
    cudaMalloc(&sptSet_d, VERTEX*sizeof(unsigned char));

    int numBlocks = (VERTEX + BLOCK_DIM - 1) / BLOCK_DIM;
    cudaMalloc(&node_u, sizeof(int));

    cudaMemcpy(graph_d, graph, VERTEX*VERTEX*sizeof(short), cudaMemcpyHostToDevice);

    // init host
    for (int i=0;i<VERTEX;i++) 
    dist[i] = (short)0x7FFF;
    dist[src] = 0;
    

    // init device
    cudaMemcpy(dist_d, dist, VERTEX*sizeof(short), cudaMemcpyHostToDevice);
    cudaMemset(sptSet_d, 0, VERTEX*sizeof(unsigned char)); // set all false

    dim3 block(BLOCK_DIM,1,1);
    dim3 grid(numBlocks,1,1);
    unsigned long long init = (((unsigned long long)(unsigned int)0x7FFFFFFF) << 32) | (unsigned int)-1;

    for (int it = 0; it < VERTEX - 1; it++) {
        //printf("V: %d\r\n", it);
        
        cudaMemcpyToSymbol(global_min, &init, sizeof(init));
        checkCuda("memCpy to symbol reset");
        fusedMinKernel<<<grid, block>>>(dist_d, sptSet_d, VERTEX, node_u);
        checkCuda("launch fudes kernel");
        cudaDeviceSynchronize();

        short_path_update_atomic<<<grid, block>>>(graph_d, dist_d, sptSet_d, node_u, VERTEX);
        checkCuda("Launch relax node");
        cudaDeviceSynchronize();
    }

    cudaMemcpy(dist, dist_d, VERTEX*sizeof(short), cudaMemcpyDeviceToHost);
    cudaFree(graph_d); cudaFree(dist_d); cudaFree(sptSet_d);
    cudaFree(node_u);
}

/* ************************************* */
/* ****DELTA STEPING IMPLEMENTATION **** */
/* ************************************* */

// --- helper check
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// --- Light edges kernel (device buckets, simpler version)
__global__ void kernel_light_step(
    const int* frontier, int frontier_size, int n,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta, int bucket_idx,
    int* next_frontier, int* next_cnt,
    int* future_buckets,
    int* d_heavy_nodes, int* pos_heavy, int* bucket_heavy,
    int* d_next_bucket, int current_gen, int* gen_seen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];
    if (u < 0) return;
    

    int du = tent[u];
    if (du == INF) return;

    bool have_heavy = false;

    size_t start = row_ptr[u];
    size_t end   = row_ptr[u+1];

    #ifdef PRINT
    printf("%d ", u);
    #endif
    for (size_t e = start; e < end; ++e) {
        int v = col_ind[e];
        int w = weights[e];
        if (w <= delta) {
            int newd = du + w;
            // atomicMin returns old value; if newd < old then current thread won the update
            int old = atomicMin(&tent[v], newd);

            if (newd < old) {
                int newBucket = newd / delta;
                int oldBucket = atomicMin(&future_buckets[v], newBucket);
                // try to mark the vertex seen in this generation, is it is already seen in this generation we don't have to add it to next_frontier because it is already added
                int old_gen = atomicMax(&gen_seen[v], current_gen);
                if(newBucket <= oldBucket){
                    if (newBucket == bucket_idx && old_gen < current_gen) {
                            // new bucket for v edge is the actual bucket
                            int pos = atomicAdd(next_cnt, 1);
                            next_frontier[pos] = v;
                            // Inside computing of nexr t bucket to be processed
                            atomicMin(d_next_bucket, newBucket);
                    } else if(newBucket != oldBucket) {                    
                        // edge is already in a lower bucket so it will be processed later
                        atomicMin(d_next_bucket, newBucket);
                    }
                }
            }
        } else {
            // heavy edge: mark that u has heavy outgoing edges; heavy edges handled later
            have_heavy = true;
            // NOT update tent[v]; heavy kernel will do that
        }
    }

    if (have_heavy) {
        // it has not to be inserted in heavy in any light phase iteration of this bucket
        int heavy = atomicMax(&bucket_heavy[u], bucket_idx);
        if(heavy < bucket_idx){
            //node has not been added to heavy nodes for this bucket
            //this will save space not adding multiple times the same node
            int pos = atomicAdd(pos_heavy, 1);
            if (pos >= 0 && pos < n) {
                // d_heavy_nodes buffer was allocated with size >= n, so this should be safe
                d_heavy_nodes[pos] = u;
            } else {
                // out-of-space (unlikely if d_heavy_nodes has size n)
                printf("Out of space in heavy nodes array\n");
                return;
            }
        }
        
    }
}

// --- Heavy edges kernel (process nodes in S)
__global__ void kernel_heavy_step(
    const int* S, int S_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta, int bucket_idx,
    int* future_buckets,
    int* d_next_bucket)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= S_size) return;
    int u = S[tid];
    if (u < 0) return;
    int du = tent[u];
    if (du == INF) return;

    size_t start = row_ptr[u];
    size_t end   = row_ptr[u+1];
    // take all the edges from the current vertex; choose only the heavy ones
    for (size_t e = start; e < end; ++e) {
        int v = col_ind[e];
        int w = weights[e];
        if (w > delta) {
            int newd = du + w;
            int old = atomicMin(&tent[v], newd);
            // only if we find a shorter path to v we update future_buckets
            if (newd < old) {
                int newBucket = newd / delta;
                int oldBucket = atomicMin(&future_buckets[v], newBucket);
                atomicMin(d_next_bucket, newBucket);
            }
        }
    }
}


// --- Collect nodes belonging to targetBucket from future arrays into frontier.
//future_buckets entries are consumed (set to -1) when collected.
//TODO: optimize mem by deleting consumed entries (compaction)
__global__ void kernel_collect_bucket(
    int* future_buckets, int n,
    int targetBucket,
    int* frontier, int* frontier_cnt)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int b = future_buckets[idx];
    // if not in target bucket, ignore
    if (b != targetBucket) return;

    int v = idx;
    
    // add to next bucket
    int pos = atomicAdd(frontier_cnt, 1);
    frontier[pos] = v;
    
    // mark consumed (so future collects ignore it)
    atomicExch(&future_buckets[idx], INT_MAX);
}


// ----------------- Host-side driver (device-centered) -----------------

// count edges in a frontier from device
static inline int get_int_from_device(int* d_ptr) {
    int x = 0;
    CUDA_CHECK(cudaMemcpy(&x, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return x;
}

// delta_stepping GPU: expects CSR on host (h_row_ptr,h_col_ind,h_weights)
void delta_stepping_gpu_device_buckets(
    int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta, int* dist_h)
{
    // copy CSR to device
    size_t* d_row_ptr; int *d_col_ind, *d_weights;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind, m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, m * sizeof(int), cudaMemcpyHostToDevice));

    // nextBucket: device-side min bucket for future bucket to be processed
    int *d_next_bucket;
    CUDA_CHECK(cudaMalloc(&d_next_bucket, sizeof(int)));
    int infv = INT_MAX;
    CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));
    int *seen_gen;
    CUDA_CHECK(cudaMalloc(&seen_gen, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(seen_gen, -1, n * sizeof(int)));
    int current_gen = 0;

    // allocate device arrays
    int* d_tent; CUDA_CHECK(cudaMalloc(&d_tent, n * sizeof(int)));
    // init tent (host->device)
    std::vector<int> h_tent_init(n, INF); 
    h_tent_init[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_tent, h_tent_init.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    //TODO: optimize memory usage by reusing
    // frontier buffers (device)
    int *d_frontier, *d_next_frontier;
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    int *d_frontier_size; 
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    int *d_next_size; 
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    // store initial frontier = {source}
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int one = 1; 
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &one, sizeof(int), cudaMemcpyHostToDevice));  

    int* d_future_buckets; 
    CUDA_CHECK(cudaMalloc(&d_future_buckets, n * sizeof(int)));
    std::vector<int> tmp_proc2(n, INT_MAX); 
    CUDA_CHECK(cudaMemcpy(d_future_buckets, tmp_proc2.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // heavy nodes bookkeeping
    int* bucket_heavy;
    CUDA_CHECK(cudaMalloc(&bucket_heavy, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(bucket_heavy, -1, n * sizeof(int)));

    int* d_future_cnt; 
    CUDA_CHECK(cudaMalloc(&d_future_cnt, sizeof(int)));

    // heavy nodes buffer in devide
    int *d_heavy_nodes; 
    CUDA_CHECK(cudaMalloc(&d_heavy_nodes, n* sizeof(int)));
    int *pos_heavy; 
    CUDA_CHECK(cudaMalloc(&pos_heavy, sizeof(int)));
    CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

    int *future_pos; 
    CUDA_CHECK(cudaMalloc(&future_pos, n * sizeof(int)));
    std::vector<int> tmp_pos(n, INT_MAX);
    CUDA_CHECK(cudaMemcpy(future_pos, tmp_pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // control params
    int block = 1024;

    // we'll just iterate bucket_idx from 0..max_bucket and process if frontier non-empty.
    int max_weight = 100; // since CSR_generate uses 1..10; adjust otherwise
    int max_bucket = ((n > 0 ? (n - 1) : 0) * max_weight) / max(1, delta) + 2;

    // reset S_size and proc_gen for nodes? proc_gen uses bucket_idx checks; we don't need to reset whole array.
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_future_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
    // host loop over buckets
    int h_nextBucket = 0;
    printf("Starting delta-stepping GPU (device buckets) with delta=%d max_bucket=%d\n", delta, max_bucket);
    while (h_nextBucket != INT_MAX) {
        // read frontier size
        int h_frontier_size = get_int_from_device(d_frontier_size);
        if (h_frontier_size == 0) continue;

        // Light-phase stabilization: while frontier non-empty, process and build next_frontier (within same bucket)
        while (true) {
            // reset next_size and future_cnt and seen_gen values for this generation
            CUDA_CHECK(cudaMemcpy(d_next_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch kernel to process current frontier (light edges)
            int grid = (h_frontier_size + block - 1) / block;
            // For seen_gen we use a generation id to mark entries added to next_frontier:
            printf("\n bucket %d: ", h_nextBucket);
            printf("frontier size %d\n", h_frontier_size);
            kernel_light_step<<<grid, block>>>(
                d_frontier, h_frontier_size, n,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta, h_nextBucket,
                d_next_frontier, d_next_size,
                d_future_buckets,
                d_heavy_nodes, pos_heavy, bucket_heavy,
                d_next_bucket,
                current_gen, seen_gen);
            current_gen++;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            // get next_size
            int h_next_size = get_int_from_device(d_next_size);

            // if no new nodes in next frontier => light-phase stable
            if (h_next_size == 0) break;

            // swap frontiers and continue light-phase
            int* tmp = d_frontier; d_frontier = d_next_frontier; d_next_frontier = tmp;
            h_frontier_size = h_next_size;
            
        } // end light-phase stabilization

        int h_pos_heavy = get_int_from_device(pos_heavy);
        CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

        // Heavy-phase: process S and append future candidates

        int h_S_size = h_pos_heavy;
        if (h_S_size > 0) {
            int gridH = (h_S_size + block - 1) / block;
            kernel_heavy_step<<<gridH, block>>>(
                d_heavy_nodes, h_S_size,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta, h_nextBucket,
                d_future_buckets,
                d_next_bucket);   // passaggio extra
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }

        // 1) read nextBucket (device computed minima)
        
        CUDA_CHECK(cudaMemcpy(&h_nextBucket, d_next_bucket, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_nextBucket == INT_MAX) {
            // no future bucket (shouldn't happen if h_future_cnt>0 but safe)
            printf("\nNo next bucket found\n");
            continue;
        }

        // 2) prepare frontier counter
        int zero_int = 0;
        CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice)); // reuse d_frontier_size as device-side counter

        // 3) launch collect kernel to gather nodes of bucket h_nextBucket
        int gridC = (n + block - 1) / block;
        kernel_collect_bucket<<<gridC, block>>>(
            d_future_buckets, n,
            h_nextBucket,
            d_frontier, d_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // 4) read new frontier size
        int new_frontier_size = get_int_from_device(d_frontier_size);

        if (new_frontier_size == 0) {
            // nothing to do; reset next_bucket and continue
            int infv = INT_MAX;
            CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }

        // 5) reset d_next_bucket to INF for next round (important: do before next kernels generate new candidates)
        int infv = INT_MAX;
        CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));

        // 6) now d_frontier contains the nodes; set h_frontier_size and continue loop (light-phase)
        CUDA_CHECK(cudaMemcpy(d_frontier_size, &new_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
        h_frontier_size = new_frontier_size;
        
    } // end bucket loop

    printf("-Moving data back to host\n");
    // copy back tent
    std::vector<int> h_tent(n);
    CUDA_CHECK(cudaMemcpy(dist_h, d_tent, n * sizeof(int), cudaMemcpyDeviceToHost));
    printf("-End moving data back to host\n");

    // free
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_weights);
    cudaFree(d_tent);
    cudaFree(d_frontier); cudaFree(d_next_frontier);
    cudaFree(d_frontier_size); cudaFree(d_next_size);
    cudaFree(d_heavy_nodes); cudaFree(pos_heavy);
    cudaFree(d_future_buckets); cudaFree(d_future_cnt);
}

// ----------------- KERNELS -----------------

// --- Light edges kernel using per-block shared memory
__global__ void kernel_light_step_blocked(
    const int* frontier, int frontier_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta, int bucket_idx,
    int* next_frontier, int* next_cnt,
    int* future_buckets,
    int* d_heavy_nodes, int* pos_heavy,
    int* d_next_bucket)
{
    extern __shared__ int dist_block[]; // shared memory per-block
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * blockDim.x;
    int u_idx = block_offset + tid;

    // carica i nodi del blocco in shared memory
    if(u_idx < frontier_size){
        int u = frontier[u_idx];
        dist_block[tid] = tent[u]; 
    } else {
        dist_block[tid] = INT_MAX;
    }
    __syncthreads();

    if(u_idx < frontier_size){
        int u = frontier[u_idx];
        int du = dist_block[tid];
        if(du == INF) return;

        bool have_heavy = false;
        size_t start = row_ptr[u];
        size_t end = row_ptr[u+1];

        for(size_t e = start; e < end; ++e){
            int v = col_ind[e];
            int w = weights[e];
            if(w <= delta){
                int newd = du + w;
                /*if v is in the current offset in shared mem*/
                int old = 0;
                if(v >= block_offset && v < block_offset + blockDim.x){
                    old = atomicMin(&dist_block[v - block_offset], newd);
                } else {
                    old = atomicMin(&tent[v], newd);
                }

                if(newd < old){
                    int newBucket = newd / delta;
                    int oldBucket = atomicMin(&future_buckets[v], newBucket);
                    if(newBucket <= oldBucket){
                        if(newBucket == bucket_idx){
                            int pos = atomicAdd(next_cnt,1);
                            next_frontier[pos] = v;
                        } else if(oldBucket == INT_MAX){
                            atomicMin(d_next_bucket, newBucket);
                        }
                    }
                }
                
            } else {
                have_heavy = true;
            }
        }

        if(have_heavy){
            int pos = atomicAdd(pos_heavy,1);
            d_heavy_nodes[pos] = u;
        }
    }
    __syncthreads();

    // scrivi i valori finali shared -> globale
    if(u_idx < frontier_size){
        int u = frontier[u_idx];
        atomicMin(&tent[u], dist_block[tid]);
    }
}


void delta_stepping_gpu_shared(
    int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta, int* dist_h)
{
    // copy CSR to device
    size_t* d_row_ptr; int *d_col_ind, *d_weights;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind, m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, m * sizeof(int), cudaMemcpyHostToDevice));

    // nextBucket: device-side min bucket for future bucket to be processed
    int *d_next_bucket;
    CUDA_CHECK(cudaMalloc(&d_next_bucket, sizeof(int)));
    int infv = INT_MAX;
    CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));

    // allocate device arrays
    int* d_tent; CUDA_CHECK(cudaMalloc(&d_tent, n * sizeof(int)));
    // init tent (host->device)
    std::vector<int> h_tent_init(n, INF); 
    h_tent_init[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_tent, h_tent_init.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // frontier buffers (device)
    int *d_frontier, *d_next_frontier;
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    int *d_frontier_size; 
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    int *d_next_size; 
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    // store initial frontier = {source}
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int one = 1; 
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &one, sizeof(int), cudaMemcpyHostToDevice));  

    int* d_future_buckets; 
    CUDA_CHECK(cudaMalloc(&d_future_buckets, n * sizeof(int)));
    std::vector<int> tmp_proc2(n, INT_MAX); 
    CUDA_CHECK(cudaMemcpy(d_future_buckets, tmp_proc2.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int* d_future_cnt; 
    CUDA_CHECK(cudaMalloc(&d_future_cnt, sizeof(int)));

    // heavy nodes buffer in devide
    int *d_heavy_nodes; 
    CUDA_CHECK(cudaMalloc(&d_heavy_nodes, n* sizeof(int)));
    int *pos_heavy; 
    CUDA_CHECK(cudaMalloc(&pos_heavy, sizeof(int)));
    CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

    int *future_pos; 
    CUDA_CHECK(cudaMalloc(&future_pos, n * sizeof(int)));
    std::vector<int> tmp_pos(n, INT_MAX);
    CUDA_CHECK(cudaMemcpy(future_pos, tmp_pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // control params
    int block = 256;

    // we'll just iterate bucket_idx from 0..max_bucket and process if frontier non-empty.
    int max_weight = 100; // since CSR_generate uses 1..10; adjust otherwise
    int max_bucket = ((n > 0 ? (n - 1) : 0) * max_weight) / max(1, delta) + 2;

    // generation counter for seen_gen dedup: use increasing int
    int gen = 1;

    // reset S_size and proc_gen for nodes? proc_gen uses bucket_idx checks; we don't need to reset whole array.
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_future_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
    // host loop over buckets

    for (int bucket_idx = 0; bucket_idx < max_bucket; bucket_idx++) {
        // read frontier size
        int h_frontier_size = get_int_from_device(d_frontier_size);
        if (h_frontier_size == 0) continue;

        // Light-phase stabilization: while frontier non-empty, process and build next_frontier (within same bucket)
        while (true) {
            // reset next_size and future_cnt and seen_gen values for this generation
            CUDA_CHECK(cudaMemcpy(d_next_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch kernel to process current frontier (light edges)
            int grid = (h_frontier_size + block - 1) / block;
            // For seen_gen we use a generation id to mark entries added to next_frontier:
            int seen_gen_val = gen;
            printf("\n bucket %d: ", bucket_idx);
            #ifdef PRINT
            printf("\n bucket %d: ", bucket_idx);
            #endif
            size_t shared_bytes = block * sizeof(int);
            kernel_light_step_blocked<<<grid, block, shared_bytes>>>(
                d_frontier, h_frontier_size,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta, bucket_idx,
                d_next_frontier, d_next_size,
                d_future_buckets,
                d_heavy_nodes, pos_heavy,
                d_next_bucket);   // passaggio extra
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            //printf(" \r\n");

            // get next_size
            int h_next_size = get_int_from_device(d_next_size);
            

            // int *h_tent = (int *)malloc(n * sizeof(int));
            // CUDA_CHECK(cudaMemcpy(h_tent, d_tent, n * sizeof(int), cudaMemcpyDeviceToHost));
            // for (int i = 55; i < 56; ++i) {
            //     printf("dist[%d] = %d - ", i, h_tent[i] == INF ? -1 : h_tent[i]);
            // }

            // if no new nodes in next frontier => light-phase stable
            if (h_next_size == 0) break;

            // swap frontier arrays: d_frontier <- d_next_frontier
            printf(" Next size: %d ", h_next_size);
            CUDA_CHECK(cudaMemcpy(d_frontier, d_next_frontier, h_next_size * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_frontier_size, &h_next_size, sizeof(int), cudaMemcpyHostToDevice));
            h_frontier_size = h_next_size;
            gen++; // advance generation for next dedup round
            
        } // end light-phase stabilization
        

        int h_next_bucket = get_int_from_device(d_next_bucket);
        printf("\nNext bucket: %d\n", h_next_bucket);
        int h_future_cnt = get_int_from_device(d_future_cnt);
        printf("Future cnt before heavy: %d\n", h_future_cnt);
        //CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));
        //printf("min next bucket after light phase: %d\n", h_next_bucket);

        int h_pos_heavy = get_int_from_device(pos_heavy);
        CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

        // Heavy-phase: process S and append future candidates
        //printf("future_cap = %d\n", future_cap);
        //printf("future_cnt = %d\n", get_int_from_device(d_future_cnt));
        printf("\n Heavy step \n");
        int h_S_size = h_pos_heavy;
        if (h_S_size > 0) {
            //printf("Heavy nodes: \n");
            int gridH = (h_S_size + block - 1) / block;
            //CUDA_CHECK(cudaMemset(d_future_cnt, 0, sizeof(int)));
            size_t shared_bytes = block * sizeof(int);
            kernel_heavy_step<<<gridH, block, shared_bytes>>>(
                d_heavy_nodes, h_S_size,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta, bucket_idx,
                d_future_buckets,
                d_next_bucket);   // passaggio extra
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }
        h_next_bucket = get_int_from_device(d_next_bucket);
        printf("\nNext bucket: %d\n", h_next_bucket);
        
        // int *h_tent = (int *)malloc(n * sizeof(int));
        //     CUDA_CHECK(cudaMemcpy(h_tent, d_tent, n * sizeof(int), cudaMemcpyDeviceToHost));
        //     for (int i = 55; i < 56; ++i) {
        //         printf("dist[%d] = %d - ", i, h_tent[i] == INF ? -1 : h_tent[i]);
        //     }

        //how many future nodes i have in the next buckets all together
        h_future_cnt = get_int_from_device(d_future_cnt);
        printf("Future cnt after heavy: %d\n", h_future_cnt);
        /*if (h_future_cnt == 0) {
            int zero_int = 0; 
            CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }*/

        // 1) read nextBucket (device computed minima)
        int h_nextBucket;
        CUDA_CHECK(cudaMemcpy(&h_nextBucket, d_next_bucket, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_nextBucket == INT_MAX) {
            // no future bucket (shouldn't happen if h_future_cnt>0 but safe)
            int zero_int = 0; 
            CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice));
            printf("\nNo next bucket found\n");
            continue;
        }
        printf("Next bucket to process: %d\n", h_nextBucket);
        // 2) prepare frontier counter
        int zero_int = 0;
        CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice)); // reuse d_frontier_size as device-side counter

        // 3) launch collect kernel to gather nodes of bucket h_nextBucket
        int gridC = (n + block - 1) / block;
        int gen_val = ++gen; // advance generation for dedup
        kernel_collect_bucket<<<gridC, block>>>(
            d_future_buckets, n,
            h_nextBucket,
            d_frontier, d_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // 4) read new frontier size
        int new_frontier_size = get_int_from_device(d_frontier_size);
        // e se devo saltare un bucket perchè è vuoto e quello dopo è pieno?

        if (new_frontier_size == 0) {
            // nothing to do; reset next_bucket and continue
            int infv = INT_MAX;
            CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }

        // 5) reset d_next_bucket to INF for next round (important: do before next kernels generate new candidates)
        int infv = INT_MAX;
        CUDA_CHECK(cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice));

        // 6) now d_frontier contains the nodes; set h_frontier_size and continue loop (light-phase)
        CUDA_CHECK(cudaMemcpy(d_frontier_size, &new_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
        h_frontier_size = new_frontier_size;
        
    } // end bucket loop

    // copy back tent
    std::vector<int> h_tent(n);
    CUDA_CHECK(cudaMemcpy(dist_h, d_tent, n * sizeof(int), cudaMemcpyDeviceToHost));
    // // print a sample
    // for (int i = 0; i < min(n, 50); ++i) {
    //     printf("dist[%d] = %d\n", i, h_tent[i] == INF ? -1 : h_tent[i]);
    // }

    // free
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_weights);
    cudaFree(d_tent);
    cudaFree(d_frontier); cudaFree(d_next_frontier);
    cudaFree(d_frontier_size); cudaFree(d_next_size);
    cudaFree(d_heavy_nodes); cudaFree(pos_heavy);
    cudaFree(d_future_buckets); cudaFree(d_future_cnt);
}