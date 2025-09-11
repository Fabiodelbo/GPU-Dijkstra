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
constexpr int INF = 0x3f3f3f3f;

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
    const int* frontier, int frontier_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta, int bucket_idx,
    int* next_frontier, int* next_cnt, int next_gen,
    int* future_nodes, int* future_buckets, int* future_cnt, int future_cap, int* future_pos,
    int* seen_gen, int seen_gen_val, int* proc_gen,
    int* d_heavy_nodes, int* pos_heavy,
    int* d_next_bucket)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];
    if (u < 0) return;
    

    // Try to claim u for this bucket. Only the thread that sees prev == -1 does the work.
    // int prevp = atomicCAS(&proc_gen[u], -1, bucket_idx);
    // if (prevp != -1) {
    //     // another thread already processed u for this bucket -> skip
    //     return;
    // }
    //printf("frontier [%d] = %d ", tid, frontier[tid]);
    int du = tent[u];
    if (du == INF) return;

    bool have_heavy = false;

    size_t start = row_ptr[u];
    size_t end   = row_ptr[u+1];

    //printf(" %d ", u);
    for (size_t e = start; e < end; ++e) {
        int v = col_ind[e];
        int w = weights[e];
        if (w <= delta) {
            //printf("%d ", v);
            int newd = du + w;
            // atomicMin returns old value; if newd < old then current thread won the update
            int old = atomicMin(&tent[v], newd);
            if (newd < old) {
                int newBucket = newd / delta;
                if (newBucket == bucket_idx) {
                    // insert in next_frontier (dedup via seen_gen)
                    //printf("-%d-", seen_gen_val);
                    int prev = atomicCAS(&seen_gen[v], -1, seen_gen_val);
                    //printf(" (%d) ", prev);
                    //printf(" + %d ", v);

                    if (prev == -1 ) {
                        int pos = atomicAdd(next_cnt, 1);
                        // safety: ensure we don't write out of bounds (next_frontier sized n)
                        // here we assume next_frontier capacity >= n; otherwise check pos < next_cap
                        next_frontier[pos] = v;
                        //printf(" + %d ", v);
                    }
                } else {
                    // goes to future arrays (bucket in future)
                    atomicMin(d_next_bucket, newBucket);
                    int old = atomicCAS(&future_pos[v], INT_MAX, -2);  // prova a riservare

                    if (old == INT_MAX) {
                        // Sei il primo che inserisce v
                        int pos = atomicAdd(future_cnt, 1);
                        if (pos < future_cap) {
                            future_nodes[pos]   = v;
                            future_buckets[pos] = newBucket;
                            // scrivi la posizione definitiva
                            future_pos[v] = pos;
                        } else {
                            // fallback: spazio esaurito
                            future_pos[v] = INT_MAX;  // reset così altri possono riprovare
                        }
                    } else {
                        // v era già stato inserito
                        //printf("Aggiorno bucket di %d da %d a %d\n", v, future_buckets[old], newBucket);
                        int pos = future_pos[v];  // lettura consistente
                        int oldb = atomicMin(&future_buckets[pos], newBucket);
                        if (newBucket < oldb) {
                            atomicMin(d_next_bucket, newBucket);
                        }
                    }
                }
            }
        } else {
            // heavy edge: mark that u has heavy outgoing edges; heavy edges handled later
            have_heavy = true;
            // note: do NOT update tent[v] or seen_gen here; heavy kernel will do that
        }
    }

    if (have_heavy) {
        int pos = atomicAdd(pos_heavy, 1);
        if (pos >= 0 && pos < (int)gridDim.x * blockDim.x) {
            // d_heavy_nodes buffer was allocated with size >= n, so this should be safe
            d_heavy_nodes[pos] = u;
        } else {
            // out-of-space (unlikely if d_heavy_nodes has size n)
        }
    }
}

// --- Heavy edges kernel (process nodes in S)
__global__ void kernel_heavy_step(
    const int* S, int S_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta,
    int* future_nodes, int* future_buckets, int* future_cnt, int future_cap, int* future_pos,
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
    for (size_t e = start; e < end; ++e) {
        int v = col_ind[e];
        int w = weights[e];
        if (w > delta) {
            int newd = du + w;
            int old = atomicMin(&tent[v], newd);
            if (newd < old) {
                int newBucket = newd / delta;

                // prova a riservare slot per v
                int oldp = atomicCAS(&future_pos[v], INT_MAX, -2);

                if (oldp == INT_MAX) {
                    // primo thread che inserisce v
                    int pos = atomicAdd(future_cnt, 1);
                    if (pos < future_cap) {
                        future_nodes[pos]   = v;
                        future_buckets[pos] = newBucket;
                        future_pos[v] = pos;
                    } else {
                        // out of space, reset
                        future_pos[v] = INT_MAX;
                    }
                } else {
                    // già inserito, aggiorna solo il bucket
                    int pos = future_pos[v];
                    int oldb = atomicMin(&future_buckets[pos], newBucket);
                    if (newBucket < oldb) {
                        atomicMin(d_next_bucket, newBucket);
                    }
                }

                // aggiornamento globale next bucket
                atomicMin(d_next_bucket, newBucket);
            }
        }
    }
}


// --- Collect nodes belonging to targetBucket from future arrays into frontier.
// Note: future_buckets entries are consumed (set to -1) when collected.
__global__ void kernel_collect_bucket(
    int* future_nodes, int* future_buckets, int future_cnt,
    int targetBucket,
    int* frontier, int* frontier_cnt,
    int* seen_gen, int gen)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= future_cnt) return;
    
    int b = future_buckets[idx];
    //printf("Target bucket %d -", targetBucket);
    //printf("Thread %d processing idx %d with bucket %d\n", idx, idx, b);
    if (b != targetBucket) return;

    int v = future_nodes[idx];
    //printf(" %d ", v);
    //printf(" bucket %d -", b);
    // try to mark seen for this generation
    int prev = atomicCAS(&seen_gen[v], -1, gen);
    if (prev == -1) {
        //printf("Adding %d to frontier\n", v);
        int pos = atomicAdd(frontier_cnt, 1);
        frontier[pos] = v;
    }
    // mark consumed (so future collects ignore it)
    atomicExch(&future_buckets[idx], -1);
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

    // allocate device arrays
    int* d_tent; CUDA_CHECK(cudaMalloc(&d_tent, n * sizeof(int)));
    // init tent (host->device)
    std::vector<int> h_tent_init(n, INF); h_tent_init[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_tent, h_tent_init.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // frontier buffers (device)
    int *d_frontier, *d_next_frontier;
    CUDA_CHECK(cudaMalloc(&d_frontier, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, m * sizeof(int)));
    int *d_frontier_size; CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    int *d_next_size; CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    // store initial frontier = {source}
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int one = 1; CUDA_CHECK(cudaMemcpy(d_frontier_size, &one, sizeof(int), cudaMemcpyHostToDevice));

    // seen_gen: for dedup of next_frontier; initialize to -1
    int* d_seen_gen; CUDA_CHECK(cudaMalloc(&d_seen_gen, n * sizeof(int)));
    //all nodes unseen initially
    

    // proc_gen: used to ensure S gets only unique u per bucket
    int* d_proc_gen; CUDA_CHECK(cudaMalloc(&d_proc_gen, n * sizeof(int)));
    std::vector<int> tmp_proc(n, -1); CUDA_CHECK(cudaMemcpy(d_proc_gen, tmp_proc.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // future buffers (candidates for future buckets)
    int future_cap = max(2048, m); // safe upper bound (to be tuned)
    int* d_future_nodes; CUDA_CHECK(cudaMalloc(&d_future_nodes, future_cap * sizeof(int)));
    int* d_future_buckets; CUDA_CHECK(cudaMalloc(&d_future_buckets, future_cap * sizeof(int)));
    int* d_future_cnt; CUDA_CHECK(cudaMalloc(&d_future_cnt, sizeof(int)));

    // heavy nodes buffer in devide
    int *d_heavy_nodes; CUDA_CHECK(cudaMalloc(&d_heavy_nodes, n* sizeof(int)));
    int *pos_heavy; CUDA_CHECK(cudaMalloc(&pos_heavy, sizeof(int)));
    CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

    int *future_pos; CUDA_CHECK(cudaMalloc(&future_pos, n * sizeof(int)));
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
    for (int bucket_idx = 0; bucket_idx < max_bucket; ++bucket_idx) {
        // read frontier size
        int h_frontier_size = get_int_from_device(d_frontier_size);
        if (h_frontier_size == 0) continue;

        
        // Light-phase stabilization: while frontier non-empty, process and build next_frontier (within same bucket)
        while (true) {
            // reset next_size and future_cnt and seen_gen values for this generation
            CUDA_CHECK(cudaMemcpy(d_next_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
            std::vector<int> tmp_seen(n, -1); 
            CUDA_CHECK(cudaMemcpy(d_seen_gen, tmp_seen.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch kernel to process current frontier (light edges)
            int grid = (h_frontier_size + block - 1) / block;
            // For seen_gen we use a generation id to mark entries added to next_frontier:
            int seen_gen_val = gen;
            //printf("\n bucket %d: ", bucket_idx);
           kernel_light_step<<<grid, block>>>(
                d_frontier, h_frontier_size,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta, bucket_idx,
                d_next_frontier, d_next_size, gen,
                d_future_nodes, d_future_buckets, d_future_cnt, future_cap, future_pos,
                d_seen_gen, seen_gen_val, d_proc_gen,
                d_heavy_nodes, pos_heavy,
                d_next_bucket);   // passaggio extra
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            //printf(" \r\n");

            // get next_size
            int h_next_size = get_int_from_device(d_next_size);
            // if no new nodes in next frontier => light-phase stable
            // int *h_tent = (int *)malloc(n * sizeof(int));
            // CUDA_CHECK(cudaMemcpy(h_tent, d_tent, n * sizeof(int), cudaMemcpyDeviceToHost));
            // for (int i = 0; i < n; ++i) {
            //     printf("dist[%d] = %d - ", i, h_tent[i] == INF ? -1 : h_tent[i]);
            // }
            if (h_next_size == 0) break;

            // swap frontier arrays: d_frontier <- d_next_frontier
            CUDA_CHECK(cudaMemcpy(d_frontier, d_next_frontier, h_next_size * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_frontier_size, &h_next_size, sizeof(int), cudaMemcpyHostToDevice));
            h_frontier_size = h_next_size;
            gen++; // advance generation for next dedup round
            
        } // end light-phase stabilization
        

        int h_next_bucket = get_int_from_device(d_next_bucket);
        //CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));
        //printf("min next bucket after light phase: %d\n", h_next_bucket);

        int h_pos_heavy = get_int_from_device(pos_heavy);
        CUDA_CHECK(cudaMemset(pos_heavy, 0, sizeof(int)));

        // Heavy-phase: process S and append future candidates
        printf("future_cap = %d\n", future_cap);
        printf("future_cnt = %d\n", get_int_from_device(d_future_cnt));
        int h_S_size = h_pos_heavy;
        if (h_S_size > 0) {
            //printf("Heavy nodes: \n");
            int gridH = (h_S_size + block - 1) / block;
            //CUDA_CHECK(cudaMemset(d_future_cnt, 0, sizeof(int)));
            kernel_heavy_step<<<gridH, block>>>(
                d_heavy_nodes, h_S_size,
                d_row_ptr, d_col_ind, d_weights,
                d_tent, delta,
                d_future_nodes, d_future_buckets, d_future_cnt, future_cap, future_pos,
                d_next_bucket);   // passaggio extra
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
        }

        int h_future_cnt = get_int_from_device(d_future_cnt);
        if (h_future_cnt == 0) {
            int zero_int = 0; CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }

        // 1) read nextBucket (device computed minima)
        int h_nextBucket;
        CUDA_CHECK(cudaMemcpy(&h_nextBucket, d_next_bucket, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_nextBucket == INT_MAX) {
            // no future bucket (shouldn't happen if h_future_cnt>0 but safe)
            int zero_int = 0; CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice));
            continue;
        }

        // 2) prepare frontier counter
        int zero_int = 0;
        CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero_int, sizeof(int), cudaMemcpyHostToDevice)); // reuse d_frontier_size as device-side counter

        std::vector<int> tmp_seen_1(n, -1); 
        CUDA_CHECK(cudaMemcpy(d_seen_gen, tmp_seen_1.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        // 3) launch collect kernel to gather nodes of bucket h_nextBucket
        int gridC = (h_future_cnt + block - 1) / block;
        int gen_val = ++gen; // advance generation for dedup
        kernel_collect_bucket<<<gridC, block>>>(
            d_future_nodes, d_future_buckets, h_future_cnt,
            h_nextBucket,
            d_frontier, d_frontier_size,
            d_seen_gen, gen_val);
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
    cudaFree(d_seen_gen); cudaFree(d_proc_gen);
    cudaFree(d_heavy_nodes); cudaFree(pos_heavy);
    cudaFree(d_future_nodes); cudaFree(d_future_buckets); cudaFree(d_future_cnt);
}

// ----------------- KERNELS -----------------

// Light edges: usa shared memory per next_frontier e dedup locale (sicuro)
__global__ void kernel_light_step_shared(
    const int* frontier, int frontier_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta, int bucket_idx,
    int* next_frontier, int* next_cnt, int next_gen,
    int* future_nodes, int* future_buckets, int* future_cnt, int future_cap,
    int* seen_gen, int seen_gen_val,
    int* proc_gen, int* d_heavy_nodes, int* pos_heavy,
    int* d_next_bucket)
{
    extern __shared__ int smem[];
    // layout smem:
    // [0 .. B-1]        -> s_next_frontier
    // [B .. 2B-1]       -> s_future_nodes
    // [2B .. 3B-1]      -> s_future_buckets
    // [3B .. 4B-1]      -> s_bucket_vals (per-thread local_next_bucket copy)
    int B = blockDim.x;
    int* s_next_frontier   = &smem[0];
    int* s_future_nodes    = &smem[B];
    int* s_future_buckets  = &smem[2*B];
    int* s_bucket_vals     = &smem[3*B];

    // shared counters (not in smem array, declared separately)
    __shared__ int s_nf_counter;   // number of entries in s_next_frontier (reserve via atomic)
    __shared__ int s_fut_counter;  // number of entries in s_future_* (reserve via atomic)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // init shared counters and bucket vals
    if (tid == 0) {
        s_nf_counter = 0;
        s_fut_counter = 0;
    }
    // init per-thread bucket mem to INT_MAX
    s_bucket_vals[tid] = INT_MAX;
    __syncthreads();

    int local_next_bucket = INT_MAX;

    if (gid < frontier_size) {
        int u = frontier[gid];
        if (u >= 0) {
            // try mark processed for this bucket (we don't check return)
            if(atomicCAS(&proc_gen[u], -1, bucket_idx) == -1) {
                
                // tent of the node in exam
                int du = tent[u];
                if (du != INF) {
                    //if it is reached by some thread, it means that it has been already processed
                    size_t start = row_ptr[u];
                    size_t end = row_ptr[u+1];
                    bool has_heavy = false;

                    for (size_t e = start; e < end; ++e) {
                        int v = col_ind[e];
                        int w = weights[e];
                        
                            if (w <= delta) {
                                // try to mark seen: do CAS first
                                int newd = du + w;
                                printf("%d ", v);
                                int old = atomicMin(&tent[v], newd);
                                if (newd < old) {
                                    int newBucket = newd / delta;
                                    // accumulate candidate for block reduction
                                    local_next_bucket = min(local_next_bucket, newBucket);
                                    //check if the node is already been added to the next frontier by some other thread
                                    int cas_res = atomicCAS(&seen_gen[v], -1, seen_gen_val);
                                    if (cas_res == -1) {
                                        // reserved a cell in shared new frontier, in that way every thread will write the node atomically in the correct cell
                                        if(newBucket == bucket_idx){
                                            int slot = atomicAdd(&s_nf_counter, 1);
                                            if (slot < B) {
                                                s_next_frontier[slot] = v;
                                            } else {
                                                // shared buffer full: fallback => write directly in global next_frontier
                                                int pos = atomicAdd(next_cnt, 1);
                                                next_frontier[pos] = v;
                                            }
                                        }
                                        else{
                                            // not in this bucket, write directly to global future arrays
                                            int pos = atomicAdd(future_cnt, 1);
                                            if (pos < future_cap) {
                                                future_nodes[pos] = v;
                                                future_buckets[pos] = newBucket;
                                            }
                                        }
                                    }
                                }
                                // else: already seen by some thread => ignored
                            } else {
                                // heavy edge -> collect in per-block shared fut buffer
                                has_heavy = true;
                                // int newd = du + w;
                                // int newBucket = newd / delta;
                                // int slot = atomicAdd(&s_fut_counter, 1);
                                // if (slot < B) {
                                //     s_future_nodes[slot] = v;
                                //     s_future_buckets[slot] = newBucket;
                                // } else {
                                //     // shared buffer full: fallback => write directly to global future arrays
                                //     int pos = atomicAdd(future_cnt, 1);
                                //     if (pos < future_cap) {
                                //         future_nodes[pos] = v;
                                //         future_buckets[pos] = newBucket;
                                //     }
                                // }
                            }
                        
                    } // for edges
                    // u has heavy nodes so they have to be processed in heavy step
                    if (has_heavy) {
                        int pos = atomicAdd(pos_heavy, 1);
                        d_heavy_nodes[pos] = u;
                    }
                } // du != INF
            }
        } // u >= 0
    } // gid < frontier_size

    __syncthreads();

    // commit local shared next_frontier (if any) to global
    int shared_nf = s_nf_counter;   // read once
    for (int i = tid; i < shared_nf; i += blockDim.x) {
        int pos = atomicAdd(next_cnt, 1);
        next_frontier[pos] = s_next_frontier[i];
    }

    // commit local shared future arrays to global (if any slots were not flushed earlier)
    int shared_fut = s_fut_counter;
    for (int i = tid; i < shared_fut; i += blockDim.x) {
        int pos = atomicAdd(future_cnt, 1);
        if (pos < future_cap) {
            future_nodes[pos] = s_future_nodes[i];
            future_buckets[pos] = s_future_buckets[i];
        }
    }

    // store local_next_bucket into s_bucket_vals for reduction
    s_bucket_vals[tid] = local_next_bucket;
    __syncthreads();

    // block-level min reduction on s_bucket_vals (standard iterative halving)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = s_bucket_vals[tid];
            int b = s_bucket_vals[tid + stride];
            s_bucket_vals[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int blockMin = s_bucket_vals[0];
        if (blockMin != INT_MAX) {
            atomicMin(d_next_bucket, blockMin);
        }
    }
}

__global__ void kernel_heavy_step_shared(
    const int* S, int S_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta,
    int* future_nodes, int* future_buckets, int* future_cnt, int future_cap,
    int* d_next_bucket)
{
    extern __shared__ int smem[]; // layout: [0..B-1]=s_future_nodes, [B..2B-1]=s_future_buckets, [2B..3B-1]=s_bucket_vals
    int B = blockDim.x;
    int* s_future_nodes   = &smem[0];
    int* s_future_buckets = &smem[B];
    int* s_bucket_vals    = &smem[2*B];

    __shared__ int s_fut_counter;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid == 0) s_fut_counter = 0;
    // init per-thread bucket val
    s_bucket_vals[tid] = INT_MAX;
    __syncthreads();

    int local_next_bucket = INT_MAX;

    if (gid < S_size) {
        int u = S[gid];
        if (u >= 0) {
            int du = tent[u];
            if (du != INF) {
                size_t start = row_ptr[u];
                size_t end = row_ptr[u+1];
                for (size_t e = start; e < end; ++e) {
                    int v = col_ind[e];
                    int w = weights[e];
                    if (w > delta) {
                        int newd = du + w;
                        int old = atomicMin(&tent[v], newd);
                        if (newd < old) {
                            int newBucket = newd / delta;
                            if (newBucket < local_next_bucket) local_next_bucket = newBucket;

                            // reserve slot in shared fut buffer
                            int slot = atomicAdd(&s_fut_counter, 1);
                            if (slot < B) {
                                s_future_nodes[slot] = v;
                                s_future_buckets[slot] = newBucket;
                            } else {
                                // fallback: write directly to global future arrays
                                int pos = atomicAdd(future_cnt, 1);
                                if (pos < future_cap) {
                                    future_nodes[pos] = v;
                                    future_buckets[pos] = newBucket;
                                } else {
                                    // out-of-space: drop safely (or record error)
                                    // You could optionally store an error counter (atomicAdd) for diagnostics.
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // flush shared future to global
    int shared_fut = s_fut_counter;
    for (int i = tid; i < shared_fut; i += blockDim.x) {
        int pos = atomicAdd(future_cnt, 1);
        if (pos < future_cap) {
            future_nodes[pos] = s_future_nodes[i];
            future_buckets[pos] = s_future_buckets[i];
        } else {
            // diagnostic: out-of-space
        }
    }

    // per-block min reduction on local_next_bucket
    s_bucket_vals[tid] = local_next_bucket;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = s_bucket_vals[tid];
            int b = s_bucket_vals[tid + stride];
            s_bucket_vals[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int blockMin = s_bucket_vals[0];
        if (blockMin != INT_MAX) atomicMin(d_next_bucket, blockMin);
    }
}

// debug-heavy: stampa eventi per capire perché non scrive future_nodes
__global__ void kernel_heavy_debug(
    const int* S, int S_size,
    const size_t* row_ptr, const int* col_ind, const int* weights,
    int* tent, int delta,
    int* future_nodes, int* future_buckets, int* future_cnt, int future_cap,
    int* d_next_bucket)
{
    extern __shared__ int smem[]; // layout: [0..B-1] nodes, [B..2B-1] buckets, [2B..3B-1] vals
    int B = blockDim.x;
    int* s_future_nodes   = &smem[0];
    int* s_future_buckets = &smem[B];
    int* s_bucket_vals    = &smem[2*B];

    __shared__ int s_fut_counter;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid == 0) s_fut_counter = 0;
    s_bucket_vals[tid] = INT_MAX;
    __syncthreads();

    int local_next_bucket = INT_MAX;

    if (gid < S_size) {
        int u = S[gid];
        int du = tent[u];
        if (du != INF) {
            size_t start = row_ptr[u];
            size_t end = row_ptr[u+1];
            for (size_t e = start; e < end; ++e) {
                int v = col_ind[e];
                int w = weights[e];
                if (w > delta) {
                    int newd = du + w;
                    int old = atomicMin(&tent[v], newd);
                    if (newd < old) {
                        printf("[HEAVY-THREAD] gid=%d u=%d v=%d w=%d du=%d newd=%d old=%d\n",
                               gid, u, v, w, du, newd, old);
                        int newBucket = newd / delta;
                        if (newBucket < local_next_bucket) local_next_bucket = newBucket;

                        int slot = atomicAdd(&s_fut_counter, 1);
                        if (slot < B) {
                            s_future_nodes[slot] = v;
                            s_future_buckets[slot] = newBucket;
                            printf("[HEAVY-THREAD] gid=%d u=%d v=%d w=%d du=%d newd=%d old=%d slot_shared=%d\n",
                                   gid, u, v, w, du, newd, old, slot);
                        } else {
                            // fallback globale
                            int pos = atomicAdd(future_cnt, 1);
                            if (pos < future_cap) {
                                future_nodes[pos] = v;
                                future_buckets[pos] = newBucket;
                                printf("[HEAVY-THREAD] gid=%d u=%d v=%d w=%d du=%d newd=%d old=%d pos_global=%d (fallback)\n",
                                       gid, u, v, w, du, newd, old, pos);
                            } else {
                                printf("[HEAVY-THREAD] gid=%d u=%d -> OUT-OF-SPACE future_cap=%d\n", gid, u, future_cap);
                            }
                        }
                    } // newd < old
                } // w > delta
            } // for edges
        } // du!=INF
    }

    __syncthreads();

    // flush shared to global
    int shared_fut = s_fut_counter;
    for (int i = tid; i < shared_fut; i += blockDim.x) {
        int pos = atomicAdd(future_cnt, 1);
        if (pos < future_cap) {
            future_nodes[pos] = s_future_nodes[i];
            future_buckets[pos] = s_future_buckets[i];
            if (tid == 0) printf("[HEAVY-FLUSH] flushed %d entries (pos start maybe)\n", shared_fut);
        } else {
            if (tid == 0) printf("[HEAVY-FLUSH] flush skipped pos=%d >= future_cap=%d\n", pos, future_cap);
        }
    }

    // reduction per-block (facoltativo per debug)
    s_bucket_vals[tid] = local_next_bucket;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = s_bucket_vals[tid];
            int b = s_bucket_vals[tid + stride];
            s_bucket_vals[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int blockMin = s_bucket_vals[0];
        if (blockMin != INT_MAX) atomicMin(d_next_bucket, blockMin);
    }
}

__global__ void kernel_collect_bucket_shared(
    const int* future_nodes, int* future_buckets, int future_cnt,
    int targetBucket,
    int* frontier, int* frontier_cnt,
    int* seen_gen, int gen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= future_cnt) return;
    //printf("idx: %d, future_cnt: %d, targetBucket: %d\n", idx, future_cnt, targetBucket);
    int b = future_buckets[idx];
    if (b != targetBucket) return;
    int v = future_nodes[idx];
    int prev = atomicCAS(&seen_gen[v], -1, gen);
    if (prev == -1) {
       // printf("Adding node %d to frontier\n", v);
        int pos = atomicAdd(frontier_cnt, 1);
        frontier[pos] = v;
    }
    // mark consumed so we won't consider it again
    atomicExch(&future_buckets[idx], -1);
}

void delta_stepping_gpu_shared(
    int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta,
    int* dist_h)
{
    // --- device CSR arrays
    size_t *d_row_ptr; int *d_col_ind, *d_weights;
    cudaMalloc(&d_row_ptr, (n+1)*sizeof(size_t));
    cudaMalloc(&d_col_ind, m*sizeof(int));
    cudaMalloc(&d_weights, m*sizeof(int));
    cudaMemcpy(d_row_ptr, h_row_ptr, (n+1)*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, m*sizeof(int), cudaMemcpyHostToDevice);

    // --- tent array
    int* d_tent; cudaMalloc(&d_tent, n*sizeof(int));
    std::vector<int> h_tent(n, INF); h_tent[source]=0;
    cudaMemcpy(d_tent, h_tent.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // --- frontier
    int* d_frontier; cudaMalloc(&d_frontier, m*sizeof(int));
    int* d_frontier_next; cudaMalloc(&d_frontier_next, m*sizeof(int));
    int* d_frontier_size; cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
    int one=1; cudaMemcpy(d_frontier_size,&one,sizeof(int),cudaMemcpyHostToDevice);

    // --- future buffers
    int future_cap = max(1024,m);
    int *d_future_nodes, *d_future_buckets, *d_future_cnt;
    cudaMalloc(&d_future_nodes, future_cap*sizeof(int));
    cudaMalloc(&d_future_buckets, future_cap*sizeof(int));
    cudaMalloc(&d_future_cnt, sizeof(int));

    // --- heavy buffer
    int *d_heavy_nodes, *d_pos_heavy;
    cudaMalloc(&d_heavy_nodes, n*sizeof(int));
    cudaMalloc(&d_pos_heavy, sizeof(int));
    cudaMemset(d_pos_heavy, 0, sizeof(int));

    // --- dedup arrays
    int *d_seen_gen, *d_proc_gen;
    cudaMalloc(&d_seen_gen, n*sizeof(int));
    cudaMemset(d_seen_gen,-1,n*sizeof(int));
    cudaMalloc(&d_proc_gen, n*sizeof(int));
    cudaMemset(d_proc_gen,-1,n*sizeof(int));

    // --- next bucket
    int *d_next_bucket; cudaMalloc(&d_next_bucket, sizeof(int));
    int infv = INT_MAX; cudaMemcpy(d_next_bucket, &infv, sizeof(int), cudaMemcpyHostToDevice);

    // --- kernel config
    int block=512;
    int grid = (n+block-1)/block;

    // --- max bucket
    int max_weight = 100; 
    int max_bucket = ((n>0?n-1:0)*max_weight)/max(1,delta)+2;
    int gen=1;

    for(int bucket_idx=0; bucket_idx<max_bucket; bucket_idx++){
        int h_front_size = get_int_from_device(d_frontier_size);
        if(h_front_size==0) continue;

        int zero=0;

        while(true){
            cudaMemcpy(d_future_cnt,&zero,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_frontier_size,&zero,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_next_bucket,&infv,sizeof(int),cudaMemcpyHostToDevice);
            printf("bucket %d: ", bucket_idx);
            size_t shared_mem_bytes = sizeof(int) * (10 * block); // s_next_frontier + s_future_nodes + s_future_buckets
            kernel_light_step_shared<<<grid,block,shared_mem_bytes>>>(
                d_frontier, h_front_size,
                d_row_ptr,d_col_ind,d_weights,
                d_tent, delta, bucket_idx,
                d_frontier_next, d_frontier_size, gen,
                d_future_nodes,d_future_buckets,d_future_cnt,future_cap,
                d_seen_gen,gen, d_proc_gen,
                d_heavy_nodes,d_pos_heavy,
                d_next_bucket);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            int *tmp = d_frontier; d_frontier = d_frontier_next; d_frontier_next = tmp;

            h_front_size = get_int_from_device(d_frontier_size);
            if(h_front_size==0) break;
            printf("  \r\n");
        }
        
        int h_pos_heavy = get_int_from_device(d_pos_heavy);
        cudaMemset(d_pos_heavy,0,sizeof(int));

        // int h_future_cnt = get_int_from_device(d_future_cnt);
        // int h_tent9; cudaMemcpy(&h_tent9, d_tent + 9, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("PRE-HEAVY: pos_heavy=%d future_cnt=%d tent[9]=%d\n", h_pos_heavy, h_future_cnt, h_tent9);

        if(h_pos_heavy>0){
            //cudaMemset(d_future_cnt,0,sizeof(int));

            // std::vector<int> h_heavy(h_pos_heavy);
            // cudaMemcpy(h_heavy.data(), d_heavy_nodes, h_pos_heavy * sizeof(int), cudaMemcpyDeviceToHost);
            // printf("HEAVY NODES:");
            // for (int i = 0; i < h_pos_heavy; ++i) {
            //     int u = h_heavy[i];
            //     int du; cudaMemcpy(&du, d_tent + u, sizeof(int), cudaMemcpyDeviceToHost);
            //     printf(" [%d: tent=%d]", u, du);
            // }
            // printf("\n");

            size_t shared_mem_bytes = sizeof(int) * (10 * block);
            int gridH = (h_pos_heavy+block-1)/block;
            kernel_heavy_step_shared<<<gridH,block,shared_mem_bytes>>>(
                d_heavy_nodes,h_pos_heavy,
                d_row_ptr,d_col_ind,d_weights,
                d_tent, delta,
                d_future_nodes,d_future_buckets,d_future_cnt,future_cap,
                d_next_bucket);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            int h_future_cnt_after = get_int_from_device(d_future_cnt);
            printf("POST-HEAVY: future_cnt=%d\n", h_future_cnt_after);
            if (h_future_cnt_after > 0) {
                std::vector<int> h_future_nodes(std::min(h_future_cnt_after, 16));
                cudaMemcpy(h_future_nodes.data(), d_future_nodes, h_future_nodes.size()*sizeof(int), cudaMemcpyDeviceToHost);
                printf("first future nodes:");
                for (int i=0;i<h_future_nodes.size();++i) printf(" %d", h_future_nodes[i]);
                printf("\n");
            }
        }
        // int h_future_cnt_after = get_int_from_device(d_future_cnt);
        // int h_tent9_after; cudaMemcpy(&h_tent9_after, d_tent + 9, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("POST-HEAVY: future_cnt=%d tent[9]=%d\n", h_future_cnt_after, h_tent9_after);
        
        // int host_tent9, host_seen9, host_proc9, host_nextbucket;
        // cudaMemcpy(&host_tent9, d_tent + 9, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&host_seen9, d_seen_gen + 9, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&host_proc9, d_proc_gen + 9, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&host_nextbucket, d_next_bucket, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("BEFORE collect: tent[9]=%d seen[9]=%d proc[9]=%d nextBucket=%d\n",
        //     host_tent9, host_seen9, host_proc9, host_nextbucket);

        int h_future_cnt = get_int_from_device(d_future_cnt);
        printf("future_cnt=%d\n", h_future_cnt);
        if(h_future_cnt==0){
            int zero_int=0; cudaMemcpy(d_frontier_size,&zero_int,sizeof(int),cudaMemcpyHostToDevice);
            continue;
        }

        // 1) leggi next bucket
        int h_nextBucket;
        cudaMemcpy(&h_nextBucket,d_next_bucket,sizeof(int),cudaMemcpyDeviceToHost);
        printf("h_nextBucket=%d\n", h_nextBucket);
        if(h_nextBucket==INT_MAX) continue;

        // 2) prepara contatore frontier
        cudaMemcpy(d_frontier_size,&zero,sizeof(int),cudaMemcpyHostToDevice);

        // 3) raccogli nodi del prossimo bucket (host-side dedup via gen)
        int gridC = (h_future_cnt+block-1)/block;
        printf("gridC=%d block=%d\n", gridC, block);
        
        kernel_collect_bucket_shared<<<gridC,block>>>(
            d_future_nodes,d_future_buckets,h_future_cnt,
            h_nextBucket,
            d_frontier,d_frontier_size,
            d_seen_gen,++gen);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        h_front_size = get_int_from_device(d_frontier_size);
        printf("h_front_size=%d\n", h_front_size);
        if(h_front_size==0){
            cudaMemcpy(d_next_bucket,&infv,sizeof(int),cudaMemcpyHostToDevice);
            continue;
        }

        cudaMemcpy(d_frontier_size,&h_front_size,sizeof(int),cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dist_h,d_tent,n*sizeof(int),cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_weights);
    cudaFree(d_tent); cudaFree(d_frontier); cudaFree(d_frontier_size);
    cudaFree(d_heavy_nodes); cudaFree(d_pos_heavy);
    cudaFree(d_future_nodes); cudaFree(d_future_buckets); cudaFree(d_future_cnt);
    cudaFree(d_seen_gen); cudaFree(d_proc_gen);
    cudaFree(d_next_bucket);
}