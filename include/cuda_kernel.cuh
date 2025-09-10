

// Number of vertices in the graph
//35000 nodes are 4.56 GB of memory
#define VERTEX 100
#define AVG_DEG 10
#define DELTA 50//for delta-stepping

#define BLOCK_DIM 512

// List wrapper function callable by .cpp file.
// TODO: define the wrapper funtions to be used wherever it is required by other CPP files
void dijkstra_parallelize_naive(short* graph, int src, short* dist, int V);
void dijkstra_parallelize_shared(short* graph, int src, short* dist);
void dijkstra_parallelize_shared_atomic(short* graph, int src, short* dist);
int minDistance(short dist[], bool sptSet[], int V);
void delta_stepping_gpu_device_buckets(int n, int m, const size_t* h_row_ptr, const int* h_col_ind, const int* h_weights, int source, int delta, int* dist_h);
void delta_stepping_gpu_shared(
    int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta,
    int* dist_h);
void delta_stepping_gpu_persistent_full(int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta,
    int* dist_h);
