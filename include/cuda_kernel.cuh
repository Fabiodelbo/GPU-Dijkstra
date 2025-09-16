

// Number of vertices in the graph
//35000 nodes are 4.56 GB of memory
#define VERTEX 20000
#define AVG_DEG 300
#define DELTA 50//for delta-stepping
#define MAX_WEIGHT 100

#define BLOCK_DIM 512

// List wrapper function callable by .cpp file.
void dijkstra_parallelize_naive(short* graph, int src, short* dist, int V);
void dijkstra_parallelize_shared(short* graph, int src, short* dist);
void dijkstra_parallelize_shared_atomic(short* graph, int src, short* dist);
int minDistance(short dist[], bool sptSet[], int V);
void delta_stepping_gpu_device_buckets(int n, int m, const size_t* h_row_ptr, const int* h_col_ind, const int* h_weights, int source, int delta, int* dist_h);
void delta_stepping_gpu_device_buckets_shared(
    int n, int m,
    const size_t* h_row_ptr,
    const int* h_col_ind,
    const int* h_weights,
    int source,
    int delta, int* dist_h);
void print_gpu_properties();
