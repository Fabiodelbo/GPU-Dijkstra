

// Number of vertices in the graph
//35000 nodes are 4.56 GB of memory
#define VERTEX 25000
#define BLOCK_DIM 1024

// List wrapper function callable by .cpp file.
// TODO: define the wrapper funtions to be used wherever it is required by other CPP files
void dijkstra_parallelize_naive(short* graph, int src, short* dist, int V);
void dijkstra_parallelize_shared(short* graph, int src, short* dist);
int minDistance(short dist[], bool sptSet[], int V);




