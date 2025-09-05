

// Number of vertices in the graph
//35000 nodes are 4.56 GB of memory
#define VERTEX 30000
<<<<<<< HEAD
#define AVG_DEG 10000
#define DELTA (AVG_DEG/2)//for delta-stepping

=======
>>>>>>> 798d618 (deleting bin file and updating .gitignore)
#define BLOCK_DIM 512

// List wrapper function callable by .cpp file.
// TODO: define the wrapper funtions to be used wherever it is required by other CPP files
void dijkstra_parallelize_naive(short* graph, int src, short* dist, int V);
void dijkstra_parallelize_shared(short* graph, int src, short* dist);
void dijkstra_parallelize_shared_atomic(short* graph, int src, short* dist);
int minDistance(short dist[], bool sptSet[], int V);




