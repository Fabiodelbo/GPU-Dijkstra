// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include <limits.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <time.h>
#include <chrono>

#include <vector>
#include <set>
#include "csr.cpp"

using namespace std;

void printSolution(short dist[], int n);
void dijkstra(short* graph, int src, short* dist);
void graph_generator(short* graph, int vertex);
void compare_dist(short *dist_cpu, short *dist_gpu);
void compare_dist_tent(vector<int> dist_cpu, short *dist_gpu);
void compare_dist_csr(vector<int> dist_cpu, int *dist_gpu);
void compare_dist_delta_dijkstra(int* dist_cpu, short *dist_gpu);


// driver program to test above function
int main(int argc, char** argv)
{
    int algorithm;

    if(argc > 2){
        printf("Error too much argument, args: %d", argc);
        exit(EXIT_FAILURE);
    }
    algorithm = strtol(argv[1], NULL, 10);
    printf("algo is: %d", algorithm);

    // costruiamo CSR
    CSRGraph G(VERTEX, AVG_DEG);
    CSRGraph G_void(VERTEX, AVG_DEG);
    int delta = DELTA;
    

    switch(algorithm){
        case 0 :{
                short *gen_graph = (short *)malloc(VERTEX*VERTEX*sizeof(short*));
                short *dist_cpu = (short *)malloc(VERTEX*sizeof(short*));
                short *dist_gpu = (short *)malloc(VERTEX*sizeof(short*));

                if(gen_graph == NULL || dist_cpu == NULL || dist_gpu == NULL){
                printf("Error during memory allocation");
                exit(EXIT_FAILURE);
                }

                std::chrono::high_resolution_clock::time_point start_cpu, end_cpu, start_gpu, end_gpu;
                std::chrono::milliseconds diff_cpu, diff_gpu;

                graph_generator(gen_graph, VERTEX);

                start_cpu = std::chrono::high_resolution_clock::now();
                dijkstra(gen_graph, 0, dist_cpu);
                end_cpu = std::chrono::high_resolution_clock::now();

                start_gpu = std::chrono::high_resolution_clock::now();
                //dijkstra_parallelize_naive(gen_graph, 0, dist_gpu, VERTEX);
                //dijkstra_parallelize_shared(gen_graph, 0, dist_gpu);
                dijkstra_parallelize_shared_atomic(gen_graph, 0, dist_gpu);
                end_gpu = std::chrono::high_resolution_clock::now();

                compare_dist(dist_cpu, dist_gpu);

                diff_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
                diff_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);
                float time_cpu = diff_cpu.count();
                float time_gpu = diff_gpu.count();
                std::cout<<"Time duration CPU function: "<<time_cpu<<" ms"<<std::endl;
                std::cout<<"Time duration GPU kernel: "<<time_gpu<<" ms"<<std::endl;
                std::cout<<"speed up: "<<(float)100-(float)(100/time_cpu*time_gpu)<<" %"<<std::endl;

                free(dist_cpu), free(dist_gpu), free(gen_graph);
        }
        break;
        case 1:{
                // Δ-stepping cpu
                G.CSR_generate();
                
                DeltaStepping ds(G, delta);
                printf("DeltaStepping fields:\n");
                printf("delta: %d\n", ds.delta);
                printf("graph vertices: %ld\n", ds.n);
                printf("graph edges: %ld\n", ds.m);
                //G.print_CSR();
                std::chrono::high_resolution_clock::time_point start_cpu, end_cpu, start_gpu, end_gpu;
                std::chrono::milliseconds diff_cpu, diff_gpu;

                start_cpu = std::chrono::high_resolution_clock::now();
                ds.run(0);
                end_cpu = std::chrono::high_resolution_clock::now();

                diff_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
                float time_cpu = diff_cpu.count();
                std::cout<<"Time duration CPU function: "<<time_cpu<<" ms"<<std::endl;
                /*for (size_t i = 0; i < G.n; ++i) {
                    cout << "dist["<<i<<"] = " << ds.tent[i] << "\n";
                }*/
        }
        break;
        case 2:{// Δ-stepping cpu && dijkstra parallel
                G.CSR_generate();
                DeltaStepping ds2(G, delta);
                printf("DeltaStepping fields:\n");
                printf("delta: %d\n", ds2.delta);
                printf("graph vertices: %ld\n", ds2.n);
                printf("graph edges: %ld\n", ds2.m);
                short *gen_graph = (short *)malloc(VERTEX*VERTEX*sizeof(short*));
                short *dist_gpu = (short *)malloc(VERTEX*sizeof(short*));

                if(gen_graph == NULL || dist_gpu == NULL){
                printf("Error during memory allocation");
                exit(EXIT_FAILURE);
                }
                
                CRS_to_dense(gen_graph, G);

                dijkstra_parallelize_shared_atomic(gen_graph, 0, dist_gpu);
                // Δ-stepping
                ds2.run(0);
                compare_dist_tent(ds2.tent, dist_gpu);
                free(dist_gpu), free(gen_graph);
        }
        break;
        case 3:{// comparing Δ-stepping cpu with GPU version timing and result
                G.CSR_generate();
                DeltaStepping ds3(G, delta);
                printf("DeltaStepping fields:\n");
                printf("delta: %d\n", ds3.delta);
                printf("graph vertices: %ld\n", ds3.n);
                printf("graph edges: %ld\n", ds3.m);

                int *dist_gpu = (int *)malloc(VERTEX*sizeof(int*));
                size_t *row_ptr = &G.row_ptr[0];
                int *col_ind = &G.col_ind[0];
                int *weights = &G.weights[0];
                if(dist_gpu == NULL){
                    printf("Error during memory allocation");
                    exit(EXIT_FAILURE);
                    }

                std::chrono::high_resolution_clock::time_point start_cpu, end_cpu, start_gpu, end_gpu;
                std::chrono::milliseconds diff_cpu, diff_gpu;

                start_gpu = std::chrono::high_resolution_clock::now();
                delta_stepping_gpu_device_buckets(G.n, G.m, row_ptr, col_ind, weights, 0, ds3.delta, dist_gpu);
                //delta_stepping_gpu_shared(G.n, G.m, row_ptr, col_ind, weights, 0, ds.delta, dist_gpu);
                //delta_stepping_gpu_persistent_full(G.n, G.m, row_ptr, col_ind, weights, 0, ds.delta, dist_gpu);
                end_gpu = std::chrono::high_resolution_clock::now();
                printf("DeltaStepping GPU end-----------\n");
                start_cpu = std::chrono::high_resolution_clock::now();
                ds3.run(0);
                end_cpu = std::chrono::high_resolution_clock::now();
                printf("DeltaStepping CPU end-----------\n");
                compare_dist_csr(ds3.tent, dist_gpu);

                diff_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
                diff_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);
                float time_cpu = diff_cpu.count();
                float time_gpu = diff_gpu.count();
                std::cout<<"Time duration CPU function: "<<time_cpu<<" ms"<<std::endl;
                std::cout<<"Time duration GPU kernel: "<<time_gpu<<" ms"<<std::endl;
                std::cout<<"speed up: "<<(float)100-(float)(100/time_cpu*time_gpu)<<" %"<<std::endl;

                free(dist_gpu);

        }
        break;
        case 4:{ // comparing Δ-stepping GPU with dijkstra GPU
                G.CSR_generate();
                DeltaStepping ds4(G, delta);
                printf("DeltaStepping fields:\n");
                printf("delta: %d\n", ds4.delta);
                printf("graph vertices: %ld\n", ds4.n);
                printf("graph edges: %ld\n", ds4.m);

                short *gen_graph = (short *)malloc(VERTEX*VERTEX*sizeof(short*));
                short *dist_gpu = (short *)malloc(VERTEX*sizeof(short*));
                int *dist_gpu_delta = (int *)malloc(VERTEX*sizeof(int*));

                if(gen_graph == NULL || dist_gpu == NULL || dist_gpu_delta == NULL){
                    printf("Error during memory allocation");
                    exit(EXIT_FAILURE);
                }
                //generate graph in dense format
                size_t *row_ptr = &G.row_ptr[0];
                int *col_ind = &G.col_ind[0];
                int *weights = &G.weights[0];

                CRS_to_dense(gen_graph, G);

                std::chrono::high_resolution_clock::time_point start_gpu_delta, end_gpu_delta, start_gpu_dijkstra, end_gpu_dijkstra;
                std::chrono::milliseconds diff_dijkstra, diff_delta;

                start_gpu_dijkstra = std::chrono::high_resolution_clock::now();
                // Δ-stepping
                dijkstra_parallelize_shared_atomic(gen_graph, 0, dist_gpu);
                end_gpu_dijkstra = std::chrono::high_resolution_clock::now();
                printf("Dijkstra GPU end-----------\n");
                start_gpu_delta = std::chrono::high_resolution_clock::now();
                delta_stepping_gpu_device_buckets(G.n, G.m, row_ptr, col_ind, weights, 0, ds4.delta, dist_gpu_delta);
                end_gpu_delta = std::chrono::high_resolution_clock::now();
                printf("DeltaStepping GPU end-----------\n");

                compare_dist_delta_dijkstra(dist_gpu_delta, dist_gpu);

                diff_dijkstra = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_dijkstra - start_gpu_dijkstra);
                diff_delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_delta - start_gpu_delta);
                float time_dijkstra = diff_dijkstra.count();
                float time_delta = diff_delta.count();
                std::cout<<"Time duration dijkstra function: "<<time_dijkstra<<" ms"<<std::endl;
                std::cout<<"Time duration delta kernel: "<<time_delta<<" ms"<<std::endl;
                std::cout<<"speed up: "<<(float)100-(float)(100/time_dijkstra*time_delta)<<" %"<<std::endl;

                free(dist_gpu), free(dist_gpu_delta), free(gen_graph);
        }
        break;
        case 5:{ // Δ-stepping GPU
                G.CSR_generate();
                DeltaStepping ds5(G, delta);
                printf("DeltaStepping fields:\n");
                printf("delta: %d\n", ds5.delta);
                printf("graph vertices: %ld\n", ds5.n);
                printf("graph edges: %ld\n", ds5.m);

                int *dist_gpu_delta = (int *)malloc(VERTEX*sizeof(int*));

                if(dist_gpu_delta == NULL){
                    printf("Error during memory allocation");
                    exit(EXIT_FAILURE);
                }
                //generate graph in dense format
                size_t *row_ptr = &G.row_ptr[0];
                int *col_ind = &G.col_ind[0];
                int *weights = &G.weights[0];

                std::chrono::high_resolution_clock::time_point start_gpu_delta, end_gpu_delta, start_gpu_dijkstra, end_gpu_dijkstra;
                std::chrono::milliseconds diff_dijkstra, diff_delta;

                start_gpu_delta = std::chrono::high_resolution_clock::now();
                delta_stepping_gpu_device_buckets(G.n, G.m, row_ptr, col_ind, weights, 0, ds5.delta, dist_gpu_delta);
                end_gpu_delta = std::chrono::high_resolution_clock::now();
                printf("DeltaStepping GPU end-----------\n");

                diff_delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_delta - start_gpu_delta);
                float time_delta = diff_delta.count();
                std::cout<<"Time duration delta kernel: "<<time_delta<<" ms"<<std::endl;

                free(dist_gpu_delta);
        }
        break;
        case 6:{
                // help menu
                print_gpu_properties();
                printf("Args: <algorithm code>\n");
                printf("Algorithm code:\nrun-dijk: Dijkstra CPU vs Dijkstra GPU\nrun-delta: Δ-stepping CPU\nrun-check-correctness-delta: Δ-stepping CPU vs Dijkstra GPU\nrun-delta-compare: Δ-stepping CPU vs Δ-stepping GPU\nrun-delta-dijk: Dijkstra GPU vs Δ-stepping GPU\nrun-delta-naive: Δ-stepping GPU\nrun-help: this menu\n");
        }
        break;
        default : {
                printf("Error algorithm code not correct");
                exit(EXIT_FAILURE);
        }
        break;
    }
    

    return 0;
}
/*generate random graph with V*V node*/
void graph_generator(short* graph, int vertex){
    srand(time(NULL));
    for(int i = 0; i<vertex; i++){
        for(int j = i; j<vertex; j++){
            /*filling matrix with random value and leaving diagonal at 0 because is the distance between a vertex and itself*/
            graph[i*VERTEX+j] = graph[j*VERTEX+i] = (i!=j)*(rand()%VERTEX+1)*(rand()%100 <= 5);
            }
    }
}

void printSolution(short dist[], int n)
{
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < VERTEX; i++)
        printf("\t%d \t\t\t\t %d\n", i, dist[i]);
}

// Function that implements Dijkstra's single source
// shortest path algorithm
void dijkstra(short* graph, int src, short* dist)
{
    bool sptSet[VERTEX]; //Set to true all the vertexes for which the shortest path has been already found

    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < VERTEX; i++)
        dist[i] = 0x7FFF, sptSet[i] = false;

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < VERTEX - 1; count++) {
        // Pick the minimum distance vertex
        int u = minDistance(dist, sptSet, VERTEX);

        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the
        // picked vertex.
        for (int v = 0; v < VERTEX; v++){
            /*updating min distance in order to not create warp divergence*/
            dist[v] = (short)std::min(((int)dist[u] + (int)graph[u*VERTEX+v])+(graph[u*VERTEX+v] == 0)*(int)dist[v], (int)dist[v]);
        }
    }
}

void compare_dist(short *dist_cpu, short *dist_gpu){
    bool equal = 1;
    for(int i = 0; i<VERTEX; i++){
        if(dist_cpu[i] != dist_gpu[i]){
            printf("dist_cpu[%d] != dist_gpu[%d] (%hu != %hu)\r\n",i,i,dist_cpu[i], dist_gpu[i] );
            equal = 0;
        }
    }
    if(equal)
        printf("Result are equal!\r\n");
}

void compare_dist_tent(vector<int> dist_cpu, short *dist_gpu){
    bool equal = 1;
    for(int i = 0; i<VERTEX; i++){
        if(dist_cpu[i] != dist_gpu[i]){
            printf("dist_tent[%d] != dist_gpu[%d] (%hu != %hu)\r\n",i,i,dist_cpu[i], dist_gpu[i] );
            equal = 0;
        }
    }
    if(equal)
        printf("Result are equal!\r\n");
}

void compare_dist_delta_dijkstra(int* dist_cpu, short *dist_gpu){
    bool equal = 1;
    for(int i = 0; i<VERTEX; i++){
        if(dist_cpu[i] != dist_gpu[i]){
            printf("dist_tent[%d] != dist_gpu[%d] (%hu != %hu)\r\n",i,i,dist_cpu[i], dist_gpu[i] );
            equal = 0;
        }
    }
    if(equal)
        printf("Result are equal!\r\n");
}

void compare_dist_csr(vector<int> dist_cpu, int *dist_gpu){
    bool equal = 1;
    for(int i = 0; i<VERTEX; i++){
        if(dist_cpu[i] != dist_gpu[i]){
            printf("dist_tent[%d] != dist_gpu[%d] (%hu != %hu)\r\n",i,i,dist_cpu[i], dist_gpu[i] );
            equal = 0;
        }
    }
    if(equal)
        printf("Result are equal!\r\n");
}