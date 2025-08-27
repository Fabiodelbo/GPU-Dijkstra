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

// Number of vertices in the graph
//35000 nodes are 4.56 GB of memory as there are 1.225 bilion of vertexes
#define V 30000

void printSolution(short dist[], int n);
void dijkstra(short* graph, int src, short* dist);
void graph_generator(short* graph, int vertex);
void compare_dist(short *dist_cpu, short *dist_gpu);



// driver program to test above function
int main()
{
    short *gen_graph = (short *)malloc(V*V*sizeof(short*));
    short dist_cpu[V];
    short dist_gpu[V];
    std::chrono::high_resolution_clock::time_point start_cpu, end_cpu, start_gpu, end_gpu;
    std::chrono::milliseconds diff_cpu, diff_gpu;

    graph_generator(gen_graph, V);

    /*for(int i = 0; i<V; i++){
        for(int j = 0; j< V; j++){
            printf("%hu ", gen_graph[i*V+j]);
        }
        printf("\r\n ");
    }*/

    start_cpu = std::chrono::high_resolution_clock::now();
    dijkstra(gen_graph, 0, dist_cpu);
    end_cpu = std::chrono::high_resolution_clock::now();
    //printSolution(dist, V);

    start_gpu = std::chrono::high_resolution_clock::now();
    dijkstra_parallelize_naive(gen_graph, 0, dist_gpu, V);
    end_gpu = std::chrono::high_resolution_clock::now();


    //printSolution(dist, V);

    compare_dist(dist_cpu, dist_gpu);
    
    diff_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    diff_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);
    std::cout<<"Time duration CPU function: "<<diff_cpu.count()<<" ms"<<std::endl;
    std::cout<<"Time duration GPU kernel: "<<diff_gpu.count()<<" ms"<<std::endl;

    return 0;
}
/*generate random graph with V*V node*/
void graph_generator(short* graph, int vertex){
    srand(time(NULL));
    for(int i = 0; i<vertex; i++){
        for(int j = i; j<vertex; j++){
            /*filling matrix with random value and leaving diagonal at 0 because is the distance between a vertex and itself*/
            graph[i*V+j] = graph[j*V+i] = (i!=j)*(rand()%V+1);
            }
            
    }
}

// A utility function to find the vertex with minimum
// distance value, from the set of vertices not yet included
// in shortest path tree


void printSolution(short dist[], int n)
{
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < V; i++)
        printf("\t%d \t\t\t\t %d\n", i, dist[i]);
}

// Function that implements Dijkstra's single source
// shortest path algorithm
void dijkstra(short* graph, int src, short* dist)
{
    //int dist[V]; // Contains all the distance between the selected source and all the other vertexes

    bool sptSet[V]; //Set to true all the vertexes for which the shortest path has been already found

    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < V; i++)
        dist[i] = 0x7FFF, sptSet[i] = false;

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        // Pick the minimum distance vertex
        int u = minDistance(dist, sptSet, V);
        //printf(" u :%d -", u);

        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the
        // picked vertex.
        for (int v = 0; v < V; v++){
            /*updating min distance in order to not create warp divergence*/
            dist[v] = std::min<short>((short)(dist[u] + graph[u*V+v])+(graph[u*V+v] == 0)*dist[v], dist[v]);
        }
        /*printf("dist: ");
        for (int v = 0; v < V; v++)
            printf("%hu, ",dist[v]);
        printf("\r\n");*/
    }
}

void compare_dist(short *dist_cpu, short *dist_gpu){
    bool equal = 1;
    for(int i = V; i<V; i++){
        if(dist_cpu[i] != dist_gpu[i]){
            printf("dist_cpu[%d] != dist_gpu[%d] (%hu != %hu)\r\n",i,i,dist_cpu[i], dist_gpu[i] );
        }
    }
    if(equal)
        printf("Result are equal!\r\n");
}