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

//#define PRINT

using namespace std;

struct CSRGraph {
    size_t n;                     // vertex
    size_t m;                     // edges (nnz)
    //size_t in order to avoid overflow for large graphs
    vector<size_t> row_ptr;       // size n+1
    vector<int> col_ind;          // size m
    vector<int> weights;          // size m

    CSRGraph(size_t n_, size_t avg_deg, unsigned seed = 42)
        : n(n_), m(n_ * avg_deg),
          row_ptr(n_ + 1, 0),
          col_ind(m),
          weights(m)
    {}
    
    void CSR_generate(){
        //mt19937 rng(time(NULL));
        mt19937 rng(2024);
        //set reange of random destination node and weight
        uniform_int_distribution<int> dist_node(0, int(n)-1);
        uniform_int_distribution<int> dist_weight(1, 100);

        size_t idx = 0;
        for (size_t u = 0; u < n; ++u) {
            for (size_t j = 0; j < (m/n); ++j) {
                if (idx >= m) {
                    cerr << "Error: idx >= m\n";
                    exit(EXIT_FAILURE);
                }
                col_ind[idx] = dist_node(rng);
                weights[idx] = dist_weight(rng);
                ++idx;
            }
            row_ptr[u + 1] = idx;
        }
        // sanity checks
        if (idx != m) {
            cerr << "Warning: generated edges (" << idx << ") != m (" << m << ")\n";
            m = idx;
            col_ind.resize(m);
            weights.resize(m);
        }
    }
    void print_CSR(){
        printf("row_ptr: ");
        for(size_t i = 0; i<row_ptr.size(); i++){
            printf("%zu ", row_ptr[i]);
        }
        printf("\ncol_ind: ");
        for(size_t i = 0; i<col_ind.size(); i++){
            printf("%d ", col_ind[i]);
        }
        printf("\nweights: ");
        for(size_t i = 0; i<weights.size(); i++){
            printf("%d ", weights[i]);
        }
        printf("\n");
    }
};

constexpr int INF = 0x3f3f3f3f;

class DeltaStepping {
public:
    size_t n;                         // vertex
    size_t m;                         // edges
    int delta;
    CSRGraph const& G;                // graph
    vector<int> tent;                 // tentative distances
    vector<vector<int>> B;            // bucket array

    DeltaStepping(CSRGraph const& G_, int delta_)
        : n(G_.n), m(G_.m), delta(delta_), G(G_), tent(G_.n, INF), B(G_.m) {}

    void relax(int v, int x, int currentBucket) {
        if ((size_t)v >= n) return; // guard
        //printf("%d ", v);
        if (x < tent[v]) {
            // if already in a bucket, remove
            if (tent[v] != INF) {
                int oldBucket = tent[v] / delta;
                int newBucket = x / delta;
                //printf("moving %d from bucket %d to %d\n", v, oldBucket, newBucket);
                if (oldBucket >= 0 && (size_t)oldBucket < B.size()) {
                    auto &bucket = B[oldBucket];
                    bucket.erase(remove(bucket.begin(), bucket.end(), v), bucket.end());
                }
            }
            // insert in the new correct bucket
            
            tent[v] = x;
            int newBucket = x / delta;
            #ifdef PRINT
            if(newBucket == currentBucket)
                    printf(" + %d", v);
            #endif
            if (newBucket >= 0 && (size_t)newBucket < B.size()) {
                B[newBucket].push_back(v);
                //printf("%d ", v);
            }
        }
    }

    void run(int source) {
        if (source < 0 || (size_t)source >= n) {
            cerr << "source out of range\n";
            return;
        }
        tent.assign(n, INF);
        for (auto &b : B) b.clear();

        tent[source] = 0;
        B[0].push_back(source);

        size_t i = 0;
        while (i < n) {
            if (B[i].empty()) { ++i; continue; }

            // heavy nodes set
            vector<int> S;

            // light edges phase
            
            while (!B[i].empty()) {
                #ifdef PRINT
                printf("\n bucket %d: \n", i);
                
                printf("\n bucket %d: ", i);
                #endif
                vector<pair<int,int>> Req;
                // iterate local copy to avoid issues if bucket changes
                vector<int> bucket_copy = B[i];
                for (int u : bucket_copy) {
                    #ifdef PRINT
                    printf("%d ", u);
                    #endif
                    if (u < 0 || (size_t)u >= n) continue;
                    //printf("Total edgs for node %d: %zu\n", u, G.row_ptr[u+1]-G.row_ptr[u]);
                    size_t start = G.row_ptr[u];
                    size_t end = G.row_ptr[u+1];
                    //get all the edges for node u
                    for (size_t j = start; j < end; ++j) {
                        int v = G.col_ind[j];
                        int w = G.weights[j];
                        if (w <= delta) {
                            // if node u has been reached so we can relax its edges
                            //printf(" -> %d ",v);
                            // if(v == 55 || v == 88){
                            //      printf("%d-> (%d) old=%d newd = (%d + %d )%d\n", v, u, tent[v], tent[u], w ,tent[u] + w);
                            // }
                            if (tent[u] != INF) 
                                Req.emplace_back(v, tent[u] + w);
                                
                        }
                    }
                    //printf(" | ");
                }
                // append S and clear bucket i
                S.insert(S.end(), B[i].begin(), B[i].end());
                B[i].clear();
                //printf("Bucket %d: ", i);
                for (auto &p : Req) relax(p.first, p.second, i);
                //printf("\n");

                // for (int i = 55; i < 56; ++i) {
                // printf("dist[%d] = %d - ", i, tent[i] == INF ? -1 : tent[i]);
                // }
            }
            

            // heavy edges phase
            vector<pair<int,int>> ReqHeavy;
            for (int u : S) {
                if (u < 0 || (size_t)u >= n) continue;
                size_t start = G.row_ptr[u];
                size_t end = G.row_ptr[u+1];
                for (size_t j = start; j < end; ++j) {
                    int v = G.col_ind[j];
                    int w = G.weights[j];
                    if (w > delta) {
                        // if(v == 55 || v == 88){
                        //          printf("%d-> (%d) old=%d newd = (%d + %d )%d\n", v, u, tent[v], tent[u], w ,tent[u] + w);
                        //     }
                        if (tent[u] != INF) 
                            ReqHeavy.emplace_back(v, tent[u] + w);
                    }
                }
            }
            printf("\n Heavy relax: \n");   
            for (auto &p : ReqHeavy) relax(p.first, p.second, i);

            // for (int i = 55; i < 56; ++i) {
            //     printf("dist[%d] = %d - ", i, tent[i] == INF ? -1 : tent[i]);
            //     }

            //printf(" \r\n");
            ++i;
        }
    }
};

void CRS_to_dense(short* graph, CSRGraph G){
    //inizializzo la matrice a 0
    for(int i = 0; i<VERTEX; i++){
        for(int j = 0; j<VERTEX; j++){
            graph[i*VERTEX+j] = 0;
        }
    }
    for(int i = 0; i<G.n; i++){
        for(int j = G.row_ptr[i]; j<G.row_ptr[i+1]; j++){
            if(graph[i*VERTEX+G.col_ind[j]] == 0)
                graph[i*VERTEX+G.col_ind[j]]  = G.weights[j];
            else if(graph[i*VERTEX+G.col_ind[j]] > G.weights[j])
                graph[i*VERTEX+G.col_ind[j]]  = G.weights[j];
            //else keep the minimum weight
            
        }
    }

    // for(int i = 0; i<VERTEX; i++){
    //     for(int j = 0; j<VERTEX; j++){
    //         printf("%d ", graph[i*VERTEX+j]);
    //     }
    //     printf("\n");
    // }
}

void dense_to_CSR(short* graph, CSRGraph &G){
    int idx = 0;
    G.row_ptr[0] = 0;
    for(int i = 0; i<G.n; i++){
        for(int j = 0; j<G.n; j++){
            if(graph[i*VERTEX+j] != 0){
                G.col_ind[idx] = j;
                G.weights[idx] = graph[i*VERTEX+j];
                idx++;
            }
        }
        G.row_ptr[i+1] = idx;
    }
    if(idx != G.m){
        printf("Warning: generated edges (%d) != m (%zu)\n", idx, G.m);
        G.m = idx;
        G.col_ind.resize(G.m);
        G.weights.resize(G.m);
    }
}