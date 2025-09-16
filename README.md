# GPU-Dijkstra
Project for the GPU programming course of Master degree in Computer Science

## Requirements

- Linux environment
- CUDA library installed

## Setup and Compilation

1. **Set the CUDA path**:  
   Open the `Makefile` and set your CUDA installation path by modifying the variable `CUDA_ROOT_DIR`.

2. **Modify graph parameters**:  
   You can configure the parameters for graph generation (number of vertices, edges per vertex, max edge weight, etc.) by editing the file `cuda_kernel.cuh`.

3. **Compile the project**:  
   ```bash
   make

4. **Info about the running mode**:  
   ```bash
   make run-help