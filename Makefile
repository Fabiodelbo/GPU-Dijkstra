###########################################################
MKDIR=mkdir
## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR_DESKTOP = /usr/local/cuda-11.8
CUDA_ROOT_DIR_LAPTOP = /usr/local/cuda-10.8
CUDA_ROOT_DIR=/usr/local/cuda-11.8

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################
SM_GTX_750 = 50
SM_LAPTOP = 53
### GPU architecture
SM_ARCH=50

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch=sm_${SM_ARCH}
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
OBJS =  $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o
##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp $(OBJ_DIR) 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h $(OBJ_DIR) 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh $(OBJ_DIR) 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)


$(OBJ_DIR):
	$(MKDIR) -p $@

# Clean objects in object directory.
clean:
	$(RM) -r bin/ *.txt


run:
	./${EXE} > stdout.txt 2> stderr.txt

