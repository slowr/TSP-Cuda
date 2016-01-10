# TSP-Cuda
Travelling Salesman Problem on CUDA (Tested on GTX970)

To compile it:
nvcc -arch=sm_52 cuda.cu

To run it:
./a num_of_cities

You can mess around changing the maximum number of permutations per thread, threads per block and blocks per kernel to achieve better performance for your own GPU.
