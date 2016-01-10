#include "header.h"

#define MAX_THREADS 1024
#define MAX_BLOCKS 30
#define MAX_PERMS 5041

#define CUDA_RUN(x_) {cudaError_t cudaStatus = x_; if (cudaStatus != cudaSuccess) {fprintf(stderr, "Error  %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus)); goto Error;}}
#define SAFE(x_) {if((x_) == NULL) printf("out of memory. %d\n", __LINE__);}

__device__ __shared__ int32_t shared_cost;

__host__ unsigned long long factorial(int32_t n) {
	int c;
	unsigned long long result = 1;

	for (c = 1; c <= n; c++)
		result = result * c;

	return result;
}

int main(int argc, char *argv[]) {
	if (argc < 2) return 0;
	int size8 = sizeof(int8_t);
	int size32 = sizeof(int32_t);
	unsigned long long total_permutations, thread_perms, num_blocks = 1, num_threads, num_kernels = 1;
	float time_passed;
	cudaEvent_t startEvent, stopEvent;
	/* host variables */
	int8_t * city_ids, *shortestPath, *graphWeights, *choices;
	int32_t size = atoi(argv[1]), *cost;
	int8_t selected_K = 0;
	unsigned long long threads_per_kernel;
	/* device variables */
	int8_t * dev_city_ids, *dev_shortestPath, *dev_graphWeights, *dev_choices;
	int32_t * dev_cost, *dev_size;
	int8_t * dev_selected_K;
	unsigned long long * dev_threads_per_kernel;

	total_permutations = factorial(size - 1);
	printf("factorial(%d): %llu\n", size - 1, total_permutations);

	for (selected_K = 1; selected_K < size - 2; selected_K++) {
		thread_perms = factorial(size - 1 - selected_K);
		if (thread_perms < MAX_PERMS) break;
	}
	num_threads = total_permutations / thread_perms;
	int k;
	while (num_threads > MAX_THREADS) {
		k = 2;
		while (num_threads % k != 0) k++;
		num_threads /= k;
		num_blocks *= k;
	}
	while (num_blocks > MAX_BLOCKS) {
		k = 2;
		while (num_blocks % k != 0) k++;
		num_blocks /= k;
		num_kernels *= k;
	}
	threads_per_kernel = num_blocks * num_threads;
	printf("K selected: %d\n", selected_K);
	printf("num_threads %llu thread_perms %llu num_blocks %llu num_kernels %llu threads_per_kernel %llu\n", num_threads, thread_perms, num_blocks, num_kernels, threads_per_kernel);

	dim3 block_dim(num_threads, 1, 1);
	dim3 grid_dim(num_blocks, 1, 1);

	SAFE(city_ids = (int8_t *)malloc(size * size8));
	SAFE(shortestPath = (int8_t *)calloc(num_blocks * size, size8));
	SAFE(graphWeights = (int8_t *)malloc(size * size8 * size));
	SAFE(cost = (int32_t *)calloc(num_blocks * size, size32));
	SAFE(choices = (int8_t *)malloc(threads_per_kernel * size * size8));

	CUDA_RUN(cudaMalloc((void **)&dev_city_ids, size * size8));
	CUDA_RUN(cudaMalloc((void **)&dev_shortestPath, size * size8 * num_blocks));
	CUDA_RUN(cudaMalloc((void **)&dev_graphWeights, size * size8 * size));
	CUDA_RUN(cudaMalloc((void **)&dev_cost, num_blocks * size32));
	CUDA_RUN(cudaMalloc((void **)&dev_size, size32));
	CUDA_RUN(cudaMalloc((void **)&dev_selected_K, size8));
	CUDA_RUN(cudaMalloc((void **)&dev_choices, threads_per_kernel * size * size8));
	CUDA_RUN(cudaMalloc((void **)&dev_threads_per_kernel, sizeof(unsigned long long)));

	srand(time(NULL));
	initialize(city_ids, graphWeights, size);

	CUDA_RUN(cudaMemcpy(dev_city_ids, city_ids, size * size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_shortestPath, shortestPath, size * size8 * num_blocks, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_graphWeights, graphWeights, size * size8 * size, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_size, &size, size32, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_selected_K, &selected_K, size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_choices, choices, threads_per_kernel * size * size8, cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_threads_per_kernel, &threads_per_kernel, sizeof(unsigned long long), cudaMemcpyHostToDevice));
	CUDA_RUN(cudaMemcpy(dev_cost, cost, num_blocks * size32, cudaMemcpyHostToDevice));

	CUDA_RUN(cudaEventCreate(&startEvent));
	CUDA_RUN(cudaEventCreate(&stopEvent));
	CUDA_RUN(cudaEventRecord(startEvent, 0));
	float percentage;
	for (int i = 0; i < num_kernels; i++) {
		find_permutations_for_threads << < 1, 1 >> >(dev_city_ids, dev_selected_K, dev_choices, dev_size, dev_threads_per_kernel);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		combinations_kernel << < grid_dim, block_dim >> > (dev_choices, dev_selected_K, dev_shortestPath, dev_graphWeights, dev_cost, dev_size);
		CUDA_RUN(cudaGetLastError());
		CUDA_RUN(cudaDeviceSynchronize());
		percentage = (100. / (float) num_kernels * (float)(i + 1));
		printf("\rProgress : ");
		for (int j = 0; j < 10; j++) {
			if ((percentage / 10) / j > 1) printf("#");
			else printf(" ");
		}
		printf(" [%.2f%%]", percentage);
		fflush(stdout);
	}
	CUDA_RUN(cudaEventRecord(stopEvent, 0));
	CUDA_RUN(cudaEventSynchronize(stopEvent));
	CUDA_RUN(cudaEventElapsedTime(&time_passed, startEvent, stopEvent));
	CUDA_RUN(cudaMemcpy(shortestPath, dev_shortestPath, num_blocks * size * size8, cudaMemcpyDeviceToHost));
	CUDA_RUN(cudaMemcpy(cost, dev_cost, num_blocks * size32, cudaMemcpyDeviceToHost));

	printf("\nTime passed:  %3.1f ms \n", time_passed);
	//print_Graph(graphWeights, size);

	{
		int32_t min = cost[0];
		int8_t index = 0;
		for (int i = 1; i < num_blocks; i++) {
			if (cost[i] < min) {
				min = cost[i];
				index = i;
			}
		}
		printf("Shortest path found on block #%d:\n", index + 1);
		print_ShortestPath(&shortestPath[index * size], min, size);
	}

Error:
	free(city_ids);
	free(shortestPath);
	free(graphWeights);
	free(cost);
	free(choices);

	cudaFree(dev_city_ids);
	cudaFree(dev_shortestPath);
	cudaFree(dev_graphWeights);
	cudaFree(dev_cost);
	cudaFree(dev_size);
	cudaFree(dev_selected_K);
	cudaFree(dev_choices);
	cudaFree(dev_threads_per_kernel);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	getchar();

	return 0;
}

__global__
void find_permutations_for_threads(int8_t * city_ids, int8_t * k, int8_t * choices, int32_t * size, unsigned long long * threads_per_kernel) {
	int32_t length = *size;
	int8_t index = 1;
	unsigned long long count = 0;
	for (count = 0; count < *threads_per_kernel; count++) {
		for (int i = 0; i < length; i++) {
			choices[i + count * length] = city_ids[i];
		}
		reverse(city_ids + *k + index, city_ids + length);
		next_permutation(city_ids + index, city_ids + length);
	}
}

__global__
void combinations_kernel(int8_t * choices, int8_t * k, int8_t * shortestPath, int8_t * graphWeights, int32_t * cost, int32_t * size) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t length = *size;
	int8_t index = 1;

	/* local variables */
	int8_t * _path, *_shortestPath;
	int32_t _tcost;

	SAFE(_path = (int8_t *)malloc(length * sizeof(int8_t)));
	SAFE(_shortestPath = (int8_t *)malloc(length * sizeof(int8_t)));
	_tcost = length * 100;

	memcpy(_path, choices + tid * length, length * sizeof(int8_t));
	memcpy(_shortestPath, shortestPath, length * sizeof(int8_t));

	if (threadIdx.x == 0) {
		if (cost[blockIdx.x] == 0) cost[blockIdx.x] = length * 100;
		shared_cost = length * 100;
	}

	__syncthreads();

	do {
		coppy_array(_path, _shortestPath, &_tcost, graphWeights, length, tid);
	} while (next_permutation(_path + *k + index, _path + length));

	if (_tcost == shared_cost) {
		atomicMin(&cost[blockIdx.x], _tcost);
		if (cost[blockIdx.x] == _tcost) {
			memcpy(shortestPath + blockIdx.x * length, _shortestPath, length * sizeof(int8_t));
		}
	}

	free(_path);
	free(_shortestPath);
}

__host__
void initialize(int8_t * city_ids, int8_t * graphWeights, int32_t size) {
	for (int i = 0; i < size; i++) {
		city_ids[i] = i;
		for (int j = 0; j < size; j++) {
			if (i == j)
				graphWeights[i * size + j] = 0;
			else
				graphWeights[i * size + j] = 99;
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size;) {
			int next = 1; // (rand() % 2) + 1;
			int road = rand() % 100 + 1;
			if (i == j) {
				j += next;
				continue;
			}
			graphWeights[i * size + j] = road;
			j += next;
		}
	}

	for (int i = size - 1; i >= 0; i--) {
		graphWeights[((i + 1) % size) * size + i] = 1;
	}
}

__host__
void print_Graph(int8_t * graphWeights, int32_t size) {
	int i, j;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			printf("%d\t", graphWeights[i * size + j]);
		}
		printf("\n");
	}
}

__host__
void print_ShortestPath(int8_t * shortestPath, int32_t cost, int32_t size) {
	int i;
	if (cost == (size * 100)) printf("no possible path found.\n");
	else {
		for (i = 0; i < size; i++) {
			printf("%d\t", shortestPath[i]);
		}
		printf("\nCost: %d\n", cost);
	}
}

__device__
void swap(int8_t *x, int8_t *y) { int8_t tmp = *x; *x = *y;	*y = tmp; }

__device__
void reverse(int8_t *first, int8_t *last) { while ((first != last) && (first != --last)) swap(first++, last); }

__device__
void coppy_array(int8_t * path, int8_t * shortestPath, int32_t * tcost, int8_t * weights, int8_t length, int tid) {
	int32_t sum = 0;
	for (int32_t i = 0; i < length; i++) {
		int8_t val = weights[path[i] * length + path[(i + 1) % length]];
		if (val == -1) return;
		sum += val;
	}
	if (sum == 0) return;
	atomicMin(&shared_cost, sum);
	if (shared_cost == sum) {
		*tcost = sum;
		memcpy(shortestPath, path, length * sizeof(int32_t));
	}
}

__device__
bool next_permutation(int8_t * first, int8_t * last) {
	if (first == last) return false;
	int8_t * i = first;
	++i;
	if (i == last) return false;
	i = last;
	--i;

	for (;;) {
		int8_t * ii = i--;
		if (*i < *ii) {
			int8_t * j = last;
			while (!(*i < *--j));
			swap(i, j);
			reverse(ii, last);
			return true;
		}
		if (i == first) {
			reverse(first, last);
			return false;
		}
	}
}
