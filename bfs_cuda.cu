/* Reference : The get neighbour is implemented by ourseleves, whereas the CUDA_BFS_KERNEL was referenced from https://github.com/siddharths2710/cuda_bfs 
				and the permutation generation code was referenced from https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/
*/ 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
// #include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

#define NUM_NODES 24
#define N 2
#define COMB "1230"
#define STR "abcd"

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

typedef struct
{
	int id;
	string s;

} HashMap;

int count = 0;
int edgeNumber = 0;
HashMap map[10000000];
Node node[NUM_NODES];

// index value is  = NUM_NODES*(NUM_NODES-1);
int edges[NUM_NODES*(NUM_NODES-1)];
// #define NUM_PERMUTATIONS 



__global__ void CUDA_BFS_KERNEL(Node *Va, int *Ea, bool *Fa, bool *Xa, int *Ca,bool *done)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > NUM_NODES)
		return;


	if (Fa[id] == true && Xa[id] == false)
	{
		printf("%d ", id); //This printf gives the order of vertices in BFS	
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads(); 
		int k = 0;
		int i;
		int start = Va[id].start;
		int end = start + Va[id].length;
		
		for (int i = start; i < end; i++) 
		{
			int nid = Ea[i];

			if (Xa[nid] == false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}
		}

	}

}

void swap(string a, int l, int i) 
{ 
	char temp; 
	temp = a[l];
	a[l] = a[i]; 
	a[i] = temp; 
} 


void permute(string a, int l, int r) 
{ 
   int i; 
   if (l == r) 
   {
	 // cout<<a<<endl;
		map[count].id = count;
		map[count].s = a;
		count+=1;
   }
   else
   { 
	   for (i = l; i <= r; i++) 
	   { 
		  	char temp; 
			temp = a[l];
			a[l] = a[i]; 
			a[i] = temp;
		  	permute(a, l+1, r); 
 
			temp = a[l];
			a[l] = a[i]; 
			a[i] = temp; //backtrack 
	   } 
   } 
}

void getneighbour(string s, int i)
{
	int mat[N][N];
	for(int j=0;j<N;j++)
	{
		for(int k=0;k<N;k++)
		{
			mat[j][k] = s[j*N+k]-'0';
		}
	}

	int posx,posy;
	for(int j=0;j<N;j++)
	{
		for(int k=0;k<N;k++)
		{
			if(mat[j][k] == 0)
			{
				posx = j;
				posy = k;
				break;
			}
		}
	}

	 if (posx == 0 && posy == 0) 
	 {
		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 2;

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}


	if (posx == 0 && posy == 1) 
	{
		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 2;

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}

	if (posx == 1 && posy == 0) 
	 {
		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 2;

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}
	if (posx == 1 && posy == 1) 
	 {
		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 2;

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}
	if (posy == 0) {

		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 3;

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}

	if (posy == N-1) {

		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 3;

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}

	if (posx == 0) {

		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 3;

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}

	if(posx == N-1) {
		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 3;

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		string s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}

	else {

		int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		node[i].start = edgeNumber;
		node[i].length = 4;

		// Moving 0 to the left
		Temp[posx][posy] = Temp[posx][posy-1];
		Temp[posx][posy-1] = 0;

		string s1 = STR;

		for(int j=0;j<2;j++)
		{
			for(int k=0;k<2;k++)
			{
				s1[j*2+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}


		// int Temp[2][2];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the right
		Temp[posx][posy] = Temp[posx][posy+1];
		Temp[posx][posy+1] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		// int Temp[N][N];
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the bottom
		Temp[posx][posy] = Temp[posx+1][posy];
		Temp[posx+1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				Temp[j][k] = mat[j][k];
			}
		}

		// Moving 0 to the top
		Temp[posx][posy] = Temp[posx-1][posy];
		Temp[posx-1][posy] = 0;

		// s1 = STR;

		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				s1[j*N+k] = Temp[j][k]+'0';
			}
		}

		for(int j=0;j<NUM_NODES;j++)
		{
			if (map[j].s == s1)
			{
				edges[edgeNumber++] = map[j].id;
			}
		}
	}
} 

// The BFS frontier corresponds to all the nodes being processed at the current level.

int main()
{

	permute(COMB, 0, N*N-1);

	for(int i=0;i<NUM_NODES;i++)
	{
		cout<<map[i].id<<" "<<map[i].s<<endl;
		// getneighbour(map[i].s, map[i].id);
	}

	for(int i=0;i<NUM_NODES;i++)
	{
		string s = map[i].s;
		int id = map[i].id;
		getneighbour(s, i);
	}

	cout<<"Done"<<endl;

	bool frontier[NUM_NODES] = { false };
	bool visited[NUM_NODES] = { false };
	int cost[NUM_NODES] = { 0 };

	int source = 0;
	frontier[source] = true;

	cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
	cudaEventRecord(start_kernel);

	Node* Va;
	cudaMalloc((void**)&Va, sizeof(Node)*NUM_NODES);
	cudaMemcpy(Va, node, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc((void**)&Ea, sizeof(int)*(NUM_NODES*(NUM_NODES+1)));
	cudaMemcpy(Ea, edges, sizeof(int)*(NUM_NODES*(NUM_NODES+1)), cudaMemcpyHostToDevice);

	bool* Fa;
	cudaMalloc((void**)&Fa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	bool* Xa;
	cudaMalloc((void**)&Xa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ca;
	cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	

	int num_blks = 1;
	int threads = 32;



	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	printf("\n\n");
	int count = 0;
    
    float runningTime = 0;

	printf("Order: \n\n");
	do 
	{
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		
		CUDA_BFS_KERNEL <<<num_blks, threads >>>(Va, Ea, Fa, Xa, Ca,d_done);
		cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);

	} while (done!=true);

	
	cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop_kernel);
	cudaEventSynchronize(stop_kernel);
	float k_time = 0;
    cudaEventElapsedTime(&k_time, start_kernel, stop_kernel);
    // runningTime+=k_time;
    cout << "\nGPU TIME : " <<k_time <<"ms"<<" "<<"Level : "<<count<<std::endl<<std::endl;
		
	
	printf("Number of times the kernel is called : %d \n", count);


	printf("\nCost: ");
	for (int i = 0; i<NUM_NODES; i++)
		printf( "%d    ", cost[i]);
	printf("\n");
	// _getch();
	cudaFree(Va);
	cudaFree(Ea);
	cudaFree(Fa);
	cudaFree(Xa);
	cudaFree(Xa);
	cudaFree(d_done);
	
}
