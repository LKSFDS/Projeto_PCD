%%gpu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 2000       // Tamanho da grade
#define T 501        // Número de iterações no tempo
#define D 0.1        // Coeficiente de difusão
#define DELTA_T 0.01 // Intervalo de tempo entre iterações
#define DELTA_X 1.0  // Espaçamento entre os pontos da grade


// Kernel CUDA para resolver a equação de difusão
__global__ void diff_eq_kernel(double *C, double *C_new) {
    // coordenadas do ponto da thread atual
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * N + idx;
    //Verificação para não acessar valor fora da matriz
    if (idx > 0 && idx < N - 1 && idy > 0 && idy < N - 1) {
        C_new[index] = C[index] + D * DELTA_T * (
            (C[(idy + 1) * N + idx] + C[(idy - 1) * N + idx] +
             C[idy * N + (idx + 1)] + C[idy * N + (idx - 1)] -
             4 * C[index]) / (DELTA_X * DELTA_X)
        );
    }
}

int main() {
    // matrizes para a CPU
    double *C, *C_new;
    // matrizes para a GPU
    double *d_C, *d_C_new;
    //armazenar o tamanho das matrizes
    size_t size = N * N * sizeof(double);

    // Alocação das matrizes
    C = (double *)malloc(size);
    C_new = (double *)malloc(size);
    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Inicialização das matrizes
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            C_new[i * N + j] = 0.0;
        }
    }

    C[(N / 2) * N + (N / 2)] = 1.0;

    // Alocção das matrizes da GPU
    cudaMalloc((void **)&d_C, size);
    cudaMalloc((void **)&d_C_new, size);

    // Copiando os dados da CPU para a GPU
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_new, C_new, size, cudaMemcpyHostToDevice);

    //variaveis para calcular diferença de tempo
    cudaEvent_t inicio, final;
    cudaEventCreate(&inicio);
    cudaEventCreate(&final);

    cudaEventRecord(inicio);


    // Configura os parâmetros de execução: threads por bloco e número de blocos  16x16 threads
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,(N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    // Percorrendo todos os T
    for (int t = 0; t < T; t++) {
        diff_eq_kernel<<<numBlocks, threadsPerBlock>>>(d_C, d_C_new);

        // Troca os ponteiros
        double *temp = d_C;
        d_C = d_C_new;
        d_C_new = temp;

        // Calcular a diferença média
        if ((t % 100) == 0) {
            //copia os dados da GPU para CPU
            cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_new, d_C_new, size, cudaMemcpyDeviceToHost);
            double difmedio = 0.0;
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    difmedio += fabs(C[i * N + j] - C_new[i * N + j]);
                }
            }
            printf("Interação %d - Diferença média: %g\n", t, difmedio / ((N - 2) * (N - 2)));
        }
    }

    // Copia os dados da GPU para CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(final);
    cudaEventSynchronize(final);

    printf("Concentração final no centro: %f\n", C[(N / 2) * N + (N / 2)]);

    float tempo=0;
    cudaEventElapsedTime(&tempo,inicio,final);

    printf("Tempo final é %f milissegunndos\n",tempo);
    printf("em segundos: %f\n",tempo/1000.0);



    return 0;
}
