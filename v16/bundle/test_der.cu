// Mini program to test gpu_is_valid_der on the suspect hash
#include <stdio.h>
#include <stdint.h>

__device__ int gpu_is_valid_der(const uint8_t *d, int l) {
    if(l<9||d[0]!=0x30) return 0;
    int tl=d[1]; if(tl+3!=l) return 0;
    int idx=2;
    for(int p=0;p<2;p++){
        if(idx>=l-1||d[idx]!=0x02) return 0; idx++;
        int il=d[idx]; idx++;
        if(il==0||idx+il>l-1) return 0;
        if(il>1&&d[idx]==0&&!(d[idx+1]&0x80)) return 0;
        if(d[idx]&0x80) return 0; idx+=il;}
    return idx==l-1;
}

__global__ void test_der(const uint8_t *h, int *result) {
    *result = gpu_is_valid_der(h, 32);
}

int main() {
    // Test on the hash that GPU should reject:
    uint8_t hash[] = {
        0xd1, 0xbb, 0x9b, 0xe6, 0x03, 0xf1, 0x01, 0x67,
        0xc1, 0x2d, 0xaf, 0xf1, 0xdf, 0xd7, 0xe9, 0x04,
        0x7e, 0xec, 0x9b, 0xd9, 0x2f, 0x19, 0xe0, 0x74,
        0x67, 0xd4, 0xc6, 0x50, 0x49, 0xb2, 0xf5, 0xc6
    };
    
    uint8_t *d_hash;
    int *d_result;
    int h_result = -1;
    
    cudaMalloc(&d_hash, 32);
    cudaMalloc(&d_result, 4);
    cudaMemcpy(d_hash, hash, 32, cudaMemcpyHostToDevice);
    
    test_der<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    
    printf("GPU is_valid_der returned: %d\n", h_result);
    printf("Expected: 0 (hash starts with 0xd1, not 0x30)\n");
    
    cudaFree(d_hash);
    cudaFree(d_result);
    return 0;
}
