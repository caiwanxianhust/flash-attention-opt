#include <cstdio>
#include <cmath>
struct __align__(8) MD_F
{
    float m; // max val
    float d; // exp sum
};

struct MDFOp
{
    __device__ __forceinline__ MD_F operator()(MD_F &a, MD_F &b)
    {
        MD_F ret;
        ret.m = max(a.m, b.m);
        ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
        return ret;
    }
};
__device__ __inline__ MD_F warpAllReduce(MD_F val)
{
    float tmp_m;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, 32));
        val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, 32) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, 32) - tmp_m);
        val.m = tmp_m;
    }
    return val;
}

__global__ void testWarpReduceMDF()
{
    MD_F val;
    val.m = threadIdx.x;
    val.d = 1.0f;
    val = warpAllReduce(val);
    printf("threadIdx.x=%d, val.m=%f, val.d=%f\n", threadIdx.x, val.m, val.d);
}

int main()
{
    testWarpReduceMDF<<<1, 32>>>();
    cudaDeviceSynchronize();
    float exp_sum = 0.0f;

    for (int i=0; i<32; i++)
    {
        exp_sum += expf(i - 31);
    }
    printf("exp_sum=%f\n", exp_sum);
    return 0;
}