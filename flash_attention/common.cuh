#include <cfloat>

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return max(a, b); }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return a + b; }
};

template <template <typename> class ReduceOp, typename T>
__device__ __inline__ T warpAllReduce(T val) {
    auto functor = ReduceOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = functor(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}


template <typename T>
__device__ __inline__ T blockAllReduceSum(T val) {
    __shared__ T shared[32];
    __shared__ T ret;
    int warp_id = (threadIdx.x >> 5);
    int lane_id = (threadIdx.x & 31);

    val = warpAllReduce<SumOp, T>(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(0.0f);
    val = warpAllReduce<SumOp, T>(val);
    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();

    return ret;
}

template <typename T>
__device__ __inline__ T blockAllReduceMax(T val) {
    __shared__ T shared[32];
    __shared__ T ret;
    int warp_id = (threadIdx.x >> 5);
    int lane_id = (threadIdx.x & 31);

    val = warpAllReduce<MaxOp, T>(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : (T)(-FLT_MAX);
    val = warpAllReduce<MaxOp, T>(val);
    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();

    return ret;
}


    struct __align__(8) MD_F {
        float m; // max val
        float d; // exp sum
    };

    struct MDFOp {
        __device__ __forceinline__ MD_F operator()(MD_F& a, MD_F& b) {
            MD_F ret;
            ret.m = max(a.m, b.m);
            ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
            return ret;
        }
    };


    static __device__ __inline__ MD_F warpAllReduceMDF(MD_F val) {
        float tmp_m;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, 32));
            val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, 32) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, 32) - tmp_m);
            val.m = tmp_m;
        }
        return val;
    }
