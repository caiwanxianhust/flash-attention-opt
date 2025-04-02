#pragma once
#include <stdio.h>
#include <iostream>
#include <cublas_v2.h>
#include <cufft.h>
#include <cstdlib>

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

static const char *_getcuFFTErrorEnum(cufftResult_t error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "The cuFFT operation was successful";

    case CUFFT_INVALID_PLAN:
        return "cuFFT was passed an invalid plan handle";

    case CUFFT_ALLOC_FAILED:
        return "cuFFT failed to allocate GPU or CPU memory";

    case CUFFT_INVALID_TYPE:
        return "No longer used";

    case CUFFT_INVALID_VALUE:
        return "User specified an invalid pointer or parameter";

    case CUFFT_INTERNAL_ERROR:
        return "Driver or internal cuFFT library error";

    case CUFFT_EXEC_FAILED:
        return "Failed to execute an FFT on the GPU";

    case CUFFT_SETUP_FAILED:
        return "The cuFFT library failed to initialize";

    case CUFFT_INVALID_SIZE:
        return "User specified an invalid transform size";

    case CUFFT_UNALIGNED_DATA:
        return "No longer used";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "Missing parameters in call";

    case CUFFT_INVALID_DEVICE:
        return "Execution of a plan was on different GPU than plan creation";

    case CUFFT_PARSE_ERROR:
        return "Internal plan database error";

    case CUFFT_NO_WORKSPACE:
        return "No workspace has been provided prior to plan execution";

    case CUFFT_NOT_IMPLEMENTED:
        return "Function does not implement functionality for parameters given.";

    case CUFFT_LICENSE_ERROR:
        return "Used in previous versions.";

    case CUFFT_NOT_SUPPORTED:
        return "Operation is not supported for parameters given.";
    }
    return "<unknown>";
}

#define PRINT_FUNC_NAME_()                                              \
    do                                                                  \
    {                                                                   \
        std::cout << "[FL][CALL] " << __FUNCTION__ << " " << std::endl; \
    } while (0)

#define CHECK_CUDA_ERROR(call)                             \
    do                                                     \
    {                                                      \
        const cudaError_t errorCode = call;                \
        if (errorCode != cudaSuccess)                      \
        {                                                  \
            printf("CUDA Error:\n");                       \
            printf("    File:   %s\n", __FILE__);          \
            printf("    Line:   %d\n", __LINE__);          \
            printf("    Error code:     %d\n", errorCode); \
            printf("    Error text:     %s\n",             \
                   cudaGetErrorString(errorCode));         \
            exit(1);                                       \
        }                                                  \
    } while (0)

#define CHECK_CUBLAS_STATUS(call)                            \
    do                                                       \
    {                                                        \
        const cublasStatus_t statusCode = call;              \
        if (statusCode != CUBLAS_STATUS_SUCCESS)             \
        {                                                    \
            printf("CUDA Error:\n");                         \
            printf("    File:   %s\n", __FILE__);            \
            printf("    Line:   %d\n", __LINE__);            \
            printf("    Status code:     %d\n", statusCode); \
            printf("    Error text:     %s\n",               \
                   _cudaGetErrorEnum(statusCode));           \
            exit(1);                                         \
        }                                                    \
    } while (0)

#define CHECK_CUFFT_STATUS(call)                             \
    do                                                       \
    {                                                        \
        const cufftResult_t statusCode = call;               \
        if (statusCode != CUFFT_SUCCESS)                     \
        {                                                    \
            printf("CUDA Error:\n");                         \
            printf("    File:   %s\n", __FILE__);            \
            printf("    Line:   %d\n", __LINE__);            \
            printf("    Status code:     %d\n", statusCode); \
            printf("    Error text:     %s\n",               \
                   _getcuFFTErrorEnum(statusCode));          \
            exit(1);                                         \
        }                                                    \
    } while (0)
    