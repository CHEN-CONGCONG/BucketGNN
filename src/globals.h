
#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <array>
#include <cstdint>

/********************* Macros *********************/
#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define _aligned_free free
#define getrandom(min, max) ((rand() % (int)(((max) + 1) - (min))) + (min))
#define floatToMyType(a) ((myType)(int)floor(a * (1 << FLOAT_PRECISION)))

/********************* AES and other globals *********************/
#define LOG_DEBUG false
#define LOG_DEBUG_NETWORK false
#define FUNCTION_TIME false
#define RANDOM_COMPUTE 256 // Size of buffer for random elements
#define STRING_BUFFER_SIZE 256
#define PARALLEL true
#define NO_CORES 8
#define PRINT_PROGRESS true // print progress of the computation
#define PRINT_INTERVAL 10  // print interval for progress

/********************* MPC globals *********************/
#define NUM_OF_PARTIES 3
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2
#define USING_EIGEN true // for fast matrix operations
#define PRIME_NUMBER 67
#define FLOAT_PRECISION 13 // 11 for uint32_t, 16 for uint64_t
#define PRECISE_DIVISION true
#define SOFTMAX_ITER_NUM 8     // for softmax, 8/16/32, Refer: Secure Softmax/Sigmoid for Machine-learning Computation
#define INV_SQRT_ITER_NUM 4    // iteration number for inv_sqrt
#define BATCH_COMPUTE_THRESHOLD_RELU 10000      // if size is too large, split into batches for ReLU
#define BATCH_COMPUTE_THRESHOLD_BASE_ARRAY 100 // if size is too large, split into batches for base array

/********************* Typedefs and others *********************/
typedef uint32_t myType;
typedef uint8_t smallType;
typedef std::pair<myType, myType> RSSMyType;
typedef std::pair<smallType, smallType> RSSSmallType;
typedef std::vector<RSSMyType> RSSVectorMyType;
typedef std::vector<RSSSmallType> RSSVectorSmallType;

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);               // myType bit size
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));       // Largest negative number, 2^(n-1), unsigned n-bit integer
const myType MINUS_ONE = (myType)-1;                            // unsigned -1, 2^n - 1
const smallType BOUNDARY_PRIME = 256;                           // boundary for prime number
const smallType BOUNDARY = (256 / PRIME_NUMBER) * PRIME_NUMBER; // boundary

typedef struct Node
{
    RSSVectorMyType neighborIDs, edgeWeights;                   // neighbor list and edge weights
    RSSVectorMyType coefNeigh;                                  // ew_ij * 1/sqrt{sw_i} * 1/sqrt{sw_j}
    RSSMyType sw, swReci, swInvSqrt;                            // sum of edge weights, 1/sw, 1/sqrt{sw}
    RSSVectorMyType feature;                                    // Feature vector
    RSSVectorMyType activation;                                 // activation
    RSSVectorMyType groundTruth;                                // Label
};

#endif
