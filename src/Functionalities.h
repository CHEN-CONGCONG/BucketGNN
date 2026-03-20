#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "EigenMatMul.h"
#include "GlobalParams.h"
using namespace std;

void funcTruncate(RSSVectorMyType &a, size_t power, size_t size);
void funcTruncatePublic(RSSVectorMyType &a, size_t divisor, size_t size);
void funcGetShares(RSSVectorMyType &a, const vector<myType> &data);
void funcGetShares(RSSVectorSmallType &a, const vector<smallType> &data);
void funcReconstructBit(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);
void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print);
void funcReconstruct(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print);
void funcReconstruct3out3(const vector<myType> &a, vector<myType> &b, size_t size, string str, bool print);
void funcMatMul(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &c,
				size_t rows, size_t common_dim, size_t columns,
				size_t transpose_a, size_t transpose_b, size_t truncation);
void funcDotProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
					RSSVectorMyType &c, size_t size, bool truncation, size_t precision);
void funcDotProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
					RSSVectorMyType &c, size_t size, bool truncation, vector<size_t> precision);
void funcDotProduct(const RSSVectorSmallType &a, const RSSVectorSmallType &b,
					RSSVectorSmallType &c, size_t size);
void funcPrivateCompare(const RSSVectorSmallType &share_m, const vector<myType> &r,
						const RSSVectorSmallType &beta, vector<smallType> &betaPrime,
						size_t size);
void funcWrap(const RSSVectorMyType &a, RSSVectorSmallType &theta, size_t size);
void funcSelectShares(const RSSVectorMyType &a, const RSSVectorSmallType &b, RSSVectorMyType &selected, size_t size);
void funcSelectBitShares(const RSSVectorSmallType &a0, const RSSVectorSmallType &a1,
						 const RSSVectorSmallType &b, RSSVectorSmallType &answer,
						 size_t rows, size_t columns, size_t loopCounter);
void funcRELUPrime(const RSSVectorMyType &a, RSSVectorSmallType &b, size_t size);
void funcRELU(const RSSVectorMyType &a, RSSVectorSmallType &temp, RSSVectorMyType &b, size_t size);
void funcPow(const RSSVectorMyType &b, vector<smallType> &alpha, size_t size);
void funcDivision(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &quotient,
				  size_t size);
void funcMaxpool(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorSmallType &maxPrime,
				 size_t rows, size_t columns);
void funcInvSqrt(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcInv(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcInnerProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
					  RSSMyType &c, size_t size, size_t precision);
void funcSoftmax(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);
void funcArrayAccess(const RSSVectorMyType &a, const RSSMyType index, RSSMyType &b, size_t size);
void funcArrayAccessVector(const vector<RSSVectorMyType> &a, const RSSMyType index, RSSVectorMyType &b, size_t size, size_t element_size);
void funcBatchNorm(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &quotient,
				   size_t batchSize, size_t B);
void funcForbeniusNorm(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size);

void funcGenerateBaseArray(const RSSVectorMyType &a, vector<vector<myType>> &base_array, RSSVectorMyType &base_r, size_t base_array_size, size_t element_size);
void funcBatchArrayAccess(const vector<vector<myType>> &base_array, const RSSVectorMyType &base_r, const RSSVectorMyType &index, RSSVectorMyType &c, size_t base_array_size, size_t index_size, size_t element_size);
void funcBatchArrayAccess(const vector<myType> &base_array, const RSSMyType &base_r, const RSSMyType &index, RSSMyType &c, size_t element_size);
void funcGenerateBaseArrayVector(const vector<RSSVectorMyType> &a, vector<vector<myType>> &base_array, RSSVectorMyType &base_r, size_t array_size, size_t base_array_size, size_t element_size);
void funcBatchArrayAccessVector(const vector<vector<myType>> &base_array, const RSSVectorMyType &base_r, const RSSVectorMyType &index, vector<RSSVectorMyType> &c, size_t array_size, size_t base_array_size, size_t index_size, size_t element_size);
void funcBatchArrayAccessVector(const vector<myType> &base_array, const RSSMyType &base_r, const RSSMyType &index, RSSVectorMyType &c, size_t array_size, size_t element_size);

// Debug
void debugTrunc();
void debugMatMul();
void debugDotProd();
void debugPC();
void debugWrap();
void debugReLUPrime();
void debugReLU();
void debugDivision();
void debugSelectBitShares();
void debugSelectShares();
void debugMaxpool();
void debugPow();
void debugInvSqrt();
void debugInv();
void debugInnerProd();
void debugSoftmax();
void debugArrayAccess();
void debugArrayAccessVector();
void debugBN();
void debugForbeniusNorm();
void debugBatchArrayAccess();
void debugBatchArrayAccessVector();

// Test
void testRSS();