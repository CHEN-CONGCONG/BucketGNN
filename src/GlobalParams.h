
#ifndef GLOBALPARAMS_H
#define GLOBALPARAMS_H

#pragma once
#include <string>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <cassert> 
#include "AESObject.h"
#include "globals.h"
#include "tools.h"

using namespace std;

extern int partyNum;

// Dataset globals
extern string DATASET_NAME;
extern string DATASET_PATH;
extern bool ENABLE_SAI;
extern bool SECGNN_MASK_METHOD;
extern bool POSITION_SWAP;

// Dataset parameters
extern size_t NUM_CLASSES;
extern size_t NUM_FEATURES;
extern size_t NUM_NODES;
extern size_t NUM_EDGES;
extern size_t NUM_TRAIN;
extern size_t NUM_VAL;
extern size_t NUM_TEST;
extern size_t MAX_DEGREE;

// Neural network globals
extern size_t HIDDEN_LAYER_SIZE;
extern size_t MINI_BATCH_SIZE;
extern string PRELOAD_NETWORK_PATH;
extern bool LARGE_NETWORK;
extern bool WITH_NORMALIZATION;
extern int NO_OF_RUNS;

// MPC globals
extern string SECURITY_TYPE;
extern string NETWORK;

extern AESObject *aes_indep;
extern AESObject *aes_next;
extern AESObject *aes_prev;

// Graph globals
extern bool MASK_NEIGHBOR;
extern bool SAVE_NODE_INIT;
extern bool REWRITE_NODE_INIT;
extern string NODE_INIT_PATH;
extern bool PRECOMPUTE_NODE_STATE;
extern string NODE_STATE_PATH;
extern bool BATCH_ARRAY_ACCESS;
extern size_t BASE_ARRAY_SIZE;
extern size_t BATCH_NODE_SIZE_ARRAY_ACCESS;
extern string NEIGHBOR_PADDING_MODE;
extern string BUCKET_PADDING_STRATEGY;
extern string BUCKET_MERGE_MODE;
extern size_t BUCKET_MIN_NODES_K;
extern size_t BUCKET_COUNT;
extern size_t BUCKET_EXP_BASE;
extern vector<size_t> BUCKET_BOUNDARIES;
extern bool EXPORT_MASK_DISTRIBUTION;
extern string MASK_DISTRIBUTION_OUTPUT_DIR;

// defined in LoadData.cpp, which is included in GlobalParams.cpp
extern vector<Node> nodes; 
extern vector<uint64_t> testIdx;

void loadGlobalParams(const string& yamlFilePath);
void setSecurityType(const string& securityType);
void setNetwork(const string& network);
void initAESObjects(char* seed_indep, char* seed_next, char* seed_prev);
void deleteAESObjects();


#endif // GLOBALPARAMS_H
