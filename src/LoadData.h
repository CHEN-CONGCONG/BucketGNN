#ifndef LOADDATA_H
#define LOADDATA_H

#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "NeuralNetwork.h"
#include "GlobalParams.h"
#include "FCLayer.h"
#include "GraphConvLayer.h"
#include "globals.h"

using namespace std;

// Load data from the dataset
void loadInferenceData();
void readInferenceMiniBatch(NeuralNetwork *net);

// invoke the graph node processing when loading the inference data
// Note: This function is called in loadInferenceData()
void processGraphNodes();
void initNodes();

// Load the pre-trained model
void loadPretrainedModel(NeuralNetwork *net, const string& network);
void loadInitModel(NeuralNetwork *net, const string& network);


#endif