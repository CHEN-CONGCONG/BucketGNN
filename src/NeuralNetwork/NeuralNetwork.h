#pragma once
#include "NeuralNetConfig.h"
#include "GraphConvLayer.h"
#include "GraphConvConfig.h"
#include "BNLayer.h"
#include "BNConfig.h"
#include "ReLULayer.h"
#include "ReLUConfig.h"
#include "FCLayer.h"
#include "FCConfig.h"
#include "Layer.h"
#include "globals.h"
#include "GlobalParams.h"
using namespace std;

class NeuralNetwork
{
private:
	void forward();
	void backward();
	void computeDelta();
	void updateEquations();


public:
	size_t NUM_LAYERS;
	string PHASE;

	vector<Layer *> layers;
	RSSVectorMyType inputData;
	RSSVectorMyType outputData;

	NeuralNetwork(NeuralNetConfig *_config);
	~NeuralNetwork();
	void test();
	void predict();
	void getAccuracy(vector<size_t> &counter);
};