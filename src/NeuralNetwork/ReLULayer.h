#pragma once
#include "ReLUConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "GlobalParams.h"
using namespace std;

class ReLULayer : public Layer
{
private:
	ReLUConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorSmallType reluPrime; // Derivative of ReLU, DReLU

public:
	// Constructor and initializer
	ReLULayer(ReLUConfig *conf, int _layerNum);

	// Functions
	void printLayer() override;
	void forward(const RSSVectorMyType &inputActivation) override;
	void computeDelta(RSSVectorMyType &prevDelta) override;
	void updateEquations(const RSSVectorMyType &prevActivations) override;

	// Getters
	RSSVectorMyType *getActivation() { return &activations; };
	RSSVectorMyType *getDelta() { return &deltas; };
	int getInputDim() { return conf.inputDim; };
	int getOutputDim() { return conf.inputDim; };
};