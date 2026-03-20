
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "GlobalParams.h"
using namespace std;


class FCLayer : public Layer
{
private:
	FCConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;

public:
	string PHASE;

	// Constructor and initializer
	FCLayer(FCConfig *conf, int _layerNum, string _phase);
	void initialize();

	// Functions
	void printLayer() override;
	void forward(const RSSVectorMyType &inputActivation) override;
	void computeDelta(RSSVectorMyType &prevDelta) override;
	void updateEquations(const RSSVectorMyType &prevActivations) override;

	// Getters
	RSSVectorMyType *getActivation() { return &activations; };
	RSSVectorMyType *getDelta() { return &deltas; };
	RSSVectorMyType *getWeights() { return &weights; };
	RSSVectorMyType *getBias() { return &biases; };
	int getInputDim() { return conf.inputDim; };
	int getOutputDim() { return conf.outputDim; };
};