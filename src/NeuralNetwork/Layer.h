
#pragma once
#include "globals.h"

class Layer
{
public:
	int layerNum = 0;
	string layerType = "Unknown";
	vector<int> trainIdx;
	Layer(int _layerNum, string _layerType) : layerNum(_layerNum), layerType(_layerType) {};

	// Virtual functions
	virtual void printLayer() {};
	virtual void forward(const RSSVectorMyType &inputActivation) {};
	virtual void computeDelta(RSSVectorMyType &prevDelta) {};
	virtual void updateEquations(const RSSVectorMyType &prevActivations) {};

	// Getters
	virtual RSSVectorMyType *getActivation() {};
	virtual RSSVectorMyType *getDelta() {};
	virtual RSSVectorMyType *getBias() {};
	virtual int getInputDim() {};
	virtual int getOutputDim() {};

};