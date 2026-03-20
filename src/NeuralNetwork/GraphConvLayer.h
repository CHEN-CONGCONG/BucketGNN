#pragma once
#include "GraphConvConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "GlobalParams.h"
using namespace std;

class GraphConvLayer : public Layer
{
private:
    GraphConvConfig conf;
    RSSVectorMyType activations;
    RSSVectorMyType biases;

public:
    // Constructor and initializer
    GraphConvLayer(GraphConvConfig *conf, int _layerNum);
    void initialize();

    // Functions
    void printLayer() override;
    void forward(const RSSVectorMyType &inputActivation) override;
    void computeDelta(RSSVectorMyType &prevDelta) override;
    void updateEquations(const RSSVectorMyType &prevActivations) override;

    void loadNodeStates(vector<RSSVectorMyType> &states);

    // Getters
    RSSVectorMyType *getActivation()
    {
        return &activations;
    };

    RSSVectorMyType *getBias()
    {
        return &biases;
    };

    int getInputDim() { return conf.inputDim; };
};