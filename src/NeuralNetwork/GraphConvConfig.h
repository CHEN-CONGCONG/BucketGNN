#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class GraphConvConfig : public LayerConfig
{
public:
	size_t inputDim = 0;
	size_t nodeSize = 0;
	string layerType = "GraphConv";

	GraphConvConfig(size_t _inputDim, size_t _nodeSize)
		: inputDim(_inputDim),
		  nodeSize(_nodeSize),
		  layerType("GraphConv"),
		  LayerConfig("GraphConv") {};
};