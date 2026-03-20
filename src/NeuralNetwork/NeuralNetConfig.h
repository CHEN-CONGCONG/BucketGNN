#pragma once
#include "LayerConfig.h"
#include "GraphConvConfig.h"
#include "globals.h"
#include "FCConfig.h"
#include "GlobalParams.h"
using namespace std;

class NeuralNetConfig
{
public:
	size_t numIterations = 0;
	size_t numLayers = 0;
	string phase = "";
	vector<LayerConfig *> layerConf;

	NeuralNetConfig(string _phase)
		: phase(_phase)
		 {};

	void addLayer(LayerConfig *fcl)
	{
		layerConf.push_back(fcl);
	};

	void setNumLayers(size_t _numLayers)
	{
		numLayers = _numLayers;
	};

	void checkNetwork()
	{
		if (!layerConf.empty())
		{
			if (layerConf.front()->type.compare("FC") == 0)
				assert(((FCConfig *)layerConf.front())->inputDim == NUM_FEATURES && "FC input error");
		}
		else
		{
			cout << "No layers in the network" << endl;
		}
	};
};