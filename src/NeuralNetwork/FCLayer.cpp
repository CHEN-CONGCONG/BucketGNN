
#pragma once
#include "FCLayer.h"
#include "Functionalities.h"
#include <random>
#include <math.h>
using namespace std;

FCLayer::FCLayer(FCConfig *conf, int _layerNum, string _phase)
	: Layer(_layerNum, conf->layerType),
	  conf(conf->inputDim, conf->batchSize, conf->outputDim),
	  activations(conf->batchSize * conf->outputDim),
	  deltas(conf->batchSize * conf->outputDim),
	  weights(conf->inputDim * conf->outputDim),
	  biases(conf->outputDim),
	  PHASE(_phase)
{
	initialize();
}

void FCLayer::initialize()
{
	// Initialize weights and biases here.
	// Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	std::default_random_engine generator(1234);

	// Kaiming initialized for feedforward phase
	double variance = (double)2 / (conf.inputDim);
	double std_dev = sqrt(variance);
	std::normal_distribution<double> distribution(0.0, std_dev);

	float temp[size];
	for (size_t i = 0; i < size; ++i)
	{
		// temp[i] = (float)(rand() % (higher - lower) + lower) / decimation;
		temp[i] = distribution(generator);

		if (partyNum == PARTY_A)
		{
			weights[i].first = floatToMyType(temp[i]);
			weights[i].second = 0;
		}

		if (partyNum == PARTY_B)
		{
			weights[i].first = 0;
			weights[i].second = 0;
		}

		if (partyNum == PARTY_C)
		{
			weights[i].first = 0;
			weights[i].second = floatToMyType(temp[i]);
		}
	}

	fill(biases.begin(), biases.end(), make_pair(0, 0));
}

void FCLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;
	cout << "(" << layerNum + 1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl
		 << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

void FCLayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("FC.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows * columns;

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul, inputActivation, weights, activations, rows, common_dim, columns, 0, 0, FLOAT_PRECISION) << endl;
	else
		funcMatMul(inputActivation, weights, activations, rows, common_dim, columns, 0, 0, FLOAT_PRECISION);

	if ((PHASE.compare("INFERENCE") == 0 && !(NETWORK.compare("GCN") == 0)))
	{
		for (size_t r = 0; r < rows; ++r)
			for (size_t c = 0; c < columns; ++c)
				activations[r * columns + c] = activations[r * columns + c] + biases[c];
	}

	if (PRINT_PROGRESS)
	{
		cout << "Progress [Layer " << layerNum + 1 << "]: Forward Completed" << endl;
	}
}

void FCLayer::computeDelta(RSSVectorMyType &prevDelta)
{
	log_print("FC.computeDelta");

	// Back Propagate
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;

	if (FUNCTION_TIME)
		cout << "funcMatMul: " << funcTime(funcMatMul, deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, FLOAT_PRECISION) << endl;
	else
		funcMatMul(deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, FLOAT_PRECISION);
}

void FCLayer::updateEquations(const RSSVectorMyType &prevActivations)
{
	log_print("FC.updateEquations");
}