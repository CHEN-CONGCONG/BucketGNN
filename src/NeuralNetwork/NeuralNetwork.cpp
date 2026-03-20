#pragma once
#include "tools.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
using namespace std;

NeuralNetwork::NeuralNetwork(NeuralNetConfig *_config)
	: NUM_LAYERS(_config->numLayers),
	  PHASE(_config->phase)
{
	if (PHASE.compare("INFERENCE") == 0)
	{
		// Resize input and output data
		inputData.resize(NUM_NODES * NUM_FEATURES);
		outputData.resize(NUM_TEST * NUM_CLASSES);
	}
	else
	{
		assert(false && "Only  inference phase allowed");
	}

	// Initialize the layers
	for (size_t i = 0; i < NUM_LAYERS; ++i)
	{
		if (_config->layerConf[i]->type.compare("GraphConv") == 0)
		{
			GraphConvConfig *cfg = static_cast<GraphConvConfig *>(_config->layerConf[i]);
			layers.push_back(new GraphConvLayer(cfg, i));
		}
		else if (_config->layerConf[i]->type.compare("ReLU") == 0)
		{
			ReLUConfig *cfg = static_cast<ReLUConfig *>(_config->layerConf[i]);
			layers.push_back(new ReLULayer(cfg, i));
		}
		else if (_config->layerConf[i]->type.compare("FC") == 0)
		{
			FCConfig *cfg = static_cast<FCConfig *>(_config->layerConf[i]);
			layers.push_back(new FCLayer(cfg, i, PHASE));
		}
		else if (_config->layerConf[i]->type.compare("BN") == 0)
		{
			BNConfig *cfg = static_cast<BNConfig *>(_config->layerConf[i]);
			layers.push_back(new BNLayer(cfg, i));
		}
		else
			error("Only FC, GraphConv, BN, and ReLU layer types currently supported");
	}

	cout << "👉 [" << PHASE << "] neural network created with [" << NUM_LAYERS << "] layers!" << endl;
}

NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer *>::iterator it = layers.begin(); it != layers.end(); ++it)
		delete (*it);

	layers.clear();
}


void NeuralNetwork::test()
{
	log_print("GNN.test");

	clock_t start, finish;
	double Times;
	start = clock();

	// Forward
	forward();

	finish = clock();
	Times = (double)(finish - start) / CLOCKS_PER_SEC;
	if (PRINT_PROGRESS)
		cout << "Test done, execution time: " << Times << " s" << endl;
}

/**
 * @brief Forward pass of the neural network
 */
void NeuralNetwork::forward()
{
	log_print("GNN.forward");

	auto printInferenceProgress = [&](size_t completedLayers)
	{
		if (!PRINT_PROGRESS)
			return;

		double progress = (completedLayers * 100.0) / NUM_LAYERS;
		cout << "Progress [Inference]: " << fixed << setprecision(2) << progress
			 << "% | [" << completedLayers << "/" << NUM_LAYERS << "] layers completed" << endl;
	};

	layers[0]->forward(inputData);
	printInferenceProgress(1);
	if (LARGE_NETWORK)
		cout << "Forward \t" << layers[0]->layerNum << " completed..." << endl;

	for (size_t i = 1; i < NUM_LAYERS; ++i)
	{
		layers[i]->forward(*(layers[i - 1]->getActivation()));
		printInferenceProgress(i + 1);
		if (LARGE_NETWORK)
			cout << "Forward \t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::predict()
{
	log_print("GNN.predict");

	size_t rows;
	size_t columns = NUM_CLASSES;

	if (PHASE.compare("INFERENCE") == 0)
		rows = NUM_TEST;
	else
		assert(false && "Only inference phase allowed");

	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows * columns);
	if (PHASE.compare("INFERENCE") == 0)
	{
		RSSVectorMyType result(rows);
		size_t index;
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = testIdx[i] * columns + j;
				result[i * columns + j] = (*(layers[NUM_LAYERS - 1]->getActivation()))[index];
			}

		funcMaxpool(result, max, maxPrime, rows, columns);
	}
	else
		assert(false && "Only inference phase allowed");
}

/* May still have bug and security flaws */
void NeuralNetwork::getAccuracy(vector<size_t> &counter)
{
	log_print("GNN.getAccuracy");

	size_t rows;
	size_t columns = NUM_CLASSES;

	if (PHASE.compare("INFERENCE") == 0)
		rows = NUM_TEST;
	else
		assert(false && "Only inference phase allowed");

	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows * columns);
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows * columns);

	vector<myType> groundTruth(rows * columns);
	vector<smallType> prediction(rows * columns);

	// reconstruct ground truth from output data
	funcReconstruct(outputData, groundTruth, rows * columns, "groundTruth", false);
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);

	// reconstruct prediction from neural network
	funcMaxpool((*(layers[NUM_LAYERS - 1])->getActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows * columns, "prediction", false);

	for (int i = 0, index = 0; i < rows; ++i)
	{
		counter[1]++;
		for (int j = 0; j < columns; j++)
		{
			index = i * columns + j;
			if ((int)groundTruth[index] * (int)prediction[index] ||
				(!(int)groundTruth[index] && !(int)prediction[index]))
			{
				if (j == columns - 1)
				{
					counter[0]++;
				}
			}
			else
			{
				break;
			}
		}
	}

	cout << "Rolling accuracy: " << counter[0] << " out of "
		 << counter[1] << " (" << (counter[0] * 100.0 / counter[1]) << " %)" << endl;
}