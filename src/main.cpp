#include <iostream>
#include <string>
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "globals.h"
#include "tools.h"
#include "unitTests.h"
#include "GlobalParams.h"

int partyNum;
string dataset, outputDir;
// Wall Clock Time(s), CPU Time(s), Communication Sent(MB), Communication Received(MB), Rounds Sent, Rounds Received
// Total Communication Sent(MB), Total Communication Recv(MB), Total Communication(MB), Total Rounds Sent, Total Rounds Received, Total Rounds, Accuracy
vector<double> results(13, 0.0);

Precompute PrecomputeObject;

int main(int argc, char **argv)
{
	/****************************** PREPROCESSING ******************************/
	parseInputs(argc, argv);
	string network, security, mode;

	/****************************** AES SETUP and SYNC ******************************/
	initAESObjects(argv[3], argv[4], argv[5]);

	// Create connections, the first parameter is the file containing the IP, the second parameter is the participant number
	initializeCommunication(argv[2], partyNum);
	// Create multiple threads to execute the sendByte and receiveByte functions
	synchronize(2000000);

	/******************************  SELECT NETWORK ******************************/

	// Network {GCN SGC APPNP}
	// Dataset {pubmed, cora, citeseer, ogbn-arxiv, ogbn-products}
	// Security {Semi-honest}
	// Output directory
	// Mode {test, init, all, debug}
	// Default network settings
	if (argc == 11)
	{
		network = argv[6];
		dataset = argv[7];
		security = argv[8];
		outputDir = argv[9];
		mode = argv[10];
	}
	else
	{
		network = "GCN";
		dataset = "pubmed";
		security = "Semi-honest";
		outputDir = "output";
		mode = "test";
	}

	// Load parameters
	loadGlobalParams("parameters/" + dataset + ".yaml");
	setSecurityType(security);
	setNetwork(network);

	initResults(outputDir);

	NeuralNetConfig *inference_config = new NeuralNetConfig("INFERENCE");
	selectNetwork(network, inference_config);
	inference_config->checkNetwork();

	NeuralNetwork *inference_net = new NeuralNetwork(inference_config);

	cout << "✅ Network [" << network << "] successfully created!" << endl;
	cout << "----------------------------------------------" << endl;
	cout << "ENABLE_SAI: " << ENABLE_SAI << endl;

	/****************************** RUN NETWORK/UNIT TESTS ******************************/

	// Run unit tests in two modes:
	//	1. Debug {Trunc, MatMul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, SelectShares, SelectBitShares, Pow,
	//			  Maxpool, InvSqrt, Inv, InnerProd, Softmax, ArrayAccess, ArrayAccessVector, BN, ForbeniusNorm,
	//			  BatchArrayAccess, BatchArrayAccessVector}
	//	2. Test {RSS}
	if (mode == "debug")
		runTest("Debug", "BatchArrayAccess", network);
	// runTest("Test", "RSS", network);

	/****************************** LOAD DATA ******************************/

	// Load inference data
	if (mode == "test" || mode == "all")
	{
		loadInferenceData();
		readInferenceMiniBatch(inference_net);
	}

	// Initialize nodes
	if (ENABLE_SAI)
	{
		start_m();
		initNodes();
		end_m("Init Nodes", results);
		writeResults("init_nodes", outputDir, results);
	}


	/**************** Run testing ****************/
	if (mode == "test" || mode == "all")
	{
		// Load pre-trained model
		preload_network(inference_net);

		// Run inference
		start_m();
		test(inference_net);
		end_m("Test", results);

		double accuracy = getAccuracy(inference_net);
		results[12] = accuracy;

		writeResults("test", outputDir, results);
	}

	cout << "----------------------------------------------" << endl;
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum
		 << "), " << " batch size " << MINI_BATCH_SIZE << endl
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl
		 << endl;

	printNetwork(inference_net);
	/****************************** CLEAN-UP ******************************/
	deleteAESObjects();
	delete inference_config;
	delete inference_net;
	deleteObjects();

	return 0;
}
