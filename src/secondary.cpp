
#include "connect.h"
#include "secondary.h"
#include "Functionalities.h"
#include "GraphConvLayer.h"
#include <sys/stat.h>

extern CommunicationObject commObject;
extern string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;
extern void log_print(string str);
extern void print_linear(myType var, string type);
#define NANOSECONDS_PER_SEC 1E9

// For faster modular operations
extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

// For time measurements
clock_t tStart;
struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;
int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

/******************* Main train and test functions *******************/

void parseInputs(int argc, char *argv[])
{
	if (argc < 6)
		print_usage(argv[0]);

	partyNum = atoi(argv[1]);

	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = ((i + j) % PRIME_NUMBER);
			subtractModPrime[i][j] = ((PRIME_NUMBER + i - j) % PRIME_NUMBER);
			multiplicationModPrime[i][j] = ((i * j) % PRIME_NUMBER); // How come you give the right answer multiplying in 8-bits??
		}
}


void test(NeuralNetwork *inference_net)
{
	log_print("test");

	cout << "----------------------------------" << endl;
	cout << "🚀 Start testing..." << endl;
	inference_net->test();
	cout << "----------------------------------" << endl;
}

double getAccuracy(NeuralNetwork *net)
{
	// counter[0]: Correct samples, counter[1]: total samples
	vector<size_t> counter(2, 0);

	net->getAccuracy(counter);
	return (double)counter[0] / counter[1];
}

void selectNetwork(string network, NeuralNetConfig *config)
{
	assert((network.compare("GCN") == 0) &&
		   "Only GCN supported");

	if (config->phase.compare("INFERENCE") == 0)
	{
		if (network.compare("GCN") == 0)
		{
			if (POSITION_SWAP)
			{
				FCConfig *l0 = new FCConfig(NUM_FEATURES, NUM_NODES, HIDDEN_LAYER_SIZE);
				GraphConvConfig *l1 = new GraphConvConfig(HIDDEN_LAYER_SIZE, NUM_NODES);
				ReLUConfig *l2 = new ReLUConfig(HIDDEN_LAYER_SIZE, NUM_NODES);
				FCConfig *l3 = new FCConfig(HIDDEN_LAYER_SIZE, NUM_NODES, NUM_CLASSES);
				GraphConvConfig *l4 = new GraphConvConfig(NUM_CLASSES, NUM_TEST);
				config->addLayer(l0);
				config->addLayer(l1);
				config->addLayer(l2);
				config->addLayer(l3);
				config->addLayer(l4);

				config->setNumLayers(5);
			}
			else
			{
				GraphConvConfig *l0 = new GraphConvConfig(NUM_FEATURES, NUM_NODES);
				FCConfig *l1 = new FCConfig(NUM_FEATURES, NUM_NODES, HIDDEN_LAYER_SIZE);
				ReLUConfig *l2 = new ReLUConfig(HIDDEN_LAYER_SIZE, NUM_NODES);
				GraphConvConfig *l3 = new GraphConvConfig(HIDDEN_LAYER_SIZE, NUM_TEST);
				FCConfig *l4 = new FCConfig(HIDDEN_LAYER_SIZE, NUM_TEST, NUM_CLASSES);
				config->addLayer(l0);
				config->addLayer(l1);
				config->addLayer(l2);
				config->addLayer(l3);
				config->addLayer(l4);

				config->setNumLayers(5);
			}
		}
		else
			assert(false && "Only GCN supported");
	}
	else
		assert(false && "Only inference phase allowed");
}

void preload_network( NeuralNetwork *inference_net)
{
	log_print("preload_network");
	loadPretrainedModel(inference_net, NETWORK);
}

void printNetwork(NeuralNetwork *net)
{
	cout << " network:" << net->PHASE << endl;
	for (int i = 0; i < net->layers.size(); ++i)
		net->layers[i]->printLayer();
	cout << "----------------------------------------------" << endl
		 << endl;
}

void initResults(string outputDir)
{
	// if outputDir does not exist, create it
	if (access(outputDir.c_str(), 0) == -1)
	{
		int flag = mkdir(outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (flag == 0)
			cout << "Create output folder successfully." << endl;
	}
	string filename = outputDir + "/" + NETWORK + "_" + DATASET_NAME + "_" + to_string(partyNum) + ".csv";
	ofstream file(filename, ios::trunc);
	file << "mask_neighbor,save_node_init,float_precision,";
	file << "Type,Network,Dataset,Security,";
	file << "Wall Clock Time(s),CPU Time(s),Communication Sent(MB),Communication Received(MB),Rounds Sent,Rounds Received,"
			"Total Communication Sent(MB),Total Communication Recv(MB),Total Communication(MB),Total Rounds Sent,Total Rounds Received,Total Rounds,"
			"Accuracy"
		 << endl;

	file.close();

	cout << "Results written to " << filename << endl;
}

void writeResults(string str, string outputDir, vector<double> results)
{
	string filename = outputDir + "/" + NETWORK + "_" + DATASET_NAME + "_" + to_string(partyNum) + ".csv";
	ofstream file(filename, ios::app);
	file << MASK_NEIGHBOR << "," << SAVE_NODE_INIT << "," << FLOAT_PRECISION << "," ;
	file << str << "," << NETWORK << "," << DATASET_NAME << "," << SECURITY_TYPE << ",";
	for (size_t i = 0; i < results.size(); ++i)
	{
		if (i == results.size() - 1)
			file << results[i];
		else
			file << results[i] << ",";
	}
	file << endl;
	file.close();
}

/********************* COMMUNICATION AND HELPERS *********************/
void start_m()
{
	// cout << endl;
	start_time();
	start_communication();
}

void end_m(string str, vector<double> &results)
{
	// wallClock, cpuTime
	vector<double> times = end_time(str);
	pause_communication();
	vector<double> total = aggregateCommunication();
	// sent, recv, roundsSent, roundsRecv
	vector<double> comm = end_communication(str);

	// times
	for (size_t i = 0; i < 2; ++i)
		results[i] = times[i];

	// communication
	for (size_t i = 0; i < 4; ++i)
		results[i + 2] = comm[i];

	// total communication
	for (size_t i = 0; i < 6; ++i)
		results[i + 6] = total[i];

	// accuracy
	// results[12] =
}

void start_time()
{
	if (alreadyMeasuringTime)
	{
		cout << "Nested timing measurements" << endl;
		exit(-1);
	}

	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
}

vector<double> end_time(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_time() never called" << endl;
		exit(-1);
	}

	clock_gettime(CLOCK_REALTIME, &requestEnd);
	double wallClock = diff(requestStart, requestEnd);
	double cpuTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
	cout << "----------------------------------------------" << endl;
	cout << "Wall Clock time for " << str << ": " << wallClock << " sec\n";
	cout << "CPU time for " << str << ": " << cpuTime << " sec\n";
	cout << "----------------------------------------------" << endl;
	alreadyMeasuringTime = false;
	return {wallClock, cpuTime};
}

void start_rounds()
{
	if (alreadyMeasuringRounds)
	{
		cout << "Nested round measurements" << endl;
		exit(-1);
	}

	roundComplexitySend = 0;
	roundComplexityRecv = 0;
	alreadyMeasuringRounds = true;
}

void end_rounds(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_rounds() never called" << endl;
		exit(-1);
	}

	cout << "----------------------------------------------" << endl;
	cout << "Send Round Complexity of " << str << ": " << roundComplexitySend << endl;
	cout << "Recv Round Complexity of " << str << ": " << roundComplexityRecv << endl;
	cout << "----------------------------------------------" << endl;
	alreadyMeasuringRounds = false;
}

vector<double> aggregateCommunication()
{
	vector<uint64_t> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (partyNum == PARTY_B || partyNum == PARTY_C)
		sendVector<uint64_t>(vec, PARTY_A, 4);

	if (partyNum == PARTY_A)
	{
		receiveVector<uint64_t>(temp, PARTY_B, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
		receiveVector<uint64_t>(temp, PARTY_C, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
	}
	cout << "Sent: " << vec[0] << " Recv: " << vec[1] << " Rounds Sent: " << vec[2] << " Rounds Recv: " << vec[3] << endl;

	double total_sent, total_recv, total_rounds_sent, total_rounds_recv, total_comm, total_rounds;
	if (partyNum == PARTY_A)
	{
		total_sent = (float)vec[0] / 1000000;
		total_recv = (float)vec[1] / 1000000;
		total_rounds_sent = vec[2];
		total_rounds_recv = vec[3];
		total_comm = total_sent + total_recv;
		total_rounds = total_rounds_sent + total_rounds_recv;
		cout << "----------------------------------------------" << endl;
		cout << "Total communication: " << total_sent << " MB (sent) and " << total_recv << " MB (recv)\n";
		cout << "Total calls: " << total_rounds_sent << " (sends) and " << total_rounds_recv << " (recvs)" << endl;
		cout << "----------------------------------------------" << endl;
	}
	return {total_sent, total_recv, total_comm, total_rounds_sent, total_rounds_recv, total_rounds};
}

void print_usage(const char *bin)
{
	cout << "Usage: ./" << bin << " PARTY_NUM IP_ADDR_FILE AES_SEED_INDEP AES_SEED_NEXT AES_SEED_PREV" << endl;
	cout << endl;
	cout << "Required Arguments:\n";
	cout << "PARTY_NUM			Party Identifier (0,1,2 or 3)\n";
	cout << "IP_ADDR_FILE		\tIP Address file (use makefile for automation)\n";
	cout << "AES_SEED_INDEP		\tAES seed file independent\n";
	cout << "AES_SEED_PUB_NEXT	\tAES seed file public (2 parties, next)\n";
	cout << "AES_SEED_PUB_PREV	\tAES seed file public (2 parties, prev)\n";
	cout << endl;
	cout << "Report bugs to chencongcong@tongji.edu.cn" << endl;
	exit(-1);
}

double diff(timespec start, timespec end)
{
	timespec temp;

	if ((end.tv_nsec - start.tv_nsec) < 0)
	{
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else
	{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp.tv_sec + (double)temp.tv_nsec / NANOSECONDS_PER_SEC;
}

void deleteObjects()
{
	// close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}
