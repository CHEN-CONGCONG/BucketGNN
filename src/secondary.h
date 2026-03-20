#ifndef SECONDARY_H
#define SECONDARY_H

#pragma once
#include "globals.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "GlobalParams.h"
#include "LoadData.h"

/******************* Main test functions *******************/
void parseInputs(int argc, char *argv[]);
void test(NeuralNetwork *net);
double getAccuracy(NeuralNetwork *net);
void preload_network(NeuralNetwork *inference_net);
void printNetwork(NeuralNetwork *net);

void selectNetwork(string network, NeuralNetConfig *config);
void initResults(string outputDir);
void writeResults(string str, string outputDir, vector<double> results);

/********************* COMMUNICATION AND HELPERS *********************/
void start_m();
void end_m(std::string str, vector<double> &results);
void start_time();
vector<double> end_time(std::string str);
void start_rounds();
void end_rounds(std::string str);
vector<double> aggregateCommunication();
void print_usage(const char *bin);
double diff(timespec start, timespec end);
void deleteObjects();
#endif