
#pragma once
#include "GraphConvLayer.h"
#include "Functionalities.h"
#include <time.h>
#include <random>
#include <math.h>
using namespace std;

GraphConvLayer::GraphConvLayer(GraphConvConfig *conf, int _layerNum)
    : Layer(_layerNum, conf->layerType),
      conf(conf->inputDim, conf->nodeSize),
      activations(conf->inputDim * conf->nodeSize),
      biases(conf->inputDim)
{
    initialize();
}

void GraphConvLayer::initialize()
{
    fill(biases.begin(), biases.end(), make_pair(0, 0));
}

void GraphConvLayer::printLayer()
{
    cout << "----------------------------------------------" << endl;
    cout << "(" << layerNum + 1 << ") GraphConv Layer\t  " << conf.inputDim << endl
         << "\t\t\t  " << conf.nodeSize << "\t\t (Node Size)" << endl;
}

void GraphConvLayer::forward(const RSSVectorMyType &inputActivation)
{
    log_print("GraphConv.forward");

    assert(testIdx.size() == NUM_TEST && "Test index is empty, please load the inference data first");
    assert(nodes.size() == NUM_NODES && "Nodes are empty, please load the inference data first");

    bool isLastLayer = (conf.nodeSize == NUM_TEST);

    clock_t start, finish;
    double Times;
    start = clock();

    size_t degree;
    RSSVectorMyType activationsTemp(conf.inputDim);

    // If the network is SGC and precompute node state is enabled, load the precomputed node states
    // Update: last layer is not precomputed
    if (NETWORK.compare("SGC") == 0 && PRECOMPUTE_NODE_STATE && !isLastLayer)
    {
        vector<RSSVectorMyType> states;
        loadNodeStates(states);
        
        if (!isLastLayer)
        {
            // All nodes
            for (size_t i = 0; i < NUM_NODES; ++i)
            {
                activationsTemp.clear();
                activationsTemp.resize(conf.inputDim);
                // read the states
                copyVectors(states[i], activationsTemp, conf.inputDim);
                for (size_t j = 0; j < conf.inputDim; ++j)
                    activations[i * conf.inputDim + j] = activationsTemp[j];

                if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                {
                    finish = clock();
                    Times = (double)(finish - start) / CLOCKS_PER_SEC;
                    printf("Progress[Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / NUM_NODES, i + 1, NUM_NODES, Times);
                    fflush(stdout);
                }
            }
        }
        else 
        {
            // Test nodes
            size_t index;
            for (size_t i = 0; i < conf.nodeSize; ++i)
            {
                activationsTemp.clear();
                activationsTemp.resize(conf.inputDim);
                index = testIdx[i];
                // read the states
                copyVectors(states[index], activationsTemp, conf.inputDim);
                for (size_t j = 0; j < conf.inputDim; ++j)
                    activations[i * conf.inputDim + j] = activationsTemp[j];

                if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                {
                    finish = clock();
                    Times = (double)(finish - start) / CLOCKS_PER_SEC;
                    printf("Progress[Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / conf.nodeSize, i + 1, conf.nodeSize, Times);
                    fflush(stdout);
                }
            }
        }

        if (PRINT_PROGRESS)
        {
            cout << endl
                 << "Progress [Layer " << layerNum + 1 << "]: Forward Completed" << endl;
        }
        return;
    }

    // convert inputActivation to a 2D array
    vector<RSSVectorMyType> inputActivation2D(NUM_NODES, RSSVectorMyType(conf.inputDim));
    for (size_t i = 0; i < NUM_NODES; ++i)
        copyVectors<RSSMyType>(inputActivation, inputActivation2D[i], i * conf.inputDim, conf.inputDim);

    // Compute node's activation
    RSSVectorMyType a(conf.inputDim), b(conf.inputDim), swReciRSS(conf.inputDim), coefNeighRSS(conf.inputDim);
    RSSVectorMyType tempInputActivation(conf.inputDim);

    // Generate base array
    if (BATCH_ARRAY_ACCESS)
    {
        vector<vector<myType>> baseArray(BASE_ARRAY_SIZE, vector<myType>(NUM_NODES * conf.inputDim, 0));
        RSSVectorMyType baseRandom(BASE_ARRAY_SIZE);
        funcGenerateBaseArrayVector(inputActivation2D, baseArray, baseRandom, NUM_NODES, BASE_ARRAY_SIZE, conf.inputDim);

        vector<vector<myType>> baseArrayInit;
        RSSVectorMyType baseRandomInit;
        bool coefBaseReady = false;

        auto ensureCoefNeigh = [&](size_t nodeIndex, size_t nodeDegree)
        {
            if (ENABLE_SAI)
                return;

            if (nodes[nodeIndex].coefNeigh.size() == nodeDegree)
                return;

            if (!coefBaseReady)
            {
                RSSVectorMyType swInvSqrt(NUM_NODES);
                for (size_t k = 0; k < NUM_NODES; ++k)
                    swInvSqrt[k] = nodes[k].swInvSqrt;

                baseArrayInit.assign(BASE_ARRAY_SIZE, vector<myType>(NUM_NODES, 0));
                baseRandomInit.resize(BASE_ARRAY_SIZE);
                funcGenerateBaseArray(swInvSqrt, baseArrayInit, baseRandomInit, BASE_ARRAY_SIZE, NUM_NODES);
                coefBaseReady = true;
            }

            nodes[nodeIndex].coefNeigh.resize(nodeDegree);
            RSSVectorMyType swInvSqrtSelf(nodeDegree, nodes[nodeIndex].swInvSqrt);
            funcBatchArrayAccess(baseArrayInit, baseRandomInit, nodes[nodeIndex].neighborIDs, nodes[nodeIndex].coefNeigh, BASE_ARRAY_SIZE, nodeDegree, NUM_NODES);

            // compute 1/sqrt{sw_i} * 1/sqrt{sw_j} * ew_{ij}
            funcDotProduct(swInvSqrtSelf, nodes[nodeIndex].coefNeigh, nodes[nodeIndex].coefNeigh, nodeDegree, true, FLOAT_PRECISION);
            funcDotProduct(nodes[nodeIndex].edgeWeights, nodes[nodeIndex].coefNeigh, nodes[nodeIndex].coefNeigh, nodeDegree, true, FLOAT_PRECISION);
        };

        size_t nodeBatchSize = BATCH_NODE_SIZE_ARRAY_ACCESS;
        if (nodeBatchSize == 0)
            nodeBatchSize = 1;

        if (!isLastLayer)
        {
            for (size_t batchStart = 0; batchStart < NUM_NODES; batchStart += nodeBatchSize)
            {
                size_t batchEnd = batchStart + nodeBatchSize;
                if (batchEnd > NUM_NODES)
                    batchEnd = NUM_NODES;

                size_t batchNodeCount = batchEnd - batchStart;
                vector<size_t> degrees(batchNodeCount, 0);
                vector<size_t> offsets(batchNodeCount + 1, 0);
                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t nodeIndex = batchStart + bi;
                    degrees[bi] = nodes[nodeIndex].neighborIDs.size();
                    offsets[bi + 1] = offsets[bi] + degrees[bi];
                }

                size_t totalDegree = offsets[batchNodeCount];
                RSSVectorMyType flatNeighborIDs(totalDegree);
                vector<RSSVectorMyType> flatNeighborStates(totalDegree, RSSVectorMyType(conf.inputDim));

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t nodeIndex = batchStart + bi;
                    for (size_t j = 0; j < degrees[bi]; ++j)
                        flatNeighborIDs[offsets[bi] + j] = nodes[nodeIndex].neighborIDs[j];
                }

                if (totalDegree > 0)
                    funcBatchArrayAccessVector(baseArray, baseRandom, flatNeighborIDs, flatNeighborStates, NUM_NODES, BASE_ARRAY_SIZE, totalDegree, conf.inputDim);

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t nodeIndex = batchStart + bi;
                    ensureCoefNeigh(nodeIndex, degrees[bi]);
                }

                RSSVectorMyType prodPacked;
                if (totalDegree > 0)
                {
                    size_t packedSize = totalDegree * conf.inputDim;
                    RSSVectorMyType coefPacked(packedSize), statePacked(packedSize);
                    prodPacked.resize(packedSize);

                    for (size_t bi = 0; bi < batchNodeCount; ++bi)
                    {
                        size_t nodeIndex = batchStart + bi;
                        for (size_t j = 0; j < degrees[bi]; ++j)
                        {
                            size_t globalNeighborOffset = offsets[bi] + j;
                            for (size_t d = 0; d < conf.inputDim; ++d)
                            {
                                size_t packedOffset = globalNeighborOffset * conf.inputDim + d;
                                coefPacked[packedOffset] = nodes[nodeIndex].coefNeigh[j];
                                statePacked[packedOffset] = flatNeighborStates[globalNeighborOffset][d];
                            }
                        }
                    }

                    funcDotProduct(coefPacked, statePacked, prodPacked, packedSize, true, FLOAT_PRECISION);
                }

                size_t selfPackedSize = batchNodeCount * conf.inputDim;
                RSSVectorMyType swReciPacked(selfPackedSize), inputPacked(selfPackedSize), aPacked(selfPackedSize);
                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t nodeIndex = batchStart + bi;
                    for (size_t d = 0; d < conf.inputDim; ++d)
                    {
                        size_t packedOffset = bi * conf.inputDim + d;
                        swReciPacked[packedOffset] = nodes[nodeIndex].swReci;
                        inputPacked[packedOffset] = inputActivation2D[nodeIndex][d];
                    }
                }
                funcDotProduct(swReciPacked, inputPacked, aPacked, selfPackedSize, true, FLOAT_PRECISION);

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t i = batchStart + bi;
                    activationsTemp.clear();
                    activationsTemp.resize(conf.inputDim);
                    degree = degrees[bi];

                    // compute a = 1/sw_{v_i} * inputStates_{v_i}
                    for (size_t d = 0; d < conf.inputDim; ++d)
                        activationsTemp[d] = aPacked[bi * conf.inputDim + d];

                    if (degree > 0)
                    {
                        for (size_t j = 0; j < degree; ++j)
                        {
                            for (size_t d = 0; d < conf.inputDim; ++d)
                            {
                                size_t packedOffset = (offsets[bi] + j) * conf.inputDim + d;
                                activationsTemp[d].first += prodPacked[packedOffset].first;
                                activationsTemp[d].second += prodPacked[packedOffset].second;
                            }
                        }
                    }

                    if (NETWORK.compare("GCN") == 0)
                        for (size_t j = 0; j < conf.inputDim; ++j)
                            activations[i * conf.inputDim + j] = activationsTemp[j] + biases[j];
                    else
                        for (size_t j = 0; j < conf.inputDim; ++j)
                            activations[i * conf.inputDim + j] = activationsTemp[j];

                    if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                    {
                        finish = clock();
                        Times = (double)(finish - start) / CLOCKS_PER_SEC;
                        printf("Progress [Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / NUM_NODES, i + 1, NUM_NODES, Times);
                        fflush(stdout);
                    }
                }
            }
        }
        else
        {
            for (size_t batchStart = 0; batchStart < NUM_TEST; batchStart += nodeBatchSize)
            {
                size_t batchEnd = batchStart + nodeBatchSize;
                if (batchEnd > NUM_TEST)
                    batchEnd = NUM_TEST;

                size_t batchNodeCount = batchEnd - batchStart;
                vector<size_t> nodeIndexes(batchNodeCount, 0);
                vector<size_t> degrees(batchNodeCount, 0);
                vector<size_t> offsets(batchNodeCount + 1, 0);
                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    nodeIndexes[bi] = testIdx[batchStart + bi];
                    degrees[bi] = nodes[nodeIndexes[bi]].neighborIDs.size();
                    offsets[bi + 1] = offsets[bi] + degrees[bi];
                }

                size_t totalDegree = offsets[batchNodeCount];
                RSSVectorMyType flatNeighborIDs(totalDegree);
                vector<RSSVectorMyType> flatNeighborStates(totalDegree, RSSVectorMyType(conf.inputDim));

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                    for (size_t j = 0; j < degrees[bi]; ++j)
                        flatNeighborIDs[offsets[bi] + j] = nodes[nodeIndexes[bi]].neighborIDs[j];

                if (totalDegree > 0)
                    funcBatchArrayAccessVector(baseArray, baseRandom, flatNeighborIDs, flatNeighborStates, NUM_NODES, BASE_ARRAY_SIZE, totalDegree, conf.inputDim);

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                    ensureCoefNeigh(nodeIndexes[bi], degrees[bi]);

                RSSVectorMyType prodPacked;
                if (totalDegree > 0)
                {
                    size_t packedSize = totalDegree * conf.inputDim;
                    RSSVectorMyType coefPacked(packedSize), statePacked(packedSize);
                    prodPacked.resize(packedSize);

                    for (size_t bi = 0; bi < batchNodeCount; ++bi)
                    {
                        size_t nodeIndex = nodeIndexes[bi];
                        for (size_t j = 0; j < degrees[bi]; ++j)
                        {
                            size_t globalNeighborOffset = offsets[bi] + j;
                            for (size_t d = 0; d < conf.inputDim; ++d)
                            {
                                size_t packedOffset = globalNeighborOffset * conf.inputDim + d;
                                coefPacked[packedOffset] = nodes[nodeIndex].coefNeigh[j];
                                statePacked[packedOffset] = flatNeighborStates[globalNeighborOffset][d];
                            }
                        }
                    }

                    funcDotProduct(coefPacked, statePacked, prodPacked, packedSize, true, FLOAT_PRECISION);
                }

                size_t selfPackedSize = batchNodeCount * conf.inputDim;
                RSSVectorMyType swReciPacked(selfPackedSize), inputPacked(selfPackedSize), aPacked(selfPackedSize);
                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t nodeIndex = nodeIndexes[bi];
                    for (size_t d = 0; d < conf.inputDim; ++d)
                    {
                        size_t packedOffset = bi * conf.inputDim + d;
                        swReciPacked[packedOffset] = nodes[nodeIndex].swReci;
                        inputPacked[packedOffset] = inputActivation2D[nodeIndex][d];
                    }
                }
                funcDotProduct(swReciPacked, inputPacked, aPacked, selfPackedSize, true, FLOAT_PRECISION);

                for (size_t bi = 0; bi < batchNodeCount; ++bi)
                {
                    size_t i = batchStart + bi;
                    size_t index = nodeIndexes[bi];

                    activationsTemp.clear();
                    activationsTemp.resize(conf.inputDim);
                    degree = degrees[bi];

                    // compute a = 1/sw_{v_i} * inputStates_{v_i}
                    for (size_t d = 0; d < conf.inputDim; ++d)
                        activationsTemp[d] = aPacked[bi * conf.inputDim + d];

                    if (degree > 0)
                    {
                        for (size_t j = 0; j < degree; ++j)
                        {
                            for (size_t d = 0; d < conf.inputDim; ++d)
                            {
                                size_t packedOffset = (offsets[bi] + j) * conf.inputDim + d;
                                activationsTemp[d].first += prodPacked[packedOffset].first;
                                activationsTemp[d].second += prodPacked[packedOffset].second;
                            }
                        }
                    }

                    if (NETWORK.compare("GCN") == 0)
                        for (size_t j = 0; j < conf.inputDim; ++j)
                            activations[i * conf.inputDim + j] = activationsTemp[j] + biases[j];
                    else
                        for (size_t j = 0; j < conf.inputDim; ++j)
                            activations[i * conf.inputDim + j] = activationsTemp[j];

                    if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                    {
                        finish = clock();
                        Times = (double)(finish - start) / CLOCKS_PER_SEC;
                        printf("Progress [Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / NUM_TEST, i + 1, NUM_TEST, Times);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    else
    {
        RSSVectorMyType neighborStates(conf.inputDim);

        vector<vector<myType>> baseArrayInit;
        RSSVectorMyType baseRandomInit;
        bool coefBaseReady = false;

        auto ensureCoefNeighNoBatch = [&](size_t nodeIndex, size_t nodeDegree)
        {
            if (ENABLE_SAI)
                return;

            if (nodes[nodeIndex].coefNeigh.size() == nodeDegree)
                return;

            if (!coefBaseReady)
            {
                RSSVectorMyType swInvSqrt(NUM_NODES);
                for (size_t k = 0; k < NUM_NODES; ++k)
                    swInvSqrt[k] = nodes[k].swInvSqrt;

                baseArrayInit.assign(BASE_ARRAY_SIZE, vector<myType>(NUM_NODES, 0));
                baseRandomInit.resize(BASE_ARRAY_SIZE);
                funcGenerateBaseArray(swInvSqrt, baseArrayInit, baseRandomInit, BASE_ARRAY_SIZE, NUM_NODES);
                coefBaseReady = true;
            }

            nodes[nodeIndex].coefNeigh.resize(nodeDegree);
            RSSVectorMyType swInvSqrtSelf(nodeDegree, nodes[nodeIndex].swInvSqrt);
            funcBatchArrayAccess(baseArrayInit, baseRandomInit, nodes[nodeIndex].neighborIDs, nodes[nodeIndex].coefNeigh, BASE_ARRAY_SIZE, nodeDegree, NUM_NODES);

            funcDotProduct(swInvSqrtSelf, nodes[nodeIndex].coefNeigh, nodes[nodeIndex].coefNeigh, nodeDegree, true, FLOAT_PRECISION);
            funcDotProduct(nodes[nodeIndex].edgeWeights, nodes[nodeIndex].coefNeigh, nodes[nodeIndex].coefNeigh, nodeDegree, true, FLOAT_PRECISION);
        };

        if (!isLastLayer)
        {
            for (size_t i = 0; i < NUM_NODES; ++i)
            {
                activationsTemp.clear();
                activationsTemp.resize(conf.inputDim);
                degree = nodes[i].neighborIDs.size();

                // compute a = 1/sw_{v_i} * inputStates_{v_i}
                fill(swReciRSS.begin(), swReciRSS.end(), nodes[i].swReci);
                copyVectors(inputActivation2D[i], tempInputActivation, conf.inputDim);
                funcDotProduct(swReciRSS, tempInputActivation, a, conf.inputDim, true, FLOAT_PRECISION);
                copyVectors(a, activationsTemp, conf.inputDim);

                ensureCoefNeighNoBatch(i, degree);

                RSSVectorMyType coefPacked, statePacked, prodPacked;
                if (degree > 0)
                {
                    size_t packedSize = degree * conf.inputDim;
                    coefPacked.resize(packedSize);
                    statePacked.resize(packedSize);
                    prodPacked.resize(packedSize);
                }

                for (size_t j = 0; j < degree; j++)
                {
                    // get the states of the neighbors
                    funcArrayAccessVector(inputActivation2D, nodes[i].neighborIDs[j], neighborStates, NUM_NODES, conf.inputDim);

                    for (size_t d = 0; d < conf.inputDim; ++d)
                    {
                        size_t packedOffset = j * conf.inputDim + d;
                        coefPacked[packedOffset] = nodes[i].coefNeigh[j];
                        statePacked[packedOffset] = neighborStates[d];
                    }
                }

                if (degree > 0)
                {
                    size_t packedSize = degree * conf.inputDim;
                    funcDotProduct(coefPacked, statePacked, prodPacked, packedSize, true, FLOAT_PRECISION);

                    for (size_t j = 0; j < degree; ++j)
                    {
                        for (size_t d = 0; d < conf.inputDim; ++d)
                        {
                            size_t packedOffset = j * conf.inputDim + d;
                            activationsTemp[d].first += prodPacked[packedOffset].first;
                            activationsTemp[d].second += prodPacked[packedOffset].second;
                        }
                    }
                }

                if (NETWORK.compare("GCN") == 0)
                    for (size_t j = 0; j < conf.inputDim; ++j)
                        activations[i * conf.inputDim + j] = activationsTemp[j] + biases[j];
                else
                    for (size_t j = 0; j < conf.inputDim; ++j)
                        activations[i * conf.inputDim + j] = activationsTemp[j];

                if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                {
                    finish = clock();
                    Times = (double)(finish - start) / CLOCKS_PER_SEC;
                    printf("Progress [Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / NUM_NODES, i + 1, NUM_NODES, Times);
                    fflush(stdout);
                }
            }
        }
        else
        {
            size_t index;
            // Test nodes
            for (size_t i = 0; i < NUM_TEST; ++i)
            {
                activationsTemp.clear();
                activationsTemp.resize(conf.inputDim);
                index = testIdx[i];
                degree = nodes[index].neighborIDs.size();

                // compute a = 1/sw_{v_i} * inputStates_{v_i}
                fill(swReciRSS.begin(), swReciRSS.end(), nodes[index].swReci);
                copyVectors(inputActivation2D[index], tempInputActivation, conf.inputDim);
                funcDotProduct(swReciRSS, tempInputActivation, a, conf.inputDim, true, FLOAT_PRECISION);
                copyVectors(a, activationsTemp, conf.inputDim);

                ensureCoefNeighNoBatch(index, degree);

                RSSVectorMyType coefPacked, statePacked, prodPacked;
                if (degree > 0)
                {
                    size_t packedSize = degree * conf.inputDim;
                    coefPacked.resize(packedSize);
                    statePacked.resize(packedSize);
                    prodPacked.resize(packedSize);
                }

                for (size_t j = 0; j < degree; j++)
                {
                    // get the states of the neighbors
                    funcArrayAccessVector(inputActivation2D, nodes[index].neighborIDs[j], neighborStates, NUM_NODES, conf.inputDim);

                    for (size_t d = 0; d < conf.inputDim; ++d)
                    {
                        size_t packedOffset = j * conf.inputDim + d;
                        coefPacked[packedOffset] = nodes[index].coefNeigh[j];
                        statePacked[packedOffset] = neighborStates[d];
                    }
                }

                if (degree > 0)
                {
                    size_t packedSize = degree * conf.inputDim;
                    funcDotProduct(coefPacked, statePacked, prodPacked, packedSize, true, FLOAT_PRECISION);

                    for (size_t j = 0; j < degree; ++j)
                    {
                        for (size_t d = 0; d < conf.inputDim; ++d)
                        {
                            size_t packedOffset = j * conf.inputDim + d;
                            activationsTemp[d].first += prodPacked[packedOffset].first;
                            activationsTemp[d].second += prodPacked[packedOffset].second;
                        }
                    }
                }

                if (NETWORK.compare("GCN") == 0)
                    for (size_t j = 0; j < conf.inputDim; ++j)
                        activations[i * conf.inputDim + j] = activationsTemp[j] + biases[j];
                else
                    for (size_t j = 0; j < conf.inputDim; ++j)
                        activations[i * conf.inputDim + j] = activationsTemp[j];

                if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                {
                    finish = clock();
                    Times = (double)(finish - start) / CLOCKS_PER_SEC;
                    printf("Progress [Layer %d]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", layerNum + 1, (i + 1) * 100.0 / NUM_TEST, i + 1, NUM_TEST, Times);
                    fflush(stdout);
                }
            }
        }
    }

    if (PRINT_PROGRESS)
    {
        cout << endl
             << "Progress [Layer " << layerNum + 1 << "]: Forward Completed" << endl;
    }
}

void GraphConvLayer::computeDelta(RSSVectorMyType &prevDelta)
{
    log_print("GraphConv.computeDelta");
}

void GraphConvLayer::updateEquations(const RSSVectorMyType &prevActivations)
{
    log_print("GraphConv.updateEquations");
    // GraphConvLayer only use in the inference phase, no need to update equations
}

void GraphConvLayer::loadNodeStates(vector<RSSVectorMyType> &states)
{
    string filename1, filename2;
    string path = NODE_STATE_PATH + "/node_state" + "_" + to_string(layerNum + 1);
    if (partyNum == PARTY_A)
    {
        filename1 = path + "_A";
        filename2 = path + "_B";
    }
    if (partyNum == PARTY_B)
    {
        filename1 = path + "_B";
        filename2 = path + "_C";
    }
    if (partyNum == PARTY_C)
    {
        filename1 = path + "_C";
        filename2 = path + "_A";
    }

    ifstream file_1(filename1), file_2(filename2);

    if (!file_1.is_open() || !file_2.is_open())
    {
        cerr << "Error: Unable to open one of the input files." << endl;
        return;
    }

    cout << "📝 Load node states from: " << filename1 << " | " << filename2 << endl;

    RSSVectorMyType temp(conf.inputDim);
    double first, second;
    string line_1, line_2;

    for (int i = 0; i < NUM_NODES; ++i)
    {
        getline(file_1, line_1);
        getline(file_2, line_2);
        istringstream iss_1(line_1);
        istringstream iss_2(line_2);

        for (size_t j = 0; j < conf.inputDim; ++j)
        {
            iss_1 >> first;
            iss_2 >> second;
            temp[j] = std::make_pair(floatToMyType(first), floatToMyType(second));
        }
        states.push_back(temp);
    }

    file_1.close();
    file_2.close();
}