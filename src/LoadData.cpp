#include "LoadData.h"
#include "Functionalities.h"
#include <string>
#include <algorithm>
#include <set>
#include <fstream>
#include <map>

using namespace std;

RSSVectorMyType inferenceData, inferenceLabels;
vector<RSSVectorMyType> adjListVector, edgeWeightsVector;
vector<uint64_t> testIdx;

vector<Node> nodes;

static vector<size_t> buildBucketBoundaries(size_t maxDegree)
{
    vector<size_t> boundaries;

    if (maxDegree == 0)
    {
        boundaries.push_back(1);
        return boundaries;
    }

    if (BUCKET_PADDING_STRATEGY == "custom" && !BUCKET_BOUNDARIES.empty())
    {
        set<size_t> uniqueBoundaries;
        for (size_t value : BUCKET_BOUNDARIES)
            if (value > 0)
                uniqueBoundaries.insert(value);

        boundaries.assign(uniqueBoundaries.begin(), uniqueBoundaries.end());
        if (boundaries.empty())
            boundaries.push_back(maxDegree);
        else if (boundaries.back() < maxDegree)
            boundaries.push_back(maxDegree);

        return boundaries;
    }

    if (BUCKET_PADDING_STRATEGY == "equal" || BUCKET_PADDING_STRATEGY == "uniform")
    {
        size_t bucketCount = max((size_t)1, BUCKET_COUNT);
        size_t width = (maxDegree + bucketCount - 1) / bucketCount;
        for (size_t boundary = width; boundary < maxDegree; boundary += width)
            boundaries.push_back(boundary);
        boundaries.push_back(maxDegree);
        return boundaries;
    }

    size_t base = max((size_t)2, BUCKET_EXP_BASE);
    size_t boundary = 1;
    while (boundary < maxDegree)
    {
        boundaries.push_back(boundary);
        if (boundary > maxDegree / base)
        {
            boundary = maxDegree;
            break;
        }
        boundary *= base;
    }
    if (boundaries.empty() || boundaries.back() < maxDegree)
        boundaries.push_back(maxDegree);

    return boundaries;
}

static vector<size_t> computeMaskSizes(const vector<size_t> &degrees)
{
    vector<size_t> maskSizes(degrees.size(), 0);
    if (!MASK_NEIGHBOR)
        return maskSizes;

    string mode = NEIGHBOR_PADDING_MODE;
    transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

    if (mode == "none")
        return maskSizes;

    if (mode == "secgnn")
    {
        for (size_t index = 0; index < degrees.size(); ++index)
            maskSizes[index] = (MAX_DEGREE > degrees[index]) ? (MAX_DEGREE - degrees[index]) : 0;
        return maskSizes;
    }

    if (mode == "random")
    {
        for (size_t index = 0; index < degrees.size(); ++index)
        {
            size_t diffDegree = (MAX_DEGREE > degrees[index]) ? (MAX_DEGREE - degrees[index]) : 0;
            maskSizes[index] = getrandom(0, diffDegree);
            if (SECGNN_MASK_METHOD)
                maskSizes[index] = diffDegree;
        }
        return maskSizes;
    }

    vector<size_t> boundaries = buildBucketBoundaries(MAX_DEGREE);
    size_t bucketCount = boundaries.size();
    vector<size_t> bucketIndex(degrees.size(), 0);
    vector<size_t> bucketNodeCount(bucketCount, 0);

    for (size_t node = 0; node < degrees.size(); ++node)
    {
        size_t index = 0;
        while (index + 1 < bucketCount && degrees[node] > boundaries[index])
            ++index;

        bucketIndex[node] = index;
        bucketNodeCount[index]++;
    }

    auto moveSomeNodes = [&](size_t fromBucket, size_t toBucket, size_t maxMove)
    {
        if (fromBucket == toBucket || maxMove == 0)
            return (size_t)0;

        size_t moved = 0;
        for (size_t node = 0; node < bucketIndex.size() && moved < maxMove; ++node)
        {
            if (bucketIndex[node] == fromBucket)
            {
                bucketIndex[node] = toBucket;
                moved++;
            }
        }

        bucketNodeCount[fromBucket] -= moved;
        bucketNodeCount[toBucket] += moved;
        return moved;
    };

    auto moveAllNodes = [&](size_t fromBucket, size_t toBucket)
    {
        size_t moved = bucketNodeCount[fromBucket];
        if (moved == 0 || fromBucket == toBucket)
            return;
        moveSomeNodes(fromBucket, toBucket, moved);
    };

    size_t minNodesPerBucket = max((size_t)1, BUCKET_MIN_NODES_K);
    if (bucketCount > 1)
    {
        auto donorMoveAmount = [&](size_t donorCount, size_t deficit)
        {
            if (donorCount == 0)
                return (size_t)0;

            if (donorCount <= minNodesPerBucket)
            {
                if (deficit >= donorCount)
                    return donorCount;
                return (size_t)0;
            }

            size_t movable = donorCount - minNodesPerBucket;
            return min(deficit, movable);
        };

        if (BUCKET_MERGE_MODE == "partial_fill")
        {
            for (size_t bucket = 0; bucket + 1 < bucketCount; ++bucket)
            {
                if (bucketNodeCount[bucket] == 0 || bucketNodeCount[bucket] >= minNodesPerBucket)
                    continue;

                size_t deficit = minNodesPerBucket - bucketNodeCount[bucket];
                for (size_t donor = bucket; donor > 0 && deficit > 0; --donor)
                {
                    size_t previous = donor - 1;
                    if (bucketNodeCount[previous] == 0)
                        continue;

                    size_t movedPlan = donorMoveAmount(bucketNodeCount[previous], deficit);
                    if (movedPlan == 0)
                        continue;
                    size_t moved = moveSomeNodes(previous, bucket, movedPlan);
                    if (moved == 0)
                        break;
                    deficit -= moved;
                }

                if (bucketNodeCount[bucket] > 0 && bucketNodeCount[bucket] < minNodesPerBucket)
                    moveAllNodes(bucket, bucket + 1);
            }

            size_t lastBucket = bucketCount - 1;
            if (bucketNodeCount[lastBucket] > 0 && bucketNodeCount[lastBucket] < minNodesPerBucket)
            {
                size_t deficit = minNodesPerBucket - bucketNodeCount[lastBucket];
                for (size_t donor = lastBucket; donor > 0 && deficit > 0; --donor)
                {
                    size_t previous = donor - 1;
                    if (bucketNodeCount[previous] == 0)
                        continue;

                    size_t movedPlan = donorMoveAmount(bucketNodeCount[previous], deficit);
                    if (movedPlan == 0)
                        continue;
                    size_t moved = moveSomeNodes(previous, lastBucket, movedPlan);
                    if (moved == 0)
                        break;
                    deficit -= moved;
                }
            }
        }
        else
        {
            for (size_t bucket = 0; bucket + 1 < bucketCount; ++bucket)
            {
                if (bucketNodeCount[bucket] > 0 && bucketNodeCount[bucket] < minNodesPerBucket)
                    moveAllNodes(bucket, bucket + 1);
            }

            size_t lastBucket = bucketCount - 1;
            if (bucketNodeCount[lastBucket] > 0 && bucketNodeCount[lastBucket] < minNodesPerBucket)
            {
                for (size_t bucket = lastBucket; bucket > 0; --bucket)
                {
                    size_t previous = bucket - 1;
                    if (bucketNodeCount[previous] > 0)
                    {
                        moveAllNodes(previous, lastBucket);
                        if (bucketNodeCount[lastBucket] >= minNodesPerBucket)
                            break;
                    }
                }
            }
        }
    }

    for (size_t node = 0; node < degrees.size(); ++node)
    {
        size_t boundary = boundaries[bucketIndex[node]];
        if (boundary < degrees[node])
            boundary = degrees[node];
        maskSizes[node] = boundary - degrees[node];
    }

    return maskSizes;
}

static void exportMaskDistribution(const vector<size_t> &degrees, const vector<size_t> &maskSizes)
{
    if (!EXPORT_MASK_DISTRIBUTION)
        return;

    string outputDir = MASK_DISTRIBUTION_OUTPUT_DIR.empty() ? "output/mask_distribution" : MASK_DISTRIBUTION_OUTPUT_DIR;
    string mkdirCmd = "mkdir -p " + outputDir;
    system(mkdirCmd.c_str());

    string filename = outputDir + "/" + DATASET_NAME + "_party" + to_string(partyNum) + "_mask_distribution.csv";
    ofstream file(filename, ios::trunc);
    if (!file.is_open())
    {
        cerr << "Could not open mask distribution output file: " << filename << endl;
        return;
    }

    file << "node_id,original_degree,mask_size,masked_degree\n";
    size_t nodeCount = min(degrees.size(), maskSizes.size());
    map<size_t, size_t> bucketNodeCount;
    for (size_t node = 0; node < nodeCount; ++node)
    {
        size_t maskedDegree = degrees[node] + maskSizes[node];
        file << node << "," << degrees[node] << "," << maskSizes[node] << "," << maskedDegree << "\n";
        bucketNodeCount[maskedDegree] += 1;
    }
    file.close();

    string summaryFilename = outputDir + "/" + DATASET_NAME + "_party" + to_string(partyNum) + "_bucket_distribution.csv";
    ofstream summaryFile(summaryFilename, ios::trunc);
    if (!summaryFile.is_open())
    {
        cerr << "Could not open bucket distribution output file: " << summaryFilename << endl;
        return;
    }

    summaryFile << "bucket_degree,node_count\n";
    for (const auto &entry : bucketNodeCount)
        summaryFile << entry.first << "," << entry.second << "\n";
    summaryFile.close();

    if (partyNum == PARTY_A)
    {
        cout << "📝 Mask distribution exported: " << filename << endl;
        cout << "📝 Bucket distribution exported: " << summaryFilename << endl;
    }
}


void loadInferenceData()
{
    log_print("loadInferenceData");

    string inferenceDataPath = DATASET_PATH + "/test/";

    cout << "📝 Start load inference data, Dataset: " << DATASET_NAME
         << ", Path: " << inferenceDataPath << endl;

    string filename_node_data_next, filename_node_data_prev;
    string filename_node_label_next, filename_node_label_prev;
    string filename_adj_list_next, filename_adj_list_prev;
    string filename_edge_weights_next, filename_edge_weights_prev;

    if (partyNum == PARTY_A)
    {
        filename_node_data_next = inferenceDataPath + "node_data_A";
        filename_node_data_prev = inferenceDataPath + "node_data_B";
        filename_node_label_next = inferenceDataPath + "node_labels_A";
        filename_node_label_prev = inferenceDataPath + "node_labels_B";

        filename_adj_list_next = inferenceDataPath + "adj_list_A";
        filename_adj_list_prev = inferenceDataPath + "adj_list_B";
        filename_edge_weights_next = inferenceDataPath + "edge_weights_A";
        filename_edge_weights_prev = inferenceDataPath + "edge_weights_B";
    }

    if (partyNum == PARTY_B)
    {
        filename_node_data_next = inferenceDataPath + "node_data_B";
        filename_node_data_prev = inferenceDataPath + "node_data_C";
        filename_node_label_next = inferenceDataPath + "node_labels_B";
        filename_node_label_prev = inferenceDataPath + "node_labels_C";

        filename_adj_list_next = inferenceDataPath + "adj_list_B";
        filename_adj_list_prev = inferenceDataPath + "adj_list_C";
        filename_edge_weights_next = inferenceDataPath + "edge_weights_B";
        filename_edge_weights_prev = inferenceDataPath + "edge_weights_C";
    }

    if (partyNum == PARTY_C)
    {
        filename_node_data_next = inferenceDataPath + "node_data_C";
        filename_node_data_prev = inferenceDataPath + "node_data_A";
        filename_node_label_next = inferenceDataPath + "node_labels_C";
        filename_node_label_prev = inferenceDataPath + "node_labels_A";

        filename_adj_list_next = inferenceDataPath + "adj_list_C";
        filename_adj_list_prev = inferenceDataPath + "adj_list_A";
        filename_edge_weights_next = inferenceDataPath + "edge_weights_C";
        filename_edge_weights_prev = inferenceDataPath + "edge_weights_A";
    }

    // Load the inference nodes features
    double temp_next = 0, temp_prev = 0;
    ifstream f_next(filename_node_data_next);
    ifstream f_prev(filename_node_data_prev);
    // check if the files exist
    if (!f_next.is_open() || !f_prev.is_open())
    {
        cerr << "Could not open file: " << filename_node_data_next << " or " << filename_node_data_prev << endl;
        assert(false && "Could not open file");
    }

    for (size_t i = 0; i < NUM_NODES * NUM_FEATURES; ++i)
    {
        f_next >> temp_next;
        f_prev >> temp_prev;
        inferenceData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
    }
    f_next.close();
    f_prev.close();

    // Load the inference nodes labels
    ifstream g_next(filename_node_label_next);
    ifstream g_prev(filename_node_label_prev);
    if (!g_next.is_open() || !g_prev.is_open())
    {
        cerr << "Could not open file: " << filename_node_label_next << " or " << filename_node_label_prev << endl;
        assert(false && "Could not open file");
    }

    for (size_t i = 0; i < NUM_NODES * NUM_CLASSES; ++i)
    {
        g_next >> temp_next;
        g_prev >> temp_prev;
        inferenceLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
    }
    g_next.close();
    g_prev.close();

    // Load the adjacency list
    ifstream adj_next(filename_adj_list_next);
    ifstream adj_prev(filename_adj_list_prev);
    // check if the files exist
    if (!adj_next.is_open() || !adj_prev.is_open())
    {
        cerr << "Could not open file: " << filename_adj_list_next << " or " << filename_adj_list_prev << endl;
        assert(false && "Could not open file");
    }

    string adj_next_line, adj_prev_line;
    vector<size_t> degrees;
    RSSVectorMyType data;

    for (int i = 0; i < NUM_NODES; ++i)
    {
        data.clear();
        if (!getline(adj_next, adj_next_line) || !getline(adj_prev, adj_prev_line))
        {
            cerr << "Unexpected EOF while reading adjacency list at node " << i
                 << ", files: " << filename_adj_list_next << " and " << filename_adj_list_prev << endl;
            assert(false && "Unexpected EOF while reading adjacency list");
        }

        istringstream iss_first(adj_next_line);
        istringstream iss_second(adj_prev_line);
        while (true)
        {
            bool has_next = static_cast<bool>(iss_first >> temp_next);
            bool has_prev = static_cast<bool>(iss_second >> temp_prev);
            if (has_next != has_prev)
            {
                cerr << "Mismatched adjacency token count at node " << i
                     << ", files: " << filename_adj_list_next << " and " << filename_adj_list_prev << endl;
                assert(false && "Mismatched adjacency token count");
            }
            if (!has_next)
                break;
            data.push_back(std::make_pair(temp_next, temp_prev));
        }

        adjListVector.push_back(data);
        degrees.push_back(data.size());
    }

    vector<size_t> mask_sizes = computeMaskSizes(degrees);
    exportMaskDistribution(degrees, mask_sizes);
    for (size_t i = 0; i < NUM_NODES; ++i)
    {
        for (size_t j = 0; j < mask_sizes[i]; ++j)
            adjListVector[i].push_back(std::make_pair(0, 0));
    }

    adj_next.close();
    adj_prev.close();

    // Load the edge weights
    ifstream edge_next(filename_edge_weights_next);
    ifstream edge_prev(filename_edge_weights_prev);
    // check if the files exist
    if (!edge_next.is_open() || !edge_prev.is_open())
    {
        cerr << "Could not open file: " << filename_edge_weights_next << " or " << filename_edge_weights_prev << endl;
        assert(false && "Could not open file");
    }
    string edge_next_line, edge_prev_line;

    for (int i = 0; i < NUM_NODES; ++i)
    {
        data.clear();
        if (!getline(edge_next, edge_next_line) || !getline(edge_prev, edge_prev_line))
        {
            cerr << "Unexpected EOF while reading edge weights at node " << i
                 << ", files: " << filename_edge_weights_next << " and " << filename_edge_weights_prev << endl;
            assert(false && "Unexpected EOF while reading edge weights");
        }

        istringstream iss_first(edge_next_line);
        istringstream iss_second(edge_prev_line);
        while (true)
        {
            bool has_next = static_cast<bool>(iss_first >> temp_next);
            bool has_prev = static_cast<bool>(iss_second >> temp_prev);
            if (has_next != has_prev)
            {
                cerr << "Mismatched edge-weight token count at node " << i
                     << ", files: " << filename_edge_weights_next << " and " << filename_edge_weights_prev << endl;
                assert(false && "Mismatched edge-weight token count");
            }
            if (!has_next)
                break;
            data.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
        }

        // mask the degree
        for (size_t j = 0; j < mask_sizes[i]; ++j)
            data.push_back(std::make_pair(0, 0));

        edgeWeightsVector.push_back(data);
    }
    edge_next.close();
    edge_prev.close();

    // load test nodes ids
    string filename_test_idx = inferenceDataPath + "test_idx";
    ifstream f_test_idx(filename_test_idx);

    // check if the files exist
    if (!f_test_idx.is_open())
    {
        cerr << "Could not open file: " << filename_test_idx << endl;
        assert(false && "Could not open file");
    }
    size_t temp;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        f_test_idx >> temp;
        testIdx.push_back(temp);
    }
    f_test_idx.close();

    // print_vector(inferenceData, "FLOAT", "inferenceData:", NUM_NODES * NUM_FEATURES);
    // print_vector(inferenceLabels, "FLOAT", "inferenceLabels:", NUM_NODES * NUM_CLASSES);

    cout << "✅ Finished loading inference data, size: " << NUM_NODES
         << ", adjaceny list size: " << adjListVector.size()
         << ", edge weights size: " << edgeWeightsVector.size() << endl;

    processGraphNodes();
}

void processGraphNodes()
{
    log_print("processGraphNodes");

    assert(adjListVector.size() == NUM_NODES && "adjListVector is not initialized, please load inference data first!");
    assert(edgeWeightsVector.size() == NUM_NODES && "edgeWeightsVector is not initialized, please load inference data first!");

    cout << "🔨 Start processing graph nodes..." << endl;

    clock_t start, finish;
    double Times;
    start = clock();

    nodes.resize(NUM_NODES);

    // Initialize the graph nodes
    RSSVectorMyType oneRSS(NUM_NODES);
    vector<myType> one(NUM_NODES, floatToMyType(1));
    funcGetShares(oneRSS, one);

    RSSVectorMyType sw(NUM_NODES);
    size_t degree = 0;
    double sw_temp;

    for (int i = 0; i < NUM_NODES; ++i)
    {
        degree = adjListVector[i].size();
        nodes[i].feature.resize(NUM_FEATURES);
        nodes[i].neighborIDs.resize(degree);
        nodes[i].edgeWeights.resize(degree);
        nodes[i].groundTruth.resize(NUM_CLASSES);

        copyVectors<RSSMyType>(inferenceData, nodes[i].feature, i * NUM_FEATURES, NUM_FEATURES);
        copyVectors<RSSMyType>(adjListVector[i], nodes[i].neighborIDs, degree);
        copyVectors<RSSMyType>(edgeWeightsVector[i], nodes[i].edgeWeights, degree);
        copyVectors<RSSMyType>(inferenceLabels, nodes[i].groundTruth, i * NUM_CLASSES, NUM_CLASSES);

        // compute sw_{v_i} = 1 + \sum_{j in [0,d_i-1]} ew_{ij}
        for (size_t j = 0; j < degree; ++j)
        {
            sw[i].first += edgeWeightsVector[i][j].first;
            sw[i].second += edgeWeightsVector[i][j].second;
        }
        sw[i].first += oneRSS[i].first;
        sw[i].second += oneRSS[i].second;
        nodes[i].sw = sw[i];

        if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
        {
            finish = clock();
            Times = (double)(finish - start) / CLOCKS_PER_SEC;
            printf("Progress [Process Node]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", (i + 1) * 100.0 / NUM_NODES, i + 1, NUM_NODES, Times);
            fflush(stdout);
        }
    }

    // compute 1/sw
    RSSVectorMyType swReci(NUM_NODES), swInvSqrt(NUM_NODES);
    funcInv(sw, swReci, NUM_NODES);

    // compute 1/sqrt{sw}
    funcInvSqrt(sw, swInvSqrt, NUM_NODES);

    for (int i = 0; i < NUM_NODES; ++i)
    {
        nodes[i].swReci = swReci[i];
        nodes[i].swInvSqrt = swInvSqrt[i];
    }

    finish = clock();
    Times = (double)(finish - start) / CLOCKS_PER_SEC;

    if (PRINT_PROGRESS)
        cout << endl;
    cout << "✅ Finished processing graph nodes, execution time: " << Times << " s" << endl;
}

void initNodes()
{
    log_print("initNodes");

    assert(nodes.size() == NUM_NODES && "Nodes are not initialized, please process graph nodes first!");

    cout << "🔨 Start initializing nodes..." << endl;

    RSSMyType neighborID;
    clock_t start, finish;
    double Times;
    start = clock();

    string filename1, filename2;
    bool fileExists = true;

    if (SAVE_NODE_INIT)
    {
        // if the directory does not exist, create it
        string cmd = "mkdir -p " + NODE_INIT_PATH;
        system(cmd.c_str());

        filename1 = NODE_INIT_PATH + "/node_init_" + to_string(partyNum) + "_next";
        filename2 = NODE_INIT_PATH + "/node_init_" + to_string(partyNum) + "_prev";

        ifstream file_1(filename1);
        ifstream file_2(filename2);
        fileExists = file_1.good() && file_2.good();
        file_1.close();
        file_2.close();

        if (!fileExists)
        {
            ofstream file_1(filename1, ios::trunc);
            ofstream file_2(filename2, ios::trunc);
            file_1.close();
            file_2.close();
        }
        else if (REWRITE_NODE_INIT)
        {
            cout << "❗️ Rewite node init files: " << filename1 << " | " << filename2 << endl;
            // remove the existing files
            remove(filename1.c_str());
            remove(filename2.c_str());

            ofstream file_1(filename1, ios::trunc);
            ofstream file_2(filename2, ios::trunc);
            file_1.close();
            file_2.close();
            fileExists = false;
        }
    }

    auto computeCoefNeighOptimized = [&]()
    {
        RSSVectorMyType swInvSqrt(NUM_NODES);
        for (size_t i = 0; i < NUM_NODES; ++i)
            swInvSqrt[i] = nodes[i].swInvSqrt;

        vector<size_t> degrees(NUM_NODES, 0);

        if (BATCH_ARRAY_ACCESS)
        {
            vector<vector<myType>> baseArray(BASE_ARRAY_SIZE, vector<myType>(NUM_NODES, 0));
            RSSVectorMyType baseRandom(BASE_ARRAY_SIZE);
            funcGenerateBaseArray(swInvSqrt, baseArray, baseRandom, BASE_ARRAY_SIZE, NUM_NODES);

            for (size_t i = 0; i < NUM_NODES; ++i)
            {
                degrees[i] = adjListVector[i].size();
                nodes[i].coefNeigh.resize(degrees[i]);
            }

            size_t accessBatchSize = BATCH_NODE_SIZE_ARRAY_ACCESS;
            if (accessBatchSize == 0)
                accessBatchSize = 1;

            for (size_t batchStart = 0; batchStart < NUM_NODES; batchStart += accessBatchSize)
            {
                size_t batchEnd = batchStart + accessBatchSize;
                if (batchEnd > NUM_NODES)
                    batchEnd = NUM_NODES;

                size_t totalDegree = 0;
                for (size_t i = batchStart; i < batchEnd; ++i)
                    totalDegree += degrees[i];

                if (totalDegree == 0)
                    continue;

                RSSVectorMyType flatNeighborIDs(totalDegree), flatCoef(totalDegree);
                vector<size_t> offsets(batchEnd - batchStart + 1, 0);

                for (size_t i = batchStart; i < batchEnd; ++i)
                {
                    size_t localIndex = i - batchStart;
                    offsets[localIndex + 1] = offsets[localIndex] + degrees[i];
                    for (size_t j = 0; j < degrees[i]; ++j)
                        flatNeighborIDs[offsets[localIndex] + j] = adjListVector[i][j];
                }

                funcBatchArrayAccess(baseArray, baseRandom, flatNeighborIDs, flatCoef, BASE_ARRAY_SIZE, totalDegree, NUM_NODES);

                for (size_t i = batchStart; i < batchEnd; ++i)
                {
                    size_t localIndex = i - batchStart;
                    for (size_t j = 0; j < degrees[i]; ++j)
                        nodes[i].coefNeigh[j] = flatCoef[offsets[localIndex] + j];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < NUM_NODES; ++i)
            {
                degrees[i] = adjListVector[i].size();
                nodes[i].coefNeigh.resize(degrees[i]);

                for (size_t j = 0; j < degrees[i]; ++j)
                {
                    neighborID = adjListVector[i][j];
                    funcArrayAccess(swInvSqrt, neighborID, nodes[i].coefNeigh[j], NUM_NODES);
                }
            }
        }

        size_t nodeBatchSize = BATCH_NODE_SIZE_ARRAY_ACCESS;
        if (nodeBatchSize == 0)
            nodeBatchSize = 1;

        for (size_t batchStart = 0; batchStart < NUM_NODES; batchStart += nodeBatchSize)
        {
            size_t batchEnd = batchStart + nodeBatchSize;
            if (batchEnd > NUM_NODES)
                batchEnd = NUM_NODES;

            size_t totalDegree = 0;
            for (size_t i = batchStart; i < batchEnd; ++i)
                totalDegree += degrees[i];

            if (totalDegree == 0)
                continue;

            RSSVectorMyType swSelfPacked(totalDegree), coefPacked(totalDegree), edgePacked(totalDegree), tempPacked(totalDegree), resultPacked(totalDegree);
            size_t offset = 0;
            for (size_t i = batchStart; i < batchEnd; ++i)
            {
                for (size_t j = 0; j < degrees[i]; ++j)
                {
                    swSelfPacked[offset] = nodes[i].swInvSqrt;
                    coefPacked[offset] = nodes[i].coefNeigh[j];
                    edgePacked[offset] = edgeWeightsVector[i][j];
                    ++offset;
                }
            }

            funcDotProduct(swSelfPacked, coefPacked, tempPacked, totalDegree, true, FLOAT_PRECISION);
            funcDotProduct(edgePacked, tempPacked, resultPacked, totalDegree, true, FLOAT_PRECISION);

            offset = 0;
            for (size_t i = batchStart; i < batchEnd; ++i)
            {
                for (size_t j = 0; j < degrees[i]; ++j)
                    nodes[i].coefNeigh[j] = resultPacked[offset++];

                if (PRINT_PROGRESS && i % PRINT_INTERVAL == 0)
                {
                    finish = clock();
                    Times = (double)(finish - start) / CLOCKS_PER_SEC;
                    printf("Progress [Init Node]: %2.4f%% | [%d/%d] | execution time : %4.4f s\r", (i + 1) * 100.0 / NUM_NODES, i + 1, NUM_NODES, Times);
                    fflush(stdout);
                }
            }
        }
    };

    // compute ew_{ij} * 1/sqrt{sw_i} * 1/sqrt{sw_j}
    if (SAVE_NODE_INIT && !fileExists)
    {
        computeCoefNeighOptimized();

        ofstream file_1(filename1, ios::app);
        ofstream file_2(filename2, ios::app);
        for (size_t i = 0; i < NUM_NODES; ++i)
        {
            for (size_t j = 0; j < nodes[i].coefNeigh.size(); ++j)
            {
                file_1 << nodes[i].coefNeigh[j].first << " ";
                file_2 << nodes[i].coefNeigh[j].second << " ";
            }
            file_1 << endl;
            file_2 << endl;
        }
        file_1.close();
        file_2.close();
    }
    else if (SAVE_NODE_INIT && fileExists)
    { // read from the files
        ifstream file_1(filename1);
        ifstream file_2(filename2);
        myType first, second;

        string line_1, line_2;
        for (int i = 0; i < NUM_NODES; ++i)
        {
            line_1.clear();
            line_2.clear();

            getline(file_1, line_1);
            getline(file_2, line_2);
            istringstream iss_1(line_1);
            istringstream iss_2(line_2);

            while (iss_1 >> first && iss_2 >> second)
                nodes[i].coefNeigh.push_back(std::make_pair(first, second));
        }

        file_1.close();
        file_2.close();
    }
    else
    {
        computeCoefNeighOptimized();
    }
    finish = clock();
    Times = (double)(finish - start) / CLOCKS_PER_SEC;

    if (PRINT_PROGRESS)
        cout << endl;
    cout << "✅ Finished initializing nodes, execution time: " << Times << " s" << endl;
}

void readInferenceMiniBatch(NeuralNetwork *inference_net)
{
    log_print("readInferenceMiniBatch");

    // load the inference nodes features
    for (int i = 0; i < NUM_FEATURES * NUM_NODES; ++i)
        inference_net->inputData[i] = inferenceData[i];

    // load the inference nodes labels
    size_t index;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            index = testIdx[i] * NUM_CLASSES + j;
            inference_net->outputData[i * NUM_CLASSES + j] = inferenceLabels[index];
        }
    }
}

void loadPretrainedModel(NeuralNetwork *net, const string &network)
{
    log_print("loadPretrainedModel");

    // transform the network name to lowercase
    string network_lower = network;
    std::transform(network_lower.begin(), network_lower.end(), network_lower.begin(), ::tolower);

    string modelPath = PRELOAD_NETWORK_PATH + "/" + network_lower + "/run_" + to_string(NO_OF_RUNS) + "/";
    cout << "📣 Start loading pre-trained model, path: " << modelPath << endl;

    string filename_fc_weight_1_next, filename_fc_weight_1_prev;
    string filename_fc_bias_1_next, filename_fc_bias_1_prev;
    string filename_fc_weight_2_next, filename_fc_weight_2_prev;
    string filename_fc_bias_2_next, filename_fc_bias_2_prev;

    if (partyNum == PARTY_A)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_A";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_B";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_A";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_B";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_A";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_B";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_A";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_B";
    }

    if (partyNum == PARTY_B)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_B";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_C";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_B";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_C";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_B";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_C";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_B";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_C";
    }

    if (partyNum == PARTY_C)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_C";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_A";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_C";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_A";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_C";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_A";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_C";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_A";
    }

    int layer_1_index, layer_2_index;
    if (network_lower.compare("gcn") == 0)
    {
        if (POSITION_SWAP) {
            layer_1_index = 0;
            layer_2_index = 3;
        } else {
            layer_1_index = 1;
            layer_2_index = 4;
        }
    }
    else
    {
        cout << "❌ Invalid network name: " << network << endl;
        exit(-1);
    }
    cout << "📣 Start loading pre-trained model, network: " << network_lower 
         << ", layer 1 index: " << layer_1_index << ", layer 2 index: " << layer_2_index << endl;

    /**************************  FC (layer 1)  **************************/
    double temp_next = 0, temp_prev = 0;
    ifstream f_fc_weight_1_next(filename_fc_weight_1_next);
    ifstream f_fc_weight_1_prev(filename_fc_weight_1_prev);
    // check if the files exist
    if (!f_fc_weight_1_next.is_open() || !f_fc_weight_1_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_weight_1_next << " or " << filename_fc_weight_1_prev << endl;
        assert(false && "Could not open file");
    }

    for (int column = 0; column < HIDDEN_LAYER_SIZE; ++column)
    {
        for (int row = 0; row < NUM_FEATURES; ++row)
        {
            f_fc_weight_1_next >> temp_next;
            f_fc_weight_1_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_1_index])->getWeights())[HIDDEN_LAYER_SIZE * row + column] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }
    f_fc_weight_1_next.close();
    f_fc_weight_1_prev.close();

    ifstream f_fc_bias_1_next(filename_fc_bias_1_next);
    ifstream f_fc_bias_1_prev(filename_fc_bias_1_prev);
    // check if the files exist
    if (!f_fc_bias_1_next.is_open() || !f_fc_bias_1_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_bias_1_next << " or " << filename_fc_bias_1_prev << endl;
        assert(false && "Could not open file");
    }

    if (network_lower.compare("gcn") == 0)
    {
        if (POSITION_SWAP) {
            for (int i = 0; i < HIDDEN_LAYER_SIZE; ++i)
            {
                f_fc_bias_1_next >> temp_next;
                f_fc_bias_1_prev >> temp_prev;
                (*((GraphConvLayer *)net->layers[layer_1_index + 1])->getBias())[i] =
                    std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
            }
        } else {
            for (int i = 0; i < HIDDEN_LAYER_SIZE; ++i)
            {
                f_fc_bias_1_next >> temp_next;
                f_fc_bias_1_prev >> temp_prev;
                (*((GraphConvLayer *)net->layers[layer_1_index - 1])->getBias())[i] =
                    std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
            }
        }
    }
    else
    {
        for (int i = 0; i < HIDDEN_LAYER_SIZE; ++i)
        {
            f_fc_bias_1_next >> temp_next;
            f_fc_bias_1_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_1_index])->getBias())[i] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }

    f_fc_bias_1_next.close();
    f_fc_bias_1_prev.close();

    /**************************  FC (layer 2)  **************************/
    ifstream f_fc_weight_2_next(filename_fc_weight_2_next);
    ifstream f_fc_weight_2_prev(filename_fc_weight_2_prev);
    // check if the files exist
    if (!f_fc_weight_2_next.is_open() || !f_fc_weight_2_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_weight_2_next << " or " << filename_fc_weight_2_prev << endl;
        assert(false && "Could not open file");
    }

    for (int column = 0; column < NUM_CLASSES; ++column)
    {
        for (int row = 0; row < HIDDEN_LAYER_SIZE; ++row)
        {
            f_fc_weight_2_next >> temp_next;
            f_fc_weight_2_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_2_index])->getWeights())[NUM_CLASSES * row + column] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }

    f_fc_weight_2_next.close();
    f_fc_weight_2_prev.close();

    ifstream f_fc_bias_2_next(filename_fc_bias_2_next);
    ifstream f_fc_bias_2_prev(filename_fc_bias_2_prev);
    // check if the files exist
    if (!f_fc_bias_2_next.is_open() || !f_fc_bias_2_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_bias_2_next << " or " << filename_fc_bias_2_prev << endl;
        assert(false && "Could not open file");
    }

    if (network_lower.compare("gcn") == 0)
    {
        if (POSITION_SWAP) {
            for (int i = 0; i < NUM_CLASSES; ++i)
            {
                f_fc_bias_2_next >> temp_next;
                f_fc_bias_2_prev >> temp_prev;
                (*((GraphConvLayer *)net->layers[layer_2_index + 1])->getBias())[i] =
                    std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
            }
        } else {
            for (int i = 0; i < NUM_CLASSES; ++i)
            {
                f_fc_bias_2_next >> temp_next;
                f_fc_bias_2_prev >> temp_prev;
                (*((GraphConvLayer *)net->layers[layer_2_index - 1])->getBias())[i] =
                    std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
            }
        }
    }
    else
    {
        for (int i = 0; i < NUM_CLASSES; ++i)
        {
            f_fc_bias_2_next >> temp_next;
            f_fc_bias_2_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_2_index])->getBias())[i] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }

    f_fc_bias_2_next.close();
    f_fc_bias_2_prev.close();

    cout << "🥳 Finished loading pre-trained model!" << endl;
}

void loadInitModel(NeuralNetwork *net, const string &network)
{
    log_print("loadPretrainedModel");

    // transform the network name to lowercase
    string network_lower = network;
    std::transform(network_lower.begin(), network_lower.end(), network_lower.begin(), ::tolower);

    string modelPath = PRELOAD_NETWORK_PATH + "/" + network_lower + "/init_run_" + to_string(NO_OF_RUNS) + "/";
    cout << "📣 Start loading init model, path: " << modelPath << endl;

    string filename_fc_weight_1_next, filename_fc_weight_1_prev;
    string filename_fc_bias_1_next, filename_fc_bias_1_prev;
    string filename_fc_weight_2_next, filename_fc_weight_2_prev;
    string filename_fc_bias_2_next, filename_fc_bias_2_prev;

    if (partyNum == PARTY_A)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_A";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_B";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_A";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_B";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_A";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_B";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_A";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_B";
    }

    if (partyNum == PARTY_B)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_B";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_C";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_B";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_C";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_B";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_C";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_B";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_C";
    }

    if (partyNum == PARTY_C)
    {
        filename_fc_weight_1_next = modelPath + "fc_weight_1_C";
        filename_fc_weight_1_prev = modelPath + "fc_weight_1_A";
        filename_fc_bias_1_next = modelPath + "fc_bias_1_C";
        filename_fc_bias_1_prev = modelPath + "fc_bias_1_A";

        filename_fc_weight_2_next = modelPath + "fc_weight_2_C";
        filename_fc_weight_2_prev = modelPath + "fc_weight_2_A";
        filename_fc_bias_2_next = modelPath + "fc_bias_2_C";
        filename_fc_bias_2_prev = modelPath + "fc_bias_2_A";
    }

    int layer_1_index=0, layer_2_index=2;
    
    cout << "📣 Start loading init model, network: " << network_lower 
         << ", layer 1 index: " << layer_1_index << ", layer 2 index: " << layer_2_index << endl;

    /**************************  FC (layer 1)  **************************/
    double temp_next = 0, temp_prev = 0;
    ifstream f_fc_weight_1_next(filename_fc_weight_1_next);
    ifstream f_fc_weight_1_prev(filename_fc_weight_1_prev);
    // check if the files exist
    if (!f_fc_weight_1_next.is_open() || !f_fc_weight_1_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_weight_1_next << " or " << filename_fc_weight_1_prev << endl;
        assert(false && "Could not open file");
    }

    for (int column = 0; column < HIDDEN_LAYER_SIZE; ++column)
    {
        for (int row = 0; row < NUM_FEATURES; ++row)
        {
            f_fc_weight_1_next >> temp_next;
            f_fc_weight_1_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_1_index])->getWeights())[HIDDEN_LAYER_SIZE * row + column] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }
    f_fc_weight_1_next.close();
    f_fc_weight_1_prev.close();

    ifstream f_fc_bias_1_next(filename_fc_bias_1_next);
    ifstream f_fc_bias_1_prev(filename_fc_bias_1_prev);
    // check if the files exist
    if (!f_fc_bias_1_next.is_open() || !f_fc_bias_1_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_bias_1_next << " or " << filename_fc_bias_1_prev << endl;
        assert(false && "Could not open file");
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; ++i)
    {
        f_fc_bias_1_next >> temp_next;
        f_fc_bias_1_prev >> temp_prev;
        (*((FCLayer *)net->layers[layer_1_index])->getBias())[i] =
            std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }
    

    f_fc_bias_1_next.close();
    f_fc_bias_1_prev.close();

    /**************************  FC (layer 2)  **************************/
    ifstream f_fc_weight_2_next(filename_fc_weight_2_next);
    ifstream f_fc_weight_2_prev(filename_fc_weight_2_prev);
    // check if the files exist
    if (!f_fc_weight_2_next.is_open() || !f_fc_weight_2_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_weight_2_next << " or " << filename_fc_weight_2_prev << endl;
        assert(false && "Could not open file");
    }

    for (int column = 0; column < NUM_CLASSES; ++column)
    {
        for (int row = 0; row < HIDDEN_LAYER_SIZE; ++row)
        {
            f_fc_weight_2_next >> temp_next;
            f_fc_weight_2_prev >> temp_prev;
            (*((FCLayer *)net->layers[layer_2_index])->getWeights())[NUM_CLASSES * row + column] =
                std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
        }
    }

    f_fc_weight_2_next.close();
    f_fc_weight_2_prev.close();

    ifstream f_fc_bias_2_next(filename_fc_bias_2_next);
    ifstream f_fc_bias_2_prev(filename_fc_bias_2_prev);
    // check if the files exist
    if (!f_fc_bias_2_next.is_open() || !f_fc_bias_2_prev.is_open())
    {
        cerr << "Could not open file: " << filename_fc_bias_2_next << " or " << filename_fc_bias_2_prev << endl;
        assert(false && "Could not open file");
    }

    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        f_fc_bias_2_next >> temp_next;
        f_fc_bias_2_prev >> temp_prev;
        (*((FCLayer *)net->layers[layer_2_index])->getBias())[i] =
            std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_fc_bias_2_next.close();
    f_fc_bias_2_prev.close();

    cout << "🥳 Finished loading init model!" << endl;
}