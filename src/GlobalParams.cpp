#include "GlobalParams.h"

using namespace std;

// define the global parameters
string DATASET_NAME = "";
string DATASET_PATH = "";
bool ENABLE_SAI = true;
bool SECGNN_MASK_METHOD = false;
bool POSITION_SWAP = true;

size_t NUM_CLASSES = 0;
size_t NUM_FEATURES = 0;
size_t NUM_NODES = 0;
size_t NUM_EDGES = 0;
size_t NUM_TRAIN = 0;
size_t NUM_VAL = 0;
size_t NUM_TEST = 0;
size_t MAX_DEGREE = 0;
size_t HIDDEN_LAYER_SIZE = 0;
size_t MINI_BATCH_SIZE = 0;
string PRELOAD_NETWORK_PATH = "";
bool LARGE_NETWORK = false;
bool WITH_NORMALIZATION = false;
int NO_OF_RUNS = 0;

string SECURITY_TYPE = "";
string NETWORK = "";

AESObject *aes_indep;
AESObject *aes_next;
AESObject *aes_prev;

bool MASK_NEIGHBOR = false;
bool SAVE_NODE_INIT = false;
bool REWRITE_NODE_INIT = false;
string NODE_INIT_PATH = "";
bool PRECOMPUTE_NODE_STATE = false;
string NODE_STATE_PATH = "";
bool BATCH_ARRAY_ACCESS = false;
size_t BASE_ARRAY_SIZE = 0;
size_t BATCH_NODE_SIZE_ARRAY_ACCESS = 1;
string NEIGHBOR_PADDING_MODE = "bucket";
string BUCKET_PADDING_STRATEGY = "exponential";
string BUCKET_MERGE_MODE = "full_bucket";
size_t BUCKET_MIN_NODES_K = 32;
size_t BUCKET_COUNT = 8;
size_t BUCKET_EXP_BASE = 2;
vector<size_t> BUCKET_BOUNDARIES;
bool EXPORT_MASK_DISTRIBUTION = false;
string MASK_DISTRIBUTION_OUTPUT_DIR = "output/mask_distribution";

void loadGlobalParams(const string &yamlFilePath)
{
    try
    {
        // load the yaml file
        YAML::Node config = YAML::LoadFile(yamlFilePath);

        const auto &dataset = config["dataset"];

        DATASET_NAME = dataset["name"].as<string>();
        DATASET_PATH = dataset["path"].as<string>();
        ENABLE_SAI = dataset["sai"].as<bool>();
        POSITION_SWAP = dataset["position_swap"].as<bool>();
        // Dataset parameters
        const auto &properties = dataset["properties"];

        NUM_CLASSES = properties["num_classes"].as<size_t>();
        NUM_FEATURES = properties["num_features"].as<size_t>();
        NUM_NODES = properties["num_nodes"].as<size_t>();
        NUM_EDGES = properties["num_edges"].as<size_t>();
        NUM_TRAIN = properties["num_train"].as<size_t>();
        NUM_VAL = properties["num_val"].as<size_t>();
        NUM_TEST = properties["num_test"].as<size_t>();
        MAX_DEGREE = properties["max_degree"].as<size_t>();

        // Network parameters
        const auto &network = dataset["network"];

        HIDDEN_LAYER_SIZE = network["hidden_layer_size"].as<size_t>();
        PRELOAD_NETWORK_PATH = network["preload_network_path"].as<string>();
        LARGE_NETWORK = network["large_network"].as<bool>();
        WITH_NORMALIZATION = network["with_normalization"].as<bool>();
        NO_OF_RUNS = network["no_of_runs"].as<int>();

        // Graph parameters
        const auto &graph = dataset["graph"];

        MASK_NEIGHBOR = graph["mask_neighbor"].as<bool>();
        SAVE_NODE_INIT = graph["save_node_init"].as<bool>();
        REWRITE_NODE_INIT = graph["rewrite_node_init"].as<bool>();
        NODE_INIT_PATH = graph["node_init_path"].as<string>();
        PRECOMPUTE_NODE_STATE = graph["precompute_node_state"].as<bool>();
        NODE_STATE_PATH = graph["node_state_path"].as<string>();
        BATCH_ARRAY_ACCESS = graph["batch_array_access"].as<bool>();
        BASE_ARRAY_SIZE = graph["base_array_size"].as<size_t>();
        BATCH_NODE_SIZE_ARRAY_ACCESS = graph["batch_node_size_array_access"] ? graph["batch_node_size_array_access"].as<size_t>() : 1;
        NEIGHBOR_PADDING_MODE = graph["neighbor_padding_mode"] ? graph["neighbor_padding_mode"].as<string>() : "bucket";

        const auto bucket = graph["bucket"];
        BUCKET_PADDING_STRATEGY = bucket && bucket["strategy"] ? bucket["strategy"].as<string>() :
                                 (graph["bucket_padding_strategy"] ? graph["bucket_padding_strategy"].as<string>() : "exponential");
        BUCKET_MERGE_MODE = bucket && bucket["merge_mode"] ? bucket["merge_mode"].as<string>() :
                    (graph["bucket_merge_mode"] ? graph["bucket_merge_mode"].as<string>() : "full_bucket");
        BUCKET_MIN_NODES_K = bucket && bucket["min_nodes_k"] ? bucket["min_nodes_k"].as<size_t>() :
                            (graph["bucket_min_nodes_k"] ? graph["bucket_min_nodes_k"].as<size_t>() : 32);
        BUCKET_COUNT = bucket && bucket["count"] ? bucket["count"].as<size_t>() :
                       (graph["bucket_count"] ? graph["bucket_count"].as<size_t>() : 8);
        BUCKET_EXP_BASE = bucket && bucket["exp_base"] ? bucket["exp_base"].as<size_t>() :
                          (graph["bucket_exp_base"] ? graph["bucket_exp_base"].as<size_t>() : 2);
        BUCKET_BOUNDARIES.clear();
        if (bucket && bucket["boundaries"])
            BUCKET_BOUNDARIES = bucket["boundaries"].as<vector<size_t>>();
        else if (graph["bucket_boundaries"])
            BUCKET_BOUNDARIES = graph["bucket_boundaries"].as<vector<size_t>>();
        EXPORT_MASK_DISTRIBUTION = graph["export_mask_distribution"] ? graph["export_mask_distribution"].as<bool>() : false;
        MASK_DISTRIBUTION_OUTPUT_DIR = graph["mask_distribution_output_dir"] ? graph["mask_distribution_output_dir"].as<string>() : "output/mask_distribution";

        if (graph["neighbor_padding_mode"])
        {
            string modeLower = NEIGHBOR_PADDING_MODE;
            transform(modeLower.begin(), modeLower.end(), modeLower.begin(), ::tolower);
            SECGNN_MASK_METHOD = (modeLower == "secgnn");
        }
        else if (SECGNN_MASK_METHOD)
        {
            NEIGHBOR_PADDING_MODE = "secgnn";
        }

        if (BATCH_NODE_SIZE_ARRAY_ACCESS == 0)
            BATCH_NODE_SIZE_ARRAY_ACCESS = 1;
        if (BUCKET_MIN_NODES_K == 0)
            BUCKET_MIN_NODES_K = 1;
        if (BUCKET_COUNT == 0)
            BUCKET_COUNT = 1;
        if (BUCKET_EXP_BASE < 2)
            BUCKET_EXP_BASE = 2;

        string mergeModeLower = BUCKET_MERGE_MODE;
        transform(mergeModeLower.begin(), mergeModeLower.end(), mergeModeLower.begin(), ::tolower);
        if (mergeModeLower != "full_bucket" && mergeModeLower != "partial_fill")
            BUCKET_MERGE_MODE = "full_bucket";
        else
            BUCKET_MERGE_MODE = mergeModeLower;

        cout << "----------------------------------------------" << endl;
        cout << "📝 Loaded global parameters from: " << yamlFilePath << endl;
        cout << "🔵 Dataset global parameters: \n"
             << " \t* " << "DATASET_NAME: " << DATASET_NAME << ", DATASET_PATH: " << DATASET_PATH << endl
             << " \t* " << "NUM_CLASSES: " << NUM_CLASSES << ", NUM_FEATURES: " << NUM_FEATURES << endl
             << " \t* " << "NUM_NODES: " << NUM_NODES << ", NUM_EDGES: " << NUM_EDGES << endl
             << " \t* " << "NUM_TRAIN: " << NUM_TRAIN << ", NUM_VAL: " << NUM_VAL << ", NUM_TEST: " << NUM_TEST << endl
             << " \t* " << "MAX_DEGREE: " << MAX_DEGREE << endl
             << endl;

        cout << "🔵 Neural network global parameters: \n"
             << " \t* " << "HIDDEN_LAYER_SIZE: " << HIDDEN_LAYER_SIZE << endl
             << " \t* " << "PRELOAD_NETWORK_PATH: " << PRELOAD_NETWORK_PATH << endl
             << " \t* " << "LARGE_NETWORK: " << (LARGE_NETWORK ? "true" : "false") << endl
             << " \t* " << "WITH_NORMALIZATION: " << (WITH_NORMALIZATION ? "true" : "false") << endl
             << " \t* " << "NO_OF_RUNS: " << NO_OF_RUNS << endl
             << endl;

        cout << "🔵 Graph global parameters: \n"
             << " \t* " << "MASK_NEIGHBOR: " << (MASK_NEIGHBOR ? "true" : "false") << endl
             << " \t* " << "SAVE_NODE_INIT: " << (SAVE_NODE_INIT ? "true" : "false") << endl
             << " \t* " << "REWRITE_NODE_INIT: " << (REWRITE_NODE_INIT ? "true" : "false") << endl
             << " \t* " << "NODE_INIT_PATH: " << NODE_INIT_PATH << endl
             << " \t* " << "PRECOMPUTE_NODE_STATE: " << (PRECOMPUTE_NODE_STATE ? "true" : "false") << endl
             << " \t* " << "NODE_STATE_PATH: " << NODE_STATE_PATH << endl
             << " \t* " << "BATCH_ARRAY_ACCESS: " << (BATCH_ARRAY_ACCESS ? "true" : "false") << endl
             << " \t* " << "BASE_ARRAY_SIZE: " << BASE_ARRAY_SIZE << endl
             << " \t* " << "BATCH_NODE_SIZE_ARRAY_ACCESS: " << BATCH_NODE_SIZE_ARRAY_ACCESS << endl
             << " \t* " << "NEIGHBOR_PADDING_MODE: " << NEIGHBOR_PADDING_MODE << endl
             << " \t* " << "BUCKET_PADDING_STRATEGY: " << BUCKET_PADDING_STRATEGY << endl
             << " \t* " << "BUCKET_MERGE_MODE: " << BUCKET_MERGE_MODE << endl
             << " \t* " << "BUCKET_MIN_NODES_K: " << BUCKET_MIN_NODES_K << endl
             << " \t* " << "BUCKET_COUNT: " << BUCKET_COUNT << endl
             << " \t* " << "BUCKET_EXP_BASE: " << BUCKET_EXP_BASE << endl
             << " \t* " << "BUCKET_BOUNDARIES_SIZE: " << BUCKET_BOUNDARIES.size() << endl
             << " \t* " << "EXPORT_MASK_DISTRIBUTION: " << (EXPORT_MASK_DISTRIBUTION ? "true" : "false") << endl
             << " \t* " << "MASK_DISTRIBUTION_OUTPUT_DIR: " << MASK_DISTRIBUTION_OUTPUT_DIR << endl
             << endl;

        cout << "✅ Loaded global parameters successfully!" << endl;
        cout << "----------------------------------------------" << endl;
    }
    catch (const YAML::Exception &e)
    {
        cerr << "❌ Error loading global parameters: " << e.what() << endl;
        exit(0);
    }
}

void setSecurityType(const string &security)
{
    assert(((security.compare("Semi-honest") == 0)) &&
           "Only Semi-honest security allowed");

    SECURITY_TYPE = security;
    cout << "🔒 Security type: [" << SECURITY_TYPE << "]" << endl;
}

void setNetwork(const string &network)
{
    assert(((network.compare("GCN") == 0) || (network.compare("SGC") == 0)) || (network.compare("APPNP") == 0) &&
                                                                                   "Only GCN, SGC, and APPNP network types allowed");

    NETWORK = network;
    cout << "🔵 Network type: [" << NETWORK << "]" << endl;
}

void initAESObjects(char *seed_indep, char *seed_next, char *seed_prev)
{
    aes_indep = new AESObject(seed_indep);
    aes_next = new AESObject(seed_next);
    aes_prev = new AESObject(seed_prev);
}

void deleteAESObjects()
{
    delete aes_indep;
    delete aes_next;
    delete aes_prev;
}