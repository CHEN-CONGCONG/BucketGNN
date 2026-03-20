# Run make to see list of commands.

########################## Executable (BucketGNN.out) arguments ##########################
# arg[0]: BucketGNN.out
# arg[1]: Party number (0,1,2)
# arg[2]: File containing IP addresses of all parties
# arg[3]: AES key for local randomness generation (each party has a different file)
# arg[4]: AES key for common randomness generation (with next party)
# arg[5]: AES key for common randomness generation (with prev party)

# OPTIONAL ARGUMENTS: If supplied, these arguments take precedence over the ones in src/main.cpp
# 					  Refer to Makefile Parameters section below for details.
# arg[6]: Which network to use
# arg[7]: What dataset to use
# arg[8]: Which adversarial model
# arg[9]: Output folder for the results
# args[10]: Mode of execution (test, init, all, debug)

############################## Makefile parameters ######################################
# Location of OPENSSL installation if running over servers without sudo access
OPEN_SSL_LOC := /usr/lib
# RUN_TYPE {localhost, LAN or WAN} 
RUN_TYPE := $(if $(filter $(RUN_TYPE),localhost LAN WAN),$(RUN_TYPE),localhost)
# NETWORK {GCN, SGC, APPNP}
NETWORK := $(if $(filter $(NETWORK),GCN SGC APPNP),$(NETWORK),GCN)
# Dataset {pubmed, cora, citeseer, ogbn-arxiv, ogbn-products}
# 'make file_command DATASET=Cora' to run with cora dataset, default is cora
DATASET := $(if $(filter $(DATASET),pubmed cora citeseer ogbn-arxiv ogbn-products),$(DATASET),cora)
# Security {Semi-honest or Malicious} 
SECURITY:= $(if $(filter $(SECURITY),Semi-honest Malicious),$(SECURITY),Semi-honest)

# Output folder for the results, only used in 'make file_command' command
# if not specified, the default is output/$(DATASET)/Party*.txt
OUTPUT_FOLDER := $(if $(OUTPUT_FOLDER),$(OUTPUT_FOLDER),output/$(DATASET))

# Mode of execution {test, init, all}
# 'make file_command MODE=test' to run test, default is all
MODE := $(if $(filter $(MODE),test init all debug),$(MODE),all)
#########################################################################################


# 找到BucketGNN.out的进程， 可能多个，若存在则kill
PID := $(shell ps -ef | grep BucketGNN.out | grep -v grep | awk '{print $$2}')
ifneq ($(PID),)
$(shell kill -9 $(PID))
endif


#########################################################################################
CXX=g++
SRC_CPP_FILES     := $(wildcard src/*.cpp)
NETWORK_CPP_FILES := $(wildcard src/NeuralNetwork/*.cpp)
OBJ_CPP_FILES     := $(wildcard util/*.cpp)
OBJ_FILES    	  := $(patsubst src/%.cpp, src/%.o,$(SRC_CPP_FILES))
OBJ_FILES    	  += $(patsubst src/NeuralNetwork/%.cpp, src/NeuralNetwork/%.o,$(NETWORK_CPP_FILES))
OBJ_FILES    	  += $(patsubst util/%.cpp, util/%.o,$(OBJ_CPP_FILES))
HEADER_FILES       = $(wildcard src/*.h)

# FLAGS := -static -g -O0 -w -std=c++11 -pthread -msse4.1 -maes -msse2 -mpclmul -fpermissive -fpic
FLAGS := -O3 -w -std=c++17 -pthread -msse4.1 -maes -msse2 -mpclmul -fpic
LIBS := -lcrypto -lssl -lyaml-cpp
OBJ_INCLUDES := -I 'lib_eigen/' -I 'util/Miracl/' -I 'util/' -I 'src/' -I 'src/NeuralNetwork/' -I '$(OPEN_SSL_LOC)/include/'
BMR_INCLUDES := -L./ -L$(OPEN_SSL_LOC)/lib/ $(OBJ_INCLUDES) 


help: ## Run make or make help to see list of options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: BucketGNN.out ## Just compile the code

BucketGNN.out: $(OBJ_FILES)
	g++ $(FLAGS) -o $@ $(OBJ_FILES) $(BMR_INCLUDES) $(LIBS)

%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

clean: ## Run this to clean all files
	rm -rf BucketGNN.out
	rm -rf src/*.o util/*.o src/NeuralNetwork/*.o

################################# Remote runs ##########################################
terminal: BucketGNN.out ## Run this to print the output of (only) Party 0 to terminal
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >/dev/null &
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >/dev/null &
	./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC 
	@echo "Execution completed"

file: BucketGNN.out ## Run this to append the output of (only) Party 0 to file output/Party*.txt
	mkdir -p output
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >output/PartyC.txt &
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >output/PartyB.txt &
	./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC >output/PartyA.txt
	@echo "Execution completed"

file_command: BucketGNN.out ## Run this to use the run parameters specified in the makefile and append the output of Party * to file $(OUTPUT_FOLDER)/Party*.txt
	mkdir -p output
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC $(NETWORK) $(DATASET) $(SECURITY) $(OUTPUT_FOLDER) $(MODE) >$(OUTPUT_FOLDER)/PartyC.txt &
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB $(NETWORK) $(DATASET) $(SECURITY) $(OUTPUT_FOLDER) $(MODE) >$(OUTPUT_FOLDER)/PartyB.txt &
	./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC $(NETWORK) $(DATASET) $(SECURITY) $(OUTPUT_FOLDER) $(MODE) >$(OUTPUT_FOLDER)/PartyA.txt
	@echo "Execution completed"

valg: BucketGNN.out ## Run this to execute (only) Party 0 using valgrind. Change FLAGS to -O0.
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >/dev/null &
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >/dev/null &
	valgrind --tool=memcheck --leak-check=full --track-origins=yes --dsymutil=yes ./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC

command: BucketGNN.out ## Run this to use the run parameters specified in the makefile. 
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC $(NETWORK) $(DATASET) $(SECURITY) >/dev/null &
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB $(NETWORK) $(DATASET) $(SECURITY) >/dev/null &
	./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC $(NETWORK) $(DATASET) $(SECURITY) 
	@echo "Execution completed"


################################## tmux runs ############################################
zero: BucketGNN.out ## Run this to only execute Party 0 (useful for multiple terminal runs)
	./BucketGNN.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC 

one: BucketGNN.out ## Run this to only execute Party 1 (useful for multiple terminal runs)
	./BucketGNN.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB

two: BucketGNN.out ## Run this to only execute Party 2 (useful for multiple terminal runs)
	./BucketGNN.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC
#########################################################################################

.PHONY: help

