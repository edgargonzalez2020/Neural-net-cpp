CXX_FLAGS = -std=c++11 -Wall -g 

HEADERS = ./include
main: ./src/main.cpp 
	g++ ./src/main.cpp ./src/NeuralNet.cpp -I${HEADERS} ${CXX_FLAGS}

