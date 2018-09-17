//
//  main.cpp
//  Neural Net
//
//  Created by Edgar Gonzalez on 8/6/18.
//  Copyright © 2018 Edgar Gonzalez. All rights reserved.
//

#include <iostream>
#include "matrix.hpp"
#include "NeuralNet.hpp"
#include "dataParser.h"

int main(int argc, const char * argv[])
{
    NeuralNet nn(784,10,10);
    Matrix<double> dataMatrix = returnMatrixData("/home/edgar/Desktop/neuralnet/data0");
    std::vector<double> zeroes(10,0);
    zeroes[0] = 1;
    Matrix<double> outputs = Matrix<double>::makeMatrixFromVec({zeroes});
    for(int j = 0; j < 10; j++) {
        for (int i = 0; i < 1000; i++) {
            Matrix<double> oneRow = dataMatrix[i];
            nn.feedForward(oneRow);
            nn.learn(oneRow, outputs);
        }
    }

    Matrix<double> testData = dataMatrix[1];
    Matrix<double> prediction = nn.feedForward(testData);
    std::cout << prediction <<std::endl;
    return 0;

}
