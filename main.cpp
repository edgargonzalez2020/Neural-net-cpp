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

int main(int argc, const char * argv[]) {
   NeuralNet nn(2,4,1);
   nn.setLearningRate(.65);
   std::vector< std::vector<double> > xorInputSet;
   xorInputSet.push_back({0,0});
   xorInputSet.push_back({1,0});
   xorInputSet.push_back({1,1});
   xorInputSet.push_back({0,1});
   std::vector< std::vector<double> > xorTargetSet;
   xorTargetSet.push_back({0});
   xorTargetSet.push_back({1});
   xorTargetSet.push_back({0});
   xorTargetSet.push_back({1});

   Matrix<double> xorMatrixInput;
   Matrix<double> xorMatrixTarget;

   try
   {
        xorMatrixInput = Matrix<double>::makeMatrixFromVec(xorInputSet);
        xorMatrixTarget = Matrix<double>::makeMatrixFromVec(xorTargetSet);
   }
   catch(std::invalid_argument& a)
   {
        std::cerr << a.what() << std::endl;
   }
   for(int j = 0; j < 10000; j++)
   {
        for(int i = 0; i < 4; i++)
        {
            std::cout << "Training..." << std::endl;
            Matrix<double> inputT = xorMatrixInput[i];
            Matrix<double> targetT = xorMatrixTarget[i];
            nn.feedForward(inputT);
            nn.learn(inputT,targetT);
        }
    }
    Matrix<double> testData = xorMatrixInput[1];
    Matrix<double> prediction = nn.feedForward(testData);
    prediction.print();
   //  Matrix<double> output = Matrix<double>::columnVector(test);
   //  Matrix<double> testOutput = nn.predict(output);
   //  testOutput.print();

    return 0;
}
