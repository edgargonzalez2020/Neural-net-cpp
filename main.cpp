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
    Matrix<double> dataMatrix = returnMatrixData("/Users/Eddie_g/Library/Autosave Information/Neural_Net/Neural_Net/data0");
    int answer;
    std::cout << "Handwritten digit classifier!" << std::endl;
    std::cout << "Would you like to train a new model, or load up a previously trained model?" << std::endl;
    std::cout << "1. Train " << std::endl << "2. Load model " << std::endl;
    std::cin >> answer;
    if(answer == 1)
    {
        std::cout << "Training..." << std::endl;
//        for(int i = 0; i < 10; i++)
//        {
            std::vector<double> zeroes(10,0);
            zeroes[0] = 1;
            Matrix<double> outputs = Matrix<double>::makeMatrixFromVec({zeroes});
            for(int j = 0; j < 10; j++)
            {
                for (int k = 0; k < 700; k++)
                {
                    Matrix<double> oneRow = dataMatrix[k];
                    nn.feedForward(oneRow);
                    nn.learn(oneRow, outputs);
                }
            //}
        }
        std::string doPrediction;
        std::cout << "Training complete." << std::endl << "Would you like to make a prediction(y/n)?";
        std::cin >> doPrediction;
        if(doPrediction == "y" || doPrediction == "Y")
        {
            int number;
            std::cout << "What number would you like to classify?(0-9)" << std::endl;
            std::cin >> number;
            std::cout << std::endl << "The program will choose a random example from 300 different images." << std::endl;
            Matrix<double> testData = dataMatrix[1];
            Matrix<double> prediction = nn.feedForward(testData);
            std::cout << prediction <<std::endl;
            
        }
    }
    else if(answer == 2)
    {
        // code for the load model option
        // TODO: implement load model in Neural Net class 
    }
    return 0;

}
