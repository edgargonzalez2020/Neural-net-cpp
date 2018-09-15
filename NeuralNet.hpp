//
//  NeuralNet.hpp
//  Neural Net
//
//  Created by Edgar Gonzalez on 8/9/18.
//  Copyright © 2018 Edgar Gonzalez. All rights reserved.
//

#ifndef NeuralNet_hpp
#define NeuralNet_hpp

#include <stdio.h>
#include <cmath>
#include "matrix.hpp"

class NeuralNet
{
public:
    NeuralNet(int inputNodes, int hiddenNodes, int outputNodes);
    void train(Matrix<double>& input,
               Matrix<double>& targets);
    Matrix<double> predict(Matrix<double>& input);
    void setLearningRate(int newRate);
    double getLearningRate(){return learningRate;}
    Matrix<double> feedForward(Matrix<double>& input);
    static std::function<double (double)> returnSigmoidFunction();
    static std::function<double (double)> returnDsigmoidFunction();
    void learn(Matrix<double>& a, Matrix<double>& b);
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    double learningRate;
    Matrix<double> biasHidden;
    Matrix<double> biasOutput;
    Matrix<double> weights_input_hidden;
    Matrix<double> weights_hidden_output;
    Matrix<double> Y;
    Matrix<double> H;
};

#endif /* NeuralNet_hpp */
