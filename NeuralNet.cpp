//
//  NeuralNet.cpp
//  Neural Net
//
//  Created by Edgar Gonzalez on 8/9/18.
//  Copyright © 2018 Edgar Gonzalez. All rights reserved.
//

#include "NeuralNet.hpp"
 NeuralNet::NeuralNet(int inputNodesA, int hiddenNodesA, int outputNodesA)
{
	
    this->input_nodes = inputNodesA;
    this->hidden_nodes = hiddenNodesA;
    this->output_nodes = outputNodesA;	
   
    this->biasHidden = Matrix<double>(1, hiddenNodesA);
    this->biasHidden.randomize();
    this->biasOutput= Matrix<double>(1, outputNodesA);
    this->biasOutput.randomize();
    this->weights_input_hidden = Matrix<double>(inputNodesA, hiddenNodesA);
    this->weights_input_hidden.randomize();
    this->weights_hidden_output= Matrix<double>(hiddenNodesA,outputNodesA);
    this->weights_hidden_output.randomize();
    this->learningRate = 0.25;
}
void NeuralNet::setLearningRate(int newRate)
{
    this->learningRate = newRate;
}


/*!
 * @details Convinience function that returns sigmoid
 * @return std::function<double (double)>, a function that accepts a double and
 * returns a double
 */
std::function<double (double)> NeuralNet::returnSigmoidFunction()
{
    std::function<double (double)> sigmoidFnc;
    sigmoidFnc = [](double x) { return 1 / (1 + exp(-x)); };
    return sigmoidFnc;
}
/*!
 * @details Convinience function that returns the derivitive of the sigmoid function
 * @return std::function<double (double)>, a function that accepts a double and
 * returns a double
 */
std::function<double (double)> NeuralNet::returnDsigmoidFunction()
{
    std::function<double (double)> dSigmoidFnc;
    dSigmoidFnc = [](double x) ->double{return exp(-x)/(pow(1+exp(-x),2));} ;
    return dSigmoidFnc;
}
/*!
 * @details Forward propagation for the Neural Net, sets all the values of the Neural Net.
 * @return Matrix<double>, the output of the network.
 */
Matrix<double> NeuralNet::feedForward(const Matrix<double>& inputs)
{
    std::function<double (double)> sigmoidFunction = returnSigmoidFunction();
    this->H = Matrix<double>::dot(inputs, this->weights_input_hidden);
    H.elementWiseAddMatrix(this->biasHidden);
    H.map(sigmoidFunction);

    this->Y = Matrix<double>::dot(H, this->weights_hidden_output);
    Y.elementWiseAddMatrix(this->biasOutput);
    Y.map(sigmoidFunction);
	return Y;
}
/*!
 * @details This function is how the network learns, using backpropagation and stochastic gradient desecent. This algorithm in particular uses the squared mean loss.
 *
 */
void NeuralNet::learn(Matrix<double>& input,Matrix<double>& outputs)
{
    std::function<double (double)> sigmoidFunction = returnDsigmoidFunction();

    //computes the derivitive of the loss function with respect to the bias, output layer
    Matrix<double> DJdb2 = Matrix<double>::subtract(this->Y, outputs);
    Matrix<double> temp = Matrix<double>::dot(this->H, this->weights_hidden_output);
    temp.elementWiseAddMatrix(this->biasOutput);
    temp.map(sigmoidFunction);
    DJdb2.elementWiseMultiplyMatrix(temp);


    //computes the derivitive of the loss function with respect to the bias, input layer
    Matrix<double> weightT = Matrix<double>::transpose(this->weights_hidden_output);
    Matrix<double> temp2 = Matrix<double>::dot(input,this->weights_input_hidden);
    temp2.elementWiseAddMatrix(this->biasHidden);
    temp2.map(sigmoidFunction);
    Matrix<double> DJdb1 = Matrix<double>::dot(DJdb2,weightT);
    DJdb1.elementWiseMultiplyMatrix(temp2);

    //computes derivitive of the loss function with respect to the weights of the output layer
    Matrix<double> DJdw2 = Matrix<double>::transpose(this->H);
    DJdw2 = Matrix<double>::dot(DJdw2,DJdb2);


    //computes derivitive of the loss function with respect to the weights of the input layer
    Matrix<double> inputT = Matrix<double>::transpose(input);
    Matrix<double> DJdw1 = Matrix<double>::dot(inputT,DJdb1);
    
    //scale the derivitives by the learning rate
    DJdw1.elementWiseMulitpyScalar(this->learningRate);
    DJdw2.elementWiseMulitpyScalar(this->learningRate);
    DJdb1.elementWiseMulitpyScalar(this->learningRate);
    DJdb2.elementWiseMulitpyScalar(this->learningRate);
    
    //adjust the weights
    this->weights_input_hidden = Matrix<double>::subtract(this->weights_input_hidden, DJdw1);
    this->weights_hidden_output = Matrix<double>::subtract(this->weights_hidden_output, DJdw2);
    this->biasHidden = Matrix<double>::subtract(this->biasHidden, DJdb1);
    this->biasOutput = Matrix<double>::subtract(this->biasOutput, DJdb2);
}


void NeuralNet::saveModel()
{
    //TODO: Account for little endian and big endian machines 
    std::ofstream stream;
    stream.open("bias_hidden.nn", std::ios::binary);
    // save values for biasHidden
    int rows = biasHidden.getRows();
    int cols = biasHidden.getColumns();
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            double val = biasHidden(i,j);
        }
    }
}
void NeuralNet::loadModel(std::string fileName)
{
    //TODO: Load file from local and set values 
}
