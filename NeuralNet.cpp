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

/*
 *  Returns a function to be used to map elements of Matrix object
 *  std::function takes in a single paramater double and returns a double
 */
std::function<double (double)> NeuralNet::returnSigmoidFunction()
{
    std::function<double (double)> sigmoidFnc;
    sigmoidFnc = [](double x) { return 1 / (1 + exp(-x)); };
    return sigmoidFnc;
}
/*
 *  Returns a function to be used to map elements of a Matrix object.
 *  std::function takes in a single paramater double and returns a double
 */
std::function<double (double)> NeuralNet::returnDsigmoidFunction()
{
    std::function<double (double)> dSigmoidFnc;
    dSigmoidFnc = [](double x) ->double{return exp(-x)/(pow(1+exp(-x),2));} ;
    return dSigmoidFnc;
}
// refactor training method by dividing work into seperate methods
// void NeuralNet::feedForward(const Matrix& input,const Matrix& targets)
// {
//     Matrix<double> z1 = Matrix<double>::multiply(this->weights_input_hidden, input);
//     std::function(double (double)) sigmoid = returnSigmoidFunction();
//     z1.elementWiseAddMatrix(this->biasHidden);
//     z1.map(sigmoid);
//     Matrix<double> z2 = Matrix<double>::multiply(this->weights_hidden_output, outputs);
//     Matrix.elementWiseAddMatrix(this->biasOutput);
//     z2.map(sigmoid);
// }
// void Matrix<double>::backPropagation()
// {
    
// }
/*
 *  Trains the Neural Network by adjusting the weights matrix and biases
 */
/*
void NeuralNet::train(Matrix<double> &input,Matrix<double> &targets)
{
    printf("Input(%d,%d)\n",input.getRows(),input.getColumns());
    printf("Targets(%d,%d)\n",targets.getRows(),targets.getColumns() );

    std::function<double (double)> sigmoidFnc = this->returnSigmoidFunction();
    std::function<double (double)> dSigmoidFnc = this->returnDsigmoidFunction();
    Matrix<double> hiddenMatrix = Matrix<double>::multiply(this->weights_input_hidden,input);
    hiddenMatrix.elementWiseAddMatrix(this->biasHidden);
    hiddenMatrix.map(sigmoidFnc);
    Matrix<double> outputs = Matrix<double>::multiply(this->weights_hidden_output, hiddenMatrix);
    outputs.elementWiseAddMatrix(this->biasOutput);
    outputs.map(sigmoidFnc);
    Matrix<double> outputErrors = Matrix<double>::subtract(targets, outputs);
    Matrix<double> gradients = Matrix<double>::map(outputs, dSigmoidFnc);
    gradients.elementWiseMultiplyMatrix(outputErrors);
    gradients.elementWiseMulitpyScalar(this->learningRate);
    Matrix<double> hiddenTranspose = Matrix<double>::transpose(hiddenMatrix);

    Matrix<double> hiddenOutputWeightDeltas = Matrix<double>::multiply(gradients, hiddenTranspose);
    //adjust weights
    this->weights_hidden_output.elementWiseAddMatrix(hiddenOutputWeightDeltas);
    this->biasOutput.elementWiseAddMatrix(gradients);



    Matrix<double> weights_hidden_outputT = Matrix<double>::transpose(this->weights_hidden_output);
    Matrix<double> hiddenError = Matrix<double>::multiply(weights_hidden_outputT, outputErrors);

    Matrix<double> hiddenGradient = Matrix<double>::map(hiddenMatrix, dSigmoidFnc);
    hiddenGradient.elementWiseMultiplyMatrix(hiddenError);
    hiddenGradient.elementWiseMulitpyScalar(this->learningRate);

    Matrix<double> inputT = Matrix<double>::transpose(input);
    Matrix<double> input_hidden_deltas = Matrix<double>::multiply(hiddenGradient, inputT);
    this->weights_input_hidden.elementWiseAddMatrix(input_hidden_deltas);
    this->biasHidden.elementWiseAddMatrix(hiddenGradient);
}

*/
// Matrix<double> NeuralNet::predict(Matrix<double> &inputs)
// {
//     std::cout <<"--------------------------" << std::endl;
//     printf("Input(%d,%d)\n",inputs.getRows(),inputs.getColumns());
//     Matrix<double> hidden = Matrix<double>::multiply(this->weights_input_hidden, inputs);
//     hidden.elementWiseAddMatrix(this->biasHidden);
//     std::function<double (double)> sigmoidFnc = this->returnSigmoidFunction();
//     hidden.map(sigmoidFnc);
//     Matrix<double> outputs = Matrix<double>::multiply(this->weights_hidden_output, hidden);
//     outputs.elementWiseAddMatrix(this->biasOutput);
//     outputs.map(sigmoidFnc);
//     return outputs;
// }

Matrix<double> NeuralNet::feedForward(Matrix<double>& inputs)
{
    std::function<double (double)> sigmoidFunction = returnSigmoidFunction();
    this->H = Matrix<double>::multiply(inputs, this->weights_input_hidden);
    H.elementWiseAddMatrix(this->biasHidden);
    H.map(sigmoidFunction);

    this->Y = Matrix<double>::multiply(H, this->weights_hidden_output);
    Y.elementWiseAddMatrix(this->biasOutput);
    Y.map(sigmoidFunction);
	return Y;
}
void NeuralNet::learn(Matrix<double>& input,Matrix<double>& outputs)
{
    std::function<double (double)> sigmoidFunction = returnDsigmoidFunction();


    Matrix<double> DJdb2 = Matrix<double>::subtract(this->Y, outputs);
    Matrix<double> temp = Matrix<double>::multiply(this->H, this->weights_hidden_output);
    temp.elementWiseAddMatrix(this->biasOutput);
    temp.map(sigmoidFunction);
    DJdb2.elementWiseMultiplyMatrix(temp);



    Matrix<double> weightT = Matrix<double>::transpose(this->weights_hidden_output);
    Matrix<double> temp2 = Matrix<double>::multiply(input,this->weights_input_hidden);
    temp2.elementWiseAddMatrix(this->biasHidden);
    temp2.map(sigmoidFunction);
    Matrix<double> DJdb1 = Matrix<double>::multiply(DJdb2,weightT);
    DJdb1.elementWiseMultiplyMatrix(temp2);

    Matrix<double> DJdw2 = Matrix<double>::transpose(this->H);
    DJdw2 = Matrix<double>::multiply(DJdw2,DJdb2);



    Matrix<double> inputT = Matrix<double>::transpose(input);
    Matrix<double> DJdw1 = Matrix<double>::multiply(inputT,DJdb1);
    DJdw1.elementWiseMulitpyScalar(this->learningRate);
    DJdw2.elementWiseMulitpyScalar(this->learningRate);
    DJdb1.elementWiseMulitpyScalar(this->learningRate);
    DJdb2.elementWiseMulitpyScalar(this->learningRate);
    this->weights_input_hidden = Matrix<double>::subtract(this->weights_input_hidden, DJdw1);
    this->weights_hidden_output = Matrix<double>::subtract(this->weights_hidden_output, DJdw2);
    this->biasHidden = Matrix<double>::subtract(this->biasHidden, DJdb1);
    this->biasOutput = Matrix<double>::subtract(this->biasOutput, DJdb2);
}
