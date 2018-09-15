//
//  dataParser.h
//  Neural Net
//
//  Created by Edgar Gonzalez on 8/23/18.
//  Copyright © 2018 Edgar Gonzalez. All rights reserved.
//
//
#ifndef dataParser_h
#define dataParser_h
#include "matrix.hpp"
#include <iostream>
#include <fstream>
#include <vector>

double normalizePixelData(unsigned char i)
{
    return static_cast<double>(i) / 255;
}
/*
 *  Read in data stored as unsigned char (1 Byte), each files consists of 28x28 digits back to back
 */
std::vector<unsigned char> readData(std::string fileName)
{
    std::ifstream file(fileName,std::ios::binary);
    file.unsetf(std::ios::skipws);

    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);


    std::vector<unsigned char> vec;
    vec.reserve(fileSize);
    vec.insert(vec.begin(),
               std::istream_iterator<unsigned char>(file),
               std::istream_iterator<unsigned char>());
    return vec;

}
/*
 *  Each file has 1000 training examples, loop through each and store in 2d array
 */
Matrix<double> returnMatrixData(std::string fileName)
{
   Matrix<double> tempMat(1000,784);
   std::vector<unsigned char> data = readData(fileName);
   //std::vector<std::vector<double> > temp;
   std::vector<double> pixelData;
   pixelData.reserve(784);
   int k = 0;
   int j = 0;
   for(int i = 0; i < static_cast<int>(data.size()); i++)
   {
       if(k == 784)
       {
           //temp.push_back(pixelData);
           pixelData.clear();
           k = 0;
           j++;
       }
       tempMat.equals(j,k, static_cast<double>(normalizePixelData(data[i])) );
       k++;
   }
   return tempMat;

}

#endif /* dataParser_h */
