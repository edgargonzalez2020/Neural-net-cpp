//
//  matrix.hpp
//  Neural Net
//
//  Created by Edgar Gonzalez on 8/6/18.
//  Copyright © 2018 Edgar Gonzalez. All rights reserved.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <random>


template <class T> class Matrix
{
public:
    int validateRows(int userRows = 0);
    int validateCols(int userCols = 0);
    Matrix();
    Matrix(int userRows,int userCols);
    Matrix(const Matrix<T>& a);
    static Matrix<T> multiply(const Matrix<T>& a, const Matrix<T>& b);
    static Matrix<T> subtract(const Matrix<T>& a, const Matrix<T>& b);
    static Matrix<T> transpose(const Matrix<T>& a);
    static Matrix<T> map(const Matrix<T>& a,std::function<T (T)>& func);
    static Matrix<T> columnVector(const std::vector<T>& a);
    static Matrix<T> makeMatrixFromVec(const std::vector<std::vector<T> >& refVec);
    static Matrix<T> horizontalConcat( Matrix<T>& a,  Matrix<T>& b);

    void map(std::function<T (T)>& func);
    void redefineInternalMatrix(const std::vector<std::vector<T> >& a);
    int getRows(){return rows;}
    int getColumns(){return columns;}
    void setRows(int row) {this->rows = row;}
    void setColumns(int col) {this->columns = col;}
    void elementWiseMultiplyMatrix(const Matrix<T>& a);
    void elementWiseMulitpyScalar(T n);
    std::vector<T> toVec();
    void elementWiseAddMatrix(const Matrix<T>& a);
    void elementWiseAddScalar(T n);
    void randomize();
    void equals(int row, int column, T newVal);
    void print();
    T operator()(int row, int col)
    const
    {
        return this->internalMatrix[row][col];
    }
    Matrix<T> operator[](int i)
    {
      if(i < 0 || i > this->rows) throw std::out_of_range("Matric access out of bounds");
      Matrix<T> row = Matrix<T>::makeMatrixFromVec({this->internalMatrix[i]});
      return row;
    }

private:
    int rows;
    int columns;
    std::vector<std::vector<T> > internalMatrix;
};
/*
 * These two functions are to verify that the caller enters a valid dimension for a matrix
 *
 */
template <typename T>
int Matrix<T>::validateRows(int userRows)
{
    return userRows <= 0 ? 5: userRows; // arbitrarily return 5 if caller invokes a negative number
}
template <typename T>
int Matrix<T>::validateCols(int userCols)
{
    return userCols <= 0 ? 5: userCols; // arbitrarily return 5 if caller invokes a negative number
}
/*
 *  No argument constructor
 */
template <typename T>
Matrix<T>::Matrix()
{
    this->rows = validateRows();
    this->columns = validateCols();
    this->internalMatrix.resize(this->rows,std::vector<T>(this->columns,0)); //initialize all values to 0
}
/*
 *  Two argument constructor
 */
template <typename T>
Matrix<T>::Matrix(int userRows, int userCols)
{
    this->rows = validateRows(userRows);
    this->columns = validateCols(userCols);
    this->internalMatrix.resize(this->rows,std::vector<T>(this->columns,0)); //initialize all values to 0
}
/*
 *  Copy constructor
 */
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& a) = default;
/*
 *  Method will return a matrix initialized to zeros if the product cannot be computed.
 *  This leaves the responsibility to the caller to make sure the
 *  result was meaningful, aka verify dims before invoking. Static versison
 */
template <typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& a, const Matrix<T>& b)
{
    try
    {
        if(a.columns != b.rows)
        {
            throw a;
        }
    }
    catch(Matrix<T> a)
    {
        std::cout << "Cannot multiply matrices of different dims." << std::endl;
        Matrix<T> temp;
        return temp;
    }
    Matrix<T> result(a.rows,b.columns);

    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < b.columns; j++)
        {
            for(int k = 0; k < b.rows; k++)
            {
                T current = result(i,j);
                T sum = current + (a(i,k) * b(k,j));
                result.equals(i,j,sum);
            }
        }
    }
    return result;
}
/*
 *  Returns the element wise difference of the Matrices.
 *  Will return Matrix of size a.rows x a.columns, initialized to zero if dims do not match
 *  responisbility of caller to verify rows and cols before invoking
 */
template <typename T>
Matrix<T> Matrix<T>::subtract(const Matrix<T>& a, const Matrix<T>& b)
{
    if(a.rows != b.rows || a.columns != b.columns)
    {
        std::cout << "Matrix dims do not match" << std::endl;
        Matrix<T> err(a.rows,a.columns);
        return err;
    }
    Matrix<T> temp(a.rows,a.columns);
    for(int i = 0; i < temp.getRows(); i++)
    {
        for(int j = 0; j < temp.getColumns(); j++)
        {
            temp.equals(i, j, a(i,j) - b(i,j));
        }
    }
    return temp;
}
/*
 *  Returns the transpose of a matrix, as a Matrix object.
 */
template <typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T> &a)
{
    Matrix<T> trasnpose(a.columns,a.rows);
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.columns; j++)
        {
            T val = a(i,j);
            trasnpose.equals(j, i, val);
        }
    }
    return trasnpose;

}
/*
 *  Returns a Matrix objects with the mapped values applied
 */
template <typename T>
Matrix<T> Matrix<T>::map(const Matrix<T>& a, std::function<T (T)>& func)
{
    Matrix<T> temp(a.rows,a.columns);
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.columns; j++)
        {
            T newVal = func(a(i,j));
            temp.equals(i, j, newVal);
        }
    }
    return temp;
}
/*
 *  Maps a function to each element in the matrix
 */
template <typename T>
void Matrix<T>::map(std::function<T (T)>& func)
{
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] = func(this->internalMatrix[i][j]);
        }
    }
}
/*
 * Method that redefines internal matrix given a vector
 */
template <typename T>
void Matrix<T>::redefineInternalMatrix(const std::vector<std::vector<T> >& a)
{
    // error check dimensions
    int row = static_cast<int>(a.size());
    int col = static_cast<int>(a[0].size());
    this->setRows(row);
    this->setColumns(col);
    this->internalMatrix = a;
}
/*
 * Multiplies the matrices element wise, must be a Matrix object.
 */
template <typename T>
void Matrix<T>::elementWiseMultiplyMatrix(const Matrix<T>& a)
{
    if(this->rows != a.rows || this->columns != a.columns)
    {
        std::cout << "Cannot multiply matrices of different dims" << std::endl;
        return;
    }
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] *= a(i,j);
        }
    }

}
/*
 * Mutliplies each element by a scalar value.
 */
template <typename T>
void Matrix<T>::elementWiseMulitpyScalar(T n)
{
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] *= n;
        }
    }

}
/*
 *  Transforms Matrix object into a 1d array
 *
 */
template <typename T>
std::vector<T> Matrix<T>::toVec()
{
    std::vector<T> temp;
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            temp.push_back(this->internalMatrix[i][j]);
        }
    }
    return temp;
}
/*
 * Adds the matrices element wise, must be a Matrix object.
 */
template <typename T>
void Matrix<T>::elementWiseAddMatrix(const Matrix<T>& a)
{
    if(this->rows != a.rows || this->columns != a.columns)
    {
        std::cout << "Cannot add matrices of different dims" << std::endl;
        return;
    }
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] += a(i,j);
        }
    }

}
/*
 * Adds each element by a scalar value.
 */
template <typename T>
void Matrix<T>::elementWiseAddScalar(T n)
{
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] += n;
        }
    }

}
/*
 * Returns a matrix with elements to represent the weights of the Neural Network. Therefore
 * every value is 0 < x < 1, for some value x.
 */
template <typename T>
void Matrix<T>::randomize()
{
	std::random_device device;
	std::mt19937 mt(device());
	for(int i = 0; i < this->rows; i++)
	{
	  for(int j = 0; j < this->columns; j++)
	  {
	  	T randomNum;
	  	std::uniform_real_distribution<T> dist(0,1);
		randomNum = dist(mt);
	    this->internalMatrix[i][j] = randomNum;
	  }
	}
}
/*
 *  Function to allow matrix values to be manipulated.
 */
template <typename T>
void Matrix<T>::equals(int row,int column, T newValue)
{
    this->internalMatrix[row][column] = newValue;
}
/*
 *  Helper method to view elements in the matrix
 */
template <typename T>
void Matrix<T>::print()
{
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            std::cout << this->internalMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
/*
 * Given a one dimensional std::vector it will 'flatten' it to a vector column

 */
template <typename T>
Matrix<T> Matrix<T>::columnVector(const std::vector<T>& a)
{
  Matrix<T> column(static_cast<int>(a.size()),1);
  for(int i = 0; i < a.size(); i++)
  {
    column.equals(i,0,a[i]);
  }
  return column;
}
template <typename T>
Matrix<T> Matrix<T>::makeMatrixFromVec(const std::vector<std::vector<T> >& refVec)
{
    if(!refVec.empty())
    {
        int columnSize = static_cast<int>(refVec[0].size());
        int rowSize = static_cast<int>(refVec.size());
        Matrix<T> temp(rowSize,columnSize);
        for(int i = 0; i < rowSize; i++)
        {
            for(int j = 0; j < columnSize; j++)
            {
                if(refVec[i].size() != columnSize) throw std::invalid_argument("All rows must have equal length");
                temp.equals(i,j,refVec[i][j]);
            }

        }
        return temp;
    }
}
/*
 *	Given two matrix objects this function return the concatanation of botr matrices 
 */
template <typename T>
Matrix<T> Matrix<T>::horizontalConcat( Matrix<T>& a,  Matrix& b)
{
	if(a.getRows() != b.getRows())
	{
		throw std::range_error("Matrix rows must match for concatenation");
	}
	int newColSize = a.getColumns() + b.getColumns();
	int loopCount = a.getColumns() + b.getColumns(); //loop through depending on the bigger size
	int i = 0;
	int j = 0;
	int k = 0;
	Matrix<T> temp(a.getRows(),newColSize);

	if(a.getColumns() < b.getColumns())
	{
		while(j < a.getRows())
		{
			while(i < loopCount)
			{
				if(i < a.getColumns())
				{
					temp.equals(j,i, a(j,i));
				}
				else
				{
					temp.equals(j,i, b(j,k));
					k++;
				}
				i++;
			}
			i = 0;
			k = 0;
			j++;
		}
	}
	else if(a.getColumns() >= b.getColumns())
	{
		while(j < a.getRows())
		{
			while(i < loopCount)
			{
				if(i < a.getColumns())
				{
					temp.equals(j,i, a(j,i));
				}
				else
				{
					temp.equals(j,i, b(j,k));
					k++;
				}
				i++;
			}
			i = 0;
			k = 0;
			j++;
		}

	}
	return temp;
}

#endif /* matrix_hpp */
