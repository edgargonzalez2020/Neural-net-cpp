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
    int validateRows(int userRows) const;
    int validateCols(int userCols) const;
    Matrix();
    Matrix(int userRows,int userCols);
    Matrix(const Matrix<T>& a);
    static Matrix<T> dot(const Matrix<T>& a,const Matrix<T>& b);
    static Matrix<T> subtract(const Matrix<T>& a, const Matrix<T>& b);
    static Matrix<T> transpose(const Matrix<T>& a);
    static Matrix<T> map(const Matrix<T>& a,std::function<T (T)>& func);
    static Matrix<T> columnVector(const std::vector<T>& a);
    static Matrix<T> makeMatrixFromVec(const std::vector<std::vector<T> >& refVec);
    static Matrix<T> horizontalConcat( Matrix<T>& a,  Matrix<T>& b);

    void map(std::function<T (T)>& func);
    void redefineInternalMatrix(const std::vector<std::vector<T> >& a);
    int getRows()const{return rows;}
    int getColumns()const{return columns;}
    void setRows(int row){this->rows = row;}
    void setColumns(int col) {this->columns = col;}
    void elementWiseMultiplyMatrix(const Matrix<T>& a);
    void elementWiseMulitpyScalar(T n);
    std::vector<T> toVec();
    void elementWiseAddMatrix(const Matrix<T>& a);
    void elementWiseAddScalar(T n);
    void randomize();
    void set(int row, int column, T newVal);
    void print();
    /*!
     * @details Returns the value of the Matrix at index i,j. Throws std::out_of_range exception if negative or out of bounds.
     * @param row
     * @param col
     * @return Value of type T.
     */
    T operator()(int row, int col)const
    {
        validateRows(row);
        validateCols(col);
        if(row > this->rows || col > this->columns) throw std::out_of_range("Matrix access out of bounds");
        return this->internalMatrix[row][col];
    }
    /*!
     * @details Overloaded operator [], this returns a row vector at index i, else if not possible throws std::out_of_range exception.
     * @param i
     * @return Matrix object of type T.
     */
    Matrix<T> operator[](int i)
    {
      if(i < 0 || i > this->rows) throw std::out_of_range("Matric access out of bounds");
      Matrix<T> row = Matrix<T>::makeMatrixFromVec({this->internalMatrix[i]});
      return row;
    }
     /*!
     * @details Overloaded << operator to neatly print array contents.
     * @tparam T
     * @param stream
     * @param a
     * @return
     */
    friend std::ostream& operator<<(std::ostream& stream, const Matrix<T>& a)
    {
        for(int i = 0; i < a.getRows(); i++)
        {
            for(int j = 0; j < a.getColumns(); j++)
            {
                stream << a(i,j) << " ";
            }
            stream << std::endl;
        }
        return stream;
    }

private:
    int rows; /*!< Matrix rows */

    int columns; /*!< Matrix columns */

    std::vector<std::vector<T> > internalMatrix; /*!< Basis of entire Matrix class, every function revolves around this vector. */
};
/*! \brief Method that checks whether input rows is non-negative
 *
 * @tparam T
 * @param userRows
 * @return Returns std::out_of_range exception if rows are non-negative.
 */
template <typename T>
int Matrix<T>::validateRows(int userRows) const
{
    return userRows < 0 ? throw std::out_of_range("Matrix rows must be non-negative"): userRows;
}
/*! \brief Method that checks whether input columns is non-negative
 *
 * @tparam T
 * @param userCols
 * @return Returns std::out_of_range exception if columns are non-negative.
 */
template <typename T>
int Matrix<T>::validateCols(int userCols) const
{
    return userCols < 0 ?  throw std::out_of_range("Matrix columns must be non-negative"):userCols;
}
/*!
 *  @brief Constructor that accepts no arguments
 *  @details Invokes constructor with two arguments, default Matrix object will be 0x0.
 */
template <typename T>
Matrix<T>::Matrix():Matrix(0,0){}
/*!@brief Constructor that accepts two paramaters
 * @details Constructor makes sure user arguments are correct by invoking validateRows and validateCols.
 * @tparam T
 * @param userRows
 * @param userCols
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
/*!
 * @details Method computes the dot product of two Matrix objects. If the dot product cannot be computed due to invalid dimension
 *  will throw std::invalid_argument.
 * @tparam T
 * @param a Matrix object of type T
 * @param b Matrix object of type T
 * @return Returns a Matrix of type T or throws std::invalid_argument.
 */
template <typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& a,const Matrix<T>& b)
{
    if(a.getColumns() != b.getRows())
    {
        throw std::invalid_argument("Matrix dims cannot be multiplied");
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
                result.set(i,j,sum);
            }
        }
    }
    return result;
}
/*!
 * @details Subtracts each individual element in a from b. Only possible if a and b have same dimensions, in that case method will
 * throw std::invalid_argument
 * @tparam T
 * @param a Matrix object of type T
 * @param b Matrix object of type T
 * @return Returns Matrix object of type T or throws std::invalid_argument.
 */
template <typename T>
Matrix<T> Matrix<T>::subtract(const Matrix<T>& a, const Matrix<T>& b)
{
    if(a.rows != b.rows || a.columns != b.columns)
    {
        throw std::invalid_argument("Matrix dims cannot be subtracted");
    }
    Matrix<T> temp(a.rows,a.columns);
    for(int i = 0; i < temp.getRows(); i++)
    {
        for(int j = 0; j < temp.getColumns(); j++)
        {
            temp.set(i, j, a(i,j) - b(i,j));
        }
    }
    return temp;
}
/*!
 * @details Given a Matrix object, method will return the transpose. The return Matrxix will have the columns and rows flipped from the input.
 * @tparam T
 * @param a
 * @return Matrix object of type T.
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
            trasnpose.set(j, i, val);
        }
    }
    return trasnpose;

}
/*!
 * @details Method utilizes std::function, it will apply a function to each element according to the function that is passed in.
 * returns a Matrix object with the new values.
 * @tparam T
 * @param a
 * @param func
 * @return Returns Matrix object of type T.
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
            temp.set(i, j, newVal);
        }
    }
    return temp;
}
/*!
 * @details Method utilizes std::function, it will apply a function to each element according to the function that is passed in.
 * This method, instead modifies the Matrix internally rather than returning a new object.
 * @tparam T
 * @param func
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
    auto row = static_cast<int>(a.size());
    auto col = static_cast<int>(a[0].size());
    this->setRows(row);
    this->setColumns(col);
    this->internalMatrix = a;
}
/*!
 * @details Multiplies each individual element from this object by each element in a. Modifies this object, operation will only
 * work on matrices with same dimensions, otherwise will throw std::invalid_argument.
 * @tparam T
 * @param a
 */
template <typename T>
void Matrix<T>::elementWiseMultiplyMatrix(const Matrix<T>& a)
{
    if(this->rows != a.rows || this->columns != a.columns)
    {
        throw std::invalid_argument("Matrix dims cannot be multiplied");
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
/*!
 *
 * @details Multiplies each element in this object by a scalar value.
 * @tparam T, scalar to be multiplied.
 * @param n
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

/*!
 * @details Transforms Matrix object into a 1 dimensional std::vector.
 * @tparam T
 * @return Returns std::vector of type T.
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
/*!
 * @details Adds each element from this object from a. Addition will only work if dimension are the same, otherwise will
 * throw std::invalid_argument.
 * @tparam T
 * @param a, Matrix to be added
 */
template <typename T>
void Matrix<T>::elementWiseAddMatrix(const Matrix<T>& a)
{
    if(this->rows != a.rows || this->columns != a.columns)
    {
        throw std::invalid_argument("Matrix dims cannot be added");
    }
    for(int i = 0; i < this->rows; i++)
    {
        for(int j = 0; j < this->columns; j++)
        {
            this->internalMatrix[i][j] += a(i,j);
        }
    }

}
/*!
 * @details Adds each element in this object by a scalar value.
 * @tparam T
 * @param n, scalar value to be added
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
/*!
 * @details Utility function to help setup a random Matrix, modifies the object internally. If type T is a floating point it will
 * default to values between 0 and 1, otherwise 0 and 10.
 * @tparam T
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
	  	std::uniform_real_distribution<T> dist(0, 1);
		randomNum = dist(mt);
	    this->internalMatrix[i][j] = randomNum;
	  }
	}
}
/*!
 * @details Allows user to set values at specified indices. Method validated rows and columns, if a negative value
 * is found throws std::out_of_range.
 * @tparam T
 * @param row
 * @param column
 * @param newValue
 */
template <typename T>
void Matrix<T>::set(int row,int column, T newValue)
{
    validateRows(row);
    validateCols(column);
    if(row > this->rows || column > this->columns) throw std::out_of_range("Matrix access out of bounds");
    this->internalMatrix[row][column] = newValue;
}
/*!
 * @details Given a 1 dimension std::vector, the method will 'flatten' the array and return a Matrix object column vector.
 * @tparam T
 * @param a
 * @return Matrix object column vector.
 */
template <typename T>
Matrix<T> Matrix<T>::columnVector(const std::vector<T>& a)
{
  Matrix<T> column(static_cast<int>(a.size()),1);
  for(int i = 0; i < a.size(); i++)
  {
    column.set(i,0,a[i]);
  }
  return column;
}
/*!
 * @details Given a 2 dimensional std::vector, method will return a Matrix object.
 * @tparam T
 * @param refVec
 * @return Returns Matrix object of type T.
 */
template <typename T>
Matrix<T> Matrix<T>::makeMatrixFromVec(const std::vector<std::vector<T> >& refVec)
{
    if(!refVec.empty())
    {
        auto columnSize = static_cast<int>(refVec[0].size());
        auto rowSize = static_cast<int>(refVec.size());
        Matrix<T> temp(rowSize,columnSize);
        for(int i = 0; i < rowSize; i++)
        {
            for(int j = 0; j < columnSize; j++)
            {
                if(refVec[i].size() != columnSize) throw std::invalid_argument("All rows must have equal length");
                temp.set(i,j,refVec[i][j]);
            }

        }
        return temp;
    }
}
/*
 *	Given two matrix objects this function return the concatanation of botr matrices 
 */
/*!
 * @details Method will return the horizontal concatenation of two Matrix objects, if Matrix rows do not match,
 * will throw std::invalid_argument
 * @tparam T
 * @param a
 * @param b
 * @return Returns Matrix object of type T.
 */
template <typename T>
Matrix<T> Matrix<T>::horizontalConcat( Matrix<T>& a,  Matrix& b)
{
	if(a.getRows() != b.getRows())
	{
		throw std::range_error("Matrix dims cannot be concatenated");
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
					temp.set(j,i, a(j,i));
				}
				else
				{
					temp.set(j,i, b(j,k));
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
					temp.set(j,i, a(j,i));
				}
				else
				{
					temp.set(j,i, b(j,k));
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
