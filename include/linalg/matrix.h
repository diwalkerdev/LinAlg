#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include <array>
#include <iostream>
#include <memory>
#include <span>
#include <stddef.h>

namespace linalg {

template <typename ScalarType, size_t Rows, size_t Cols>
struct Matrix {
    static const size_t NumCols = Cols;
    static const size_t NumRows = Rows;
    static const size_t Size    = Cols * Rows;

    using Scalar  = ScalarType;
    using RowType = std::span<Scalar>;
    using array2d = std::array<Scalar, Cols * Rows>;
    array2d data;
    // std::shared_ptr<array2d> data = std::make_shared<array2d>();

    Matrix()                                        = default;
    Matrix(Matrix<Scalar, Rows, Cols> const& other) = default;
    Matrix(Matrix<Scalar, Rows, Cols>&& other)      = default;

    Matrix(array2d initializer)
        : data(initializer)
    {
    }

    // static
    // Matrix<Scalar, Rows, Cols> zero()
    // {
    //     Matrix<Scalar, Rows, Cols> m;
    //     ::zero(m);
    //     return m;
    // }

    // static
    // Matrix<Scalar, Rows, Cols> I()
    // {
    //     static_assert(Rows == Cols, "Matrix is not square");

    //     Matrix<Scalar, Rows, Cols> m;
    //     ::zero(m);
    //     for (int i=0; i < Rows; ++i) {
    //         m[i][i] = 1;
    //     }

    //     return m;
    // }

    std::span<Scalar> operator[](size_t index)
    {
        return {&data[index * Cols], Cols};
    }

    // Matrix<Scalar, Rows, Cols>& operator=(Matrix<Scalar, Rows, Cols> other)
    // {
    //     std::cout << "matrix matrix assign" << std::endl;
    //     this->data = std::make_shared<array2d>();

    //     for (size_t i = 0; i < Rows; ++i)
    //     {
    //         for (size_t j = 0; j < Cols; ++j)
    //         {
    //             (*data)[i][j] = (*other.data)[i][j];
    //         }
    //     }
    //     return *this;
    // }

    // Matrix<Scalar, Rows, Cols>& operator=(const std::array<Scalar, Rows*Cols>
    // &other)
    // {
    //     std::cout << "matrix array assign" << std::endl;
    //     this->data = std::make_shared<array2d>();

    //     for (size_t i = 0; i < Rows; ++i)
    //     {
    //         for (size_t j = 0; j < Cols; ++j)
    //         {
    //             (*data)[i][j] = other[i*Cols + j];
    //         }
    //     }
    //     return (*this);
    // }
};

//////////////////////////////////////////////////////////////////////////////

template <typename MLeft, typename MRight>
auto operator*(MLeft A, MRight B) -> Matrix<decltype(A[0][0] * B[0][0]), MLeft::NumRows, MRight::NumCols>
{
    static_assert(MLeft::NumCols == MRight::NumRows,
                  "Invalid matrix dimensions.");

    Matrix<decltype(A[0][0] * B[0][0]), MLeft::NumRows, MRight::NumCols> result {{0}};

    size_t i, j, k;

    for (i = 0; i < MLeft::NumRows; ++i)
    {
        for (j = 0; j < MRight::NumCols; ++j)
        {
            for (k = 0; k < MLeft::NumCols; ++k)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


template <typename MLeft, typename Scalar>
auto operator*(MLeft A, Scalar B) -> Matrix<decltype(A[0][0] * B), MLeft::NumRows, MLeft::NumCols>
{
    Matrix<decltype(A[0][0] * B), MLeft::NumRows, MLeft::NumCols> result(A);

    for (auto& el : result.data)
    {
        el *= B;
    }

    return result;
}


template <typename MLeft, typename Scalar>
auto operator*=(MLeft A, Scalar B) -> Matrix<decltype(A[0][0] * B), MLeft::NumRows, MLeft::NumCols>
{
    for (auto& el : A.data)
    {
        el *= B;
    }

    return A;
}

//////////////////////////////////////////////////////////////////////////////
template <size_t Rows, size_t Cols>
using Matrixi = Matrix<int, Rows, Cols>;

template <size_t Rows, size_t Cols>
using Matrixf = Matrix<float, Rows, Cols>;

template <size_t Rows, size_t Cols>
using Matrixd = Matrix<double, Rows, Cols>;


} // end namespace linalg

//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <vector>


template <typename M>
auto row_idxs(M& mat) -> std::vector<int>
{
    std::vector<int> v(M::NumRows);

    int i = -1;
    std::generate(v.begin(), v.end(), [&i]() -> int { return ++i; });

    return v;
}


template <typename M>
auto iter(M& mat) -> std::vector<typename M::RowType>
{
    std::vector<typename M::RowType> v;
    for (auto i : row_idxs(mat))
    {
        v.push_back(mat[i]);
    }
    return v;
}


//////////////////////////////////////////////////////////////////////////////

template <typename Scalar, size_t Rows, size_t Cols>
std::ostream& operator<<(std::ostream&                       os,
                         linalg::Matrix<Scalar, Rows, Cols>& A)
{
    for (auto const& row : iter(A))
    {
        for (auto el : row)
        {
            os << el << " ";
        }
        os << std::endl;
    }

    return os;
}


#endif // LINALG_MATRIX_H