#ifndef LINALG_MATRIX_H
#define LINALG_MATRIX_H

#include "misc.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <span>

//////////////////////////////////////////////////////////////////////////////

namespace linalg {

template <std::floating_point ScalarType, size_t Rows, size_t Cols>
struct Matrix {
    static const size_t NumCols = Cols;
    static const size_t NumRows = Rows;
    static const size_t Size    = Cols * Rows;

    using Scalar  = ScalarType;
    using RowType = std::span<Scalar>;
    std::array<Scalar, Cols * Rows> data;

    Matrix()                                        = default;
    Matrix(Matrix<Scalar, Rows, Cols> const& other) = default;
    Matrix(Matrix<Scalar, Rows, Cols>&& other)      = default;
    Matrix<Scalar, Rows, Cols>& operator=(Matrix<Scalar, Rows, Cols> const& other) = default;
    Matrix<Scalar, Rows, Cols>& operator=(Matrix<Scalar, Rows, Cols>&& other) = default;

    Matrix(ScalarType const (&initializer)[1])
    {
        std::fill(data.begin(), data.end(), initializer[0]);
    }

    template <size_t M, size_t N>
    Matrix(ScalarType const (&initializer)[M][N])
    {
        // If we have are getting passed a 1d init list, then make sure our matrix is a vector.
        if constexpr (M == 1)
        {
            static_assert((N == NumCols && NumRows == 1) || (N == NumRows && NumCols == 1));
        }
        else
        {
            static_assert(M == NumRows);
            static_assert(N == NumCols);
        }

        std::copy_n(&initializer[0][0], M * N, data.begin());
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

    auto operator[](size_t index)
    {
        assert(index >= 0);

        if constexpr (Cols == 1)
        {
            assert(index < Rows);
            return data[index];
        }
        else if constexpr (Rows == 1)
        {
            assert(index < Cols);
            return data[index];
        }
        else
        {
            assert(index < Rows);
            return std::span<Scalar, Cols> {&data[index * Cols], Cols};
        }
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

// _idx avoids the use of the [] operator. This is expected to be useful when
// vectors are added as it allows vectors and matrices to use the same code,
// i.e. vectors are matrices where one of the dimensions is 1.
template <typename M>
inline auto _idx(M& mat, size_t r, size_t c) -> float*
{
    return &mat.data[0] + ((r * M::NumCols) + c);
}

template <typename M>
inline auto _idxv(M const& mat, size_t r, size_t c) -> float
{
    return *(&mat.data[0] + ((r * M::NumCols) + c));
}

template <typename MLeft, typename MRight>
auto operator*(MLeft A, MRight B) -> Matrix<decltype(_idxv(A, 0, 0) * _idxv(B, 0, 0)), MLeft::NumRows, MRight::NumCols>
{
    static_assert(MLeft::NumCols == MRight::NumRows,
                  "Invalid matrix dimensions.");

    Matrix<decltype(_idxv(A, 0, 0) * _idxv(B, 0, 0)), MLeft::NumRows, MRight::NumCols> result {{0}};

    size_t i, j, k;

    for (i = 0; i < MLeft::NumRows; ++i)
    {
        for (j = 0; j < MRight::NumCols; ++j)
        {
            for (k = 0; k < MLeft::NumCols; ++k)
            {
                *_idx(result, i, j) += _idxv(A, i, k) * _idxv(B, k, j);
            }
        }
    }

    return result;
}


template <typename MLeft>
auto operator*(MLeft A, typename MLeft::Scalar B) -> Matrix<decltype(_idxv(A, 0, 0) * B), MLeft::NumRows, MLeft::NumCols>
{
    Matrix<decltype(_idxv(A, 0, 0) * B), MLeft::NumRows, MLeft::NumCols>
        result(A);

    for (auto& el : result.data)
    {
        el *= B;
    }

    return result;
}


template <typename MRight>
auto operator*(typename MRight::Scalar B, MRight A) -> Matrix<decltype(_idxv(A, 0, 0) * B), MRight::NumRows, MRight::NumCols>
{
    return A * B;
}


template <typename MLeft>
auto operator*=(MLeft& A, typename MLeft::Scalar B) -> Matrix<decltype(_idxv(A, 0, 0) * B), MLeft::NumRows, MLeft::NumCols>
{
    for (auto& el : A.data)
    {
        el *= B;
    }

    return A;
}


template <typename MLeft, typename MRight>
auto operator+(MLeft A, MRight B) -> Matrix<decltype(_idxv(A, 0, 0) + _idxv(B, 0, 0)), MLeft::NumRows, MRight::NumCols>
{
    static_assert(MLeft::NumCols == MRight::NumCols,
                  "Invalid matrix dimensions.");
    static_assert(MLeft::NumRows == MRight::NumRows,
                  "Invalid matrix dimensions.");

    Matrix<decltype(_idxv(A, 0, 0) + _idxv(B, 0, 0)), MLeft::NumRows, MRight::NumCols> result;

    size_t i, j;

    for (i = 0; i < MLeft::NumRows; ++i)
    {
        for (j = 0; j < MRight::NumCols; ++j)
        {
            *_idx(result, i, j) = _idxv(A, i, j) + _idxv(B, i, j);
        }
    }

    return result;
}


template <typename MLeft>
auto operator+(MLeft A, typename MLeft::Scalar B) -> Matrix<decltype(_idxv(A, 0, 0) + B), MLeft::NumRows, MLeft::NumCols>
{
    Matrix<decltype(_idxv(A, 0, 0) + B), MLeft::NumRows, MLeft::NumCols> result(A);

    for (auto& el : result.data)
    {
        el += B;
    }

    return result;
}


template <typename MRight>
auto operator+(typename MRight::Scalar B, MRight A) -> Matrix<decltype(_idxv(A, 0, 0) + B), MRight::NumRows, MRight::NumCols>
{
    return A + B;
}

//////////////////////////////////////////////////////////////////////////////

template <typename MAT, size_t N>
auto cols(MAT& mat, int const (&values)[N]) -> linalg::Matrix<typename MAT::Scalar, MAT::NumRows, N>
{
    linalg::Matrix<typename MAT::Scalar, MAT::NumRows, N> other;

    int dst = 0;
    for (auto src : values)
    {
        assert(src < MAT::NumCols);

        for (int r : irange<MAT::NumRows>())
        {
            other[r][dst] = mat[r][src];
        }
        ++dst;
    }

    return other;
}

template <size_t Rows, size_t Cols>
using Matrixf = Matrix<float, Rows, Cols>;

template <size_t Rows, size_t Cols>
using Matrixd = Matrix<double, Rows, Cols>;


template <typename Scalar, size_t Rows>
using Vector = Matrix<Scalar, Rows, 1>;

template <size_t Rows>
using Vectorf = Matrix<float, Rows, 1>;

template <size_t Rows>
using Vectord = Matrix<double, Rows, 1>;

} // end namespace linalg

//////////////////////////////////////////////////////////////////////////////

template <typename M>
auto iter(M& mat) -> std::array<typename M::RowType, M::NumRows>
{
    std::array<typename M::RowType, M::NumRows> v;
    for (auto i : irange<M::NumRows>())
    {
        v[i] = mat[i];
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

//////////////////////////////////////////////////////////////////////////////

#endif // LINALG_MATRIX_H