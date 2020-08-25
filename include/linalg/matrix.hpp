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

template <std::floating_point Tp, std::size_t Rows, std::size_t Cols>
struct Matrix {
    static constexpr auto size() noexcept { return Rows * Cols; }
    static constexpr auto rows() noexcept { return Rows; }
    static constexpr auto cols() noexcept { return Cols; }

    using value_type      = Tp;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = value_type*;
    using const_iterator  = value_type const*;
    using size_type       = std::size_t;

    value_type _elems[Rows * Cols];

    inline value_type*       data() noexcept { return _elems; }
    inline value_type const* data() const noexcept { return _elems; }

    // iterators:
    inline constexpr iterator       begin() noexcept { return iterator(data()); }
    inline constexpr const_iterator begin() const noexcept { return const_iterator(data()); }
    inline constexpr iterator       end() noexcept { return iterator(data() + size()); }
    inline constexpr const_iterator end() const noexcept { return const_iterator(data() + size()); }

    Matrix()                                            = default;
    Matrix(Matrix<value_type, Rows, Cols> const& other) = default;
    Matrix(Matrix<value_type, Rows, Cols>&& other)      = default;
    Matrix<value_type, Rows, Cols>& operator=(Matrix<value_type, Rows, Cols> const& other) = default;
    Matrix<value_type, Rows, Cols>& operator=(Matrix<value_type, Rows, Cols>&& other) = default;

    Matrix(Tp value)
    {
        std::fill(begin(), end(), value);
    }

    template <size_type N>
    Matrix(Tp const (&initializer)[N])
    {
        static_assert((N == cols() && rows() == 1) || (N == rows() && cols() == 1));
        std::copy_n(&initializer[0], size(), begin());
    }

    template <size_t M, size_t N>
    Matrix(Tp const (&initializer)[M][N])
    {
        // If we have are getting passed a 1d init list, then make sure our matrix is a vector.
        if constexpr (M == 1)
        {
            static_assert((N == cols() && rows() == 1) || (N == rows() && cols() == 1));
        }
        else
        {
            static_assert(M == rows());
            static_assert(N == cols());
        }

        std::copy_n(&initializer[0][0], size(), begin());
    }

    // static
    // Matrix<value_type, Rows, Cols> zero()
    // {
    //     Matrix<value_type, Rows, Cols> m;
    //     ::zero(m);
    //     return m;
    // }

    // static
    // Matrix<value_type, Rows, Cols> I()
    // {
    //     static_assert(Rows == Cols, "Matrix is not square");

    //     Matrix<value_type, Rows, Cols> m;
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
            return _elems[index];
        }
        else if constexpr (Rows == 1)
        {
            assert(index < Cols);
            return _elems[index];
        }
        else
        {
            assert(index < Rows);
            return std::span<value_type, Cols>{&_elems[index * Cols], Cols};
        }
    }

    // Matrix<value_type, Rows, Cols>& operator=(Matrix<value_type, Rows, Cols> other)
    // {
    //     std::cout << "matrix matrix assign" << std::endl;
    //     this->_elems = std::make_shared<array2d>();

    //     for (size_t i = 0; i < Rows; ++i)
    //     {
    //         for (size_t j = 0; j < Cols; ++j)
    //         {
    //             (*_elems)[i][j] = (*other._elems)[i][j];
    //         }
    //     }
    //     return *this;
    // }

    // Matrix<value_type, Rows, Cols>& operator=(const std::array<value_type, Rows*Cols>
    // &other)
    // {
    //     std::cout << "matrix array assign" << std::endl;
    //     this->_elems = std::make_shared<array2d>();

    //     for (size_t i = 0; i < Rows; ++i)
    //     {
    //         for (size_t j = 0; j < Cols; ++j)
    //         {
    //             (*_elems)[i][j] = other[i*Cols + j];
    //         }
    //     }
    //     return (*this);
    // }
};

// template <class _Tp, class... _Args, class = std::enable_if<std::all<std::is_same<_Tp, _Args>::value...>::value>>
// array(_Tp, _Args...)
//     -> array<_Tp, 1 + sizeof...(_Args)>;

//////////////////////////////////////////////////////////////////////////////

// _idx avoids the use of Matrix::operator[].
// This  allows vectors and matrices to use the same code:
// i.e. vectors are matrices where one of the dimensions is 1.


template <typename MatRef,
          typename Mat = std::remove_reference_t<MatRef>>
auto _idx(MatRef&& mat, size_t r, size_t c) -> typename Mat::reference
{
    return mat._elems[(r * mat.cols()) + c];
}

/*
Explanation of template madness.

Operator* uses perfect forwarding so l and r value references can be passed to the function.
This means the type passed in could be a reference (i.e T&). You can't use :: on a reference, hence remove_reference_t.

Because operator* is templated, scalars also match, which we don't want (use SFINAE).
Remove scalars by enable_if_t. disjunction and is floating point is used to check both parameters are not floats.
*/
template <typename MLeftRef,
          typename MRightRef,
          typename MLeft  = std::remove_reference_t<MLeftRef>,
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator*(MLeftRef&& A, MRightRef&& B)
    -> std::enable_if_t<
        !std::disjunction_v<std::is_floating_point<MLeft>,
                            std::is_floating_point<MRight>>,
        Matrix<decltype(_idx(A, 0, 0) * _idx(B, 0, 0)),
               MLeft::rows(),
               MRight::cols()>>
{
    static_assert(MLeft::cols() == MRight::rows(),
                  "Invalid matrix dimensions.");

    Matrix<decltype(_idx(A, 0, 0) * _idx(B, 0, 0)),
           MLeft::rows(),
           MRight::cols()>
        result{0};

    size_t i, j, k;

    for (i = 0; i < MLeft::rows(); ++i)
    {
        for (j = 0; j < MRight::cols(); ++j)
        {
            for (k = 0; k < MLeft::cols(); ++k)
            {
                _idx(result, i, j) += _idx(A, i, k) * _idx(B, k, j);
            }
        }
    }

    return result;
}


template <typename MLeftRef,
          typename MLeft = std::remove_reference_t<MLeftRef>>
auto operator*(MLeftRef&& A, typename MLeft::value_type B)
{
    Matrix<decltype(_idx(std::forward<MLeftRef>(A), 0, 0) * B),
           MLeft::rows(),
           MLeft::cols()>
        result(A);

    for (auto& el : result._elems)
    {
        el *= B;
    }

    return result;
}

/*

template <typename MRight>
auto operator*(typename std::remove_reference_t<MRight>::value_type B, MRight&& A)
    -> Matrix<decltype(_idx(std::forward<MRight>(A), 0, 0) * B),
              MRight::rows(),
              MRight::cols()>
{
    return A * std::forward(B);
}
*/

template <typename MLeft, typename Tp>
auto operator*=(MLeft&& A, Tp B) -> MLeft
{
    for (auto& el : A._elems)
    {
        el *= B;
    }

    return A;
}


template <typename MLeft, typename MRight>
auto operator+(MLeft&& A, MRight&& B) -> Matrix<decltype(_idx(A, 0, 0) + _idx(B, 0, 0)), MLeft::rows(), MRight::cols()>
{
    static_assert(MLeft::cols() == MRight::cols(),
                  "Invalid matrix dimensions.");
    static_assert(MLeft::rows() == MRight::rows(),
                  "Invalid matrix dimensions.");

    Matrix<decltype(_idx(A, 0, 0) + _idx(B, 0, 0)), MLeft::rows(), MRight::cols()> result;

    size_t i, j;

    for (i = 0; i < MLeft::rows(); ++i)
    {
        for (j = 0; j < MRight::cols(); ++j)
        {
            _idx(result, i, j) = _idx(A, i, j) + _idx(B, i, j);
        }
    }

    return result;
}


template <typename MLeft>
auto operator+(MLeft&& A, typename MLeft::value_type B) -> Matrix<decltype(_idx(A, 0, 0) + B), MLeft::rows(), MLeft::cols()>
{
    Matrix<decltype(_idx(A, 0, 0) + B), MLeft::rows(), MLeft::cols()> result(A);

    for (auto& el : result._elems)
    {
        el += B;
    }

    return result;
}


template <typename MRight>
auto operator+(typename MRight::value_type B, MRight&& A) -> Matrix<decltype(_idx(A, 0, 0) + B), MRight::rows(), MRight::cols()>
{
    return A + B;
}

//////////////////////////////////////////////////////////////////////////////

template <typename MAT, size_t N>
auto cols(MAT& mat, int const (&values)[N]) -> linalg::Matrix<typename MAT::value_type, MAT::rows(), N>
{
    linalg::Matrix<typename MAT::value_type, MAT::rows(), N> other;

    int dst = 0;
    for (auto src : values)
    {
        assert(src < MAT::cols());

        for (int r : irange<MAT::rows()>())
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


template <typename value_type, size_t Rows>
using Vector = Matrix<value_type, Rows, 1>;

template <size_t Rows>
using Vectorf = Matrix<float, Rows, 1>;

template <size_t Rows>
using Vectord = Matrix<double, Rows, 1>;

} // end namespace linalg

//////////////////////////////////////////////////////////////////////////////

template <typename M>
auto iter(M& mat) -> std::array<std::span<typename M::value_type>, M::rows()>
{
    std::array<std::span<typename M::value_type>, M::rows()> v;
    for (auto i : irange<M::rows()>())
    {
        v[i] = mat[i];
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////

template <typename value_type, size_t Rows, size_t Cols>
std::ostream& operator<<(std::ostream&                           os,
                         linalg::Matrix<value_type, Rows, Cols>& A)
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