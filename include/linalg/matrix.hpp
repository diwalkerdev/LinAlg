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

// TODO: Use a more rebust method for determining if two floats are similar.
inline float epsilon = 0.000001;

float approximately_equal(float a, float b)
{
    return std::abs(a - b) < epsilon;
}

// TODO: Use floating_point concept again when osx updates.
// template <std::floating_point Tp, std::size_t Rows, std::size_t Cols>
template <typename Tp, std::size_t Rows, std::size_t Cols>
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

    Matrix()                                            = default;
    Matrix(Matrix<value_type, Rows, Cols> const& other) = default;
    Matrix(Matrix<value_type, Rows, Cols>&& other)      = default;
    Matrix<value_type, Rows, Cols>& operator=(Matrix<value_type, Rows, Cols> const& other) = default;
    Matrix<value_type, Rows, Cols>& operator=(Matrix<value_type, Rows, Cols>&& other) = default;

    auto _first() -> iterator { return &_elems[0]; }
    auto _first() const -> const_iterator { return &_elems[0]; }
    auto _last() -> iterator { return &_elems[size()]; }
    auto _last() const -> const_iterator { return &_elems[size()]; }

    Matrix(value_type value)
    {
        std::fill(_first(), _last(), value);
    }

    template <size_type N>
    Matrix(value_type const (&initializer)[N])
    {
        static_assert((N == cols() && rows() == 1) || (N == rows() && cols() == 1));
        std::copy_n(&initializer[0], size(), _first());
    }

    // template <size_t Rows, size_t Cols>
    Matrix(value_type const (&initializer)[Rows][Cols])
    {
        // If we have are getting passed a 1d init list, then make sure our matrix is a vector.
        if constexpr (Rows == 1)
        {
            static_assert((Cols == cols() && rows() == 1) || (Cols == rows() && cols() == 1));
        }
        else
        {
            static_assert(Rows == rows());
            static_assert(Cols == cols());
        }

        std::copy_n(&initializer[0][0], size(), _first());
    }

    static Matrix<value_type, rows(), cols()> I()
    {
        Matrix<value_type, rows(), cols()> result{0};

        for (auto i : irange<rows()>())
        {
            result[i][i] = 1;
        }

        return result;
    }

    auto T() -> Matrix<value_type, cols(), rows()>
    {
        Matrix<value_type, cols(), rows()> result;
        for (auto i : irange<rows()>())
        {
            for (auto k : irange<cols()>())
            {
                result[k][i] = (*this)[i][k];
            }
        }
        return result;
    }

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

    auto operator[](size_t index) const
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
            return std::span<const value_type, Cols>{&_elems[index * Cols], Cols};
        }
    }
};

//////////////////////////////////////////////////////////////////////////////

// Type Traits to identify matrix types
//
template <typename Tp>
struct is_matrix : std::false_type {
};

template <typename Tp, std::size_t M, std::size_t N>
struct is_matrix<Matrix<Tp, M, N>> : std::true_type {
};

template <typename Tp, std::size_t M, std::size_t N>
struct is_matrix<const Matrix<Tp, M, N>> : std::true_type {
};

template <typename T>
inline constexpr bool is_matrix_v = is_matrix<T>::value;

//////////////////////////////////////////////////////////////////////////////

// _idx avoids the use of Matrix::operator[].
// This  allows vectors and matrices to use the same code:
// i.e. vectors are matrices where one of the dimensions is 1.

template <typename MatRef,
          typename Mat = std::remove_reference_t<MatRef>>
auto _idx(MatRef const& mat, size_t r, size_t c)
    -> std::enable_if_t<
        is_matrix<Mat>::value,
        typename Mat::value_type>
{
    return mat._elems[(r * mat.cols()) + c];
}

template <typename MatRef,
          typename Mat = std::remove_reference_t<MatRef>>
auto _idx(MatRef&& mat, size_t r, size_t c)
    -> std::enable_if_t<
        is_matrix<Mat>::value,
        typename Mat::reference>
{
    return mat._elems[(r * mat.cols()) + c];
}

//////////////////////////////////////////////////////////////////////////////

/*
Explanation of template madness.

Operator* uses perfect forwarding so l and r value references can be passed to the function.
This means the type passed in could be a reference (i.e T&). You can't use :: on a reference, hence remove_reference_t.

Because operator* is templated, scalars also match, which we don't want (use SFINAE).
Remove scalars by enable_if_t. disjunction and is floating point is used to check both parameters are not floats.
*/

template <typename MLeftRef,
          typename MRightRef,
          typename MLeft        = std::remove_reference_t<MLeftRef>,
          typename MRight       = std::remove_reference_t<MRightRef>,
          typename return_value = decltype(
              typename MLeft::value_type{} * typename MRight::value_type{}),
          typename return_type = Matrix<return_value, MLeft::rows(), MRight::cols()>>
auto operator*(MLeftRef&& A, MRightRef&& B)
    -> std::enable_if_t<
        std::conjunction_v<is_matrix<MLeft>, is_matrix<MRight>>,
        return_type>
{
    static_assert(MLeft::cols() == MRight::rows(), "Invalid matrix dimensions.");

    return_type result{0};

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
          //typename T, // TODO: std::floating_point.
          typename MLeft = std::remove_reference_t<MLeftRef>,
          /*typename return_value = decltype(
              typename MLeft::value_type{} * T{}),*/
          typename return_type = MLeft>
auto operator*(MLeftRef&& A, float b)
    -> std::enable_if_t<is_matrix_v<MLeft>, MLeft>
{
    return_type result(A);

    for (auto& el : result._elems)
    {
        el *= b;
    }

    return result;
}


template <typename MRightRef,
          //typename T, // TODO: std::floating_point.
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator*(float b, MRightRef&& A)
    -> std::enable_if_t<is_matrix_v<MRight>, MRight>
{
    return A * b;
}


template <typename MLeftRef,
          //typename T, // TODO: std::floating_point.
          typename MLeft = std::remove_reference_t<MLeftRef>>
auto operator*=(MLeftRef&& A, float B)
    -> std::enable_if_t<is_matrix_v<MLeft>, MLeft>
{
    for (auto& el : A._elems)
    {
        el *= B;
    }

    return A;
}

//////////////////////////////////////////////////////////////////////////////
// Additive functions
//

template <typename MLeftRef,
          typename MRightRef,
          typename MLeft        = std::remove_reference_t<MLeftRef>,
          typename MRight       = std::remove_reference_t<MRightRef>,
          typename return_value = decltype(
              typename MLeft::value_type{} + typename MRight::value_type{}),
          typename return_type = Matrix<return_value, MLeft::rows(), MLeft::cols()>>
auto operator+(MLeftRef&& A, MRightRef&& B)
    -> std::enable_if_t<
        std::conjunction_v<is_matrix<MLeft>, is_matrix<MRight>>,
        return_type>
{
    static_assert(MLeft::cols() == MRight::cols(), "Invalid matrix dimensions.");
    static_assert(MLeft::rows() == MRight::rows(), "Invalid matrix dimensions.");

    return_type result;

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


template <typename MLeftRef,
          typename MLeft = std::remove_reference_t<MLeftRef>>
auto operator+(MLeftRef&& A, float B)
    -> std::enable_if_t<is_matrix_v<MLeft>, MLeft>
{
    auto result(A);

    for (auto& el : result._elems)
    {
        el += B;
    }

    return result;
}


template <typename MRightRef,
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator+(float B, MRightRef&& A)
    -> std::enable_if_t<is_matrix<MRight>::value, MRight>
{
    return A + B;
}

template <typename MLeftRef,
          typename MLeft = std::remove_reference_t<MLeftRef>>
auto operator+=(MLeftRef&& A, float B)
    -> std::enable_if_t<is_matrix<MLeft>::value, MLeft>
{
    for (auto& el : A._elems)
    {
        el += B;
    }

    return A;
}


template <typename MLeftRef,
          typename MRightRef,
          typename MLeft  = std::remove_reference_t<MLeftRef>,
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator+=(MLeftRef&& A, MRightRef&& B)
    -> std::enable_if_t<
        std::conjunction_v<is_matrix<MLeft>, is_matrix<MRight>>>
{
    static_assert(MLeft::cols() == MRight::cols(), "Invalid matrix dimensions.");
    static_assert(MLeft::rows() == MRight::rows(), "Invalid matrix dimensions.");

    size_t i, j;

    for (i = 0; i < MLeft::rows(); ++i)
    {
        for (j = 0; j < MRight::cols(); ++j)
        {
            _idx(A, i, j) += _idx(B, i, j);
        }
    }
}


//////////////////////////////////////////////////////////////////////////////
// equality functions
//
template <typename MLeftRef,
          typename MRightRef,
          typename MLeft  = std::remove_reference_t<MLeftRef>,
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator==(MLeftRef&& A, MRightRef&& B) -> bool
{
    static_assert(MLeft::cols() == MRight::cols(), "Invalid matrix dimensions.");
    static_assert(MLeft::rows() == MRight::rows(), "Invalid matrix dimensions.");

    return std::equal(A._first(),
                      A._last(),
                      B._first(),
                      [](float x, float y) -> bool {
                          return approximately_equal(x, y);
                      });
}

template <typename MLeftRef,
          typename MRightRef,
          typename MLeft  = std::remove_reference_t<MLeftRef>,
          typename MRight = std::remove_reference_t<MRightRef>>
auto operator!=(MLeftRef&& A, MRightRef&& B) -> bool
{
    static_assert(MLeft::cols() == MRight::cols(), "Invalid matrix dimensions.");
    static_assert(MLeft::rows() == MRight::rows(), "Invalid matrix dimensions.");

    return !(A == B);
}

//////////////////////////////////////////////////////////////////////////////
// Misc functions
//

template <typename MLeft,
          size_t N,
          typename return_value = typename MLeft::value_type,
          typename return_type  = linalg::Matrix<return_value, MLeft::rows(), N>>
auto cols(MLeft&& mat, int const (&values)[N])
    -> std::enable_if_t<is_matrix_v<MLeft>, return_type>
{
    return_type result;

    int dst = 0;
    for (auto src : values)
    {
        assert(src < MLeft::cols());

        for (int r : irange<MLeft::rows()>())
        {
            result[r][dst] = mat[r][src];
        }
        ++dst;
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////

} // end namespace linalg

//////////////////////////////////////////////////////////////////////////////

template <typename MLeftRef,
          typename MLeft        = std::remove_reference_t<MLeftRef>,
          typename return_value = std::span<const typename MLeft::value_type>,
          typename return_type  = std::array<return_value, MLeft::rows()>>
auto iter(MLeftRef const& mat)
    -> std::enable_if_t<linalg::is_matrix_v<MLeft>, return_type>
{
    return_type v;
    for (auto i : irange<MLeft::rows()>())
    {
        v[i] = mat[i];
    }
    return v;
}


template <typename MLeftRef,
          typename MLeft        = std::remove_reference_t<MLeftRef>,
          typename return_value = std::span<typename MLeft::value_type>,
          typename return_type  = std::array<return_value, MLeft::rows()>>
auto iter(MLeftRef&& mat)
    -> std::enable_if_t<linalg::is_matrix_v<MLeft>, return_type>
{
    return_type v;
    for (auto i : irange<MLeft::rows()>())
    {
        v[i] = mat[i];
    }
    return v;
}

//////////////////////////////////////////////////////////////////////////////

template <typename MRightRef,
          typename MRight      = std::remove_reference_t<MRightRef>,
          typename return_type = std::ostream&>
auto operator<<(std::ostream& os, MRightRef&& A)
    -> std::enable_if_t<linalg::is_matrix_v<MRight>, return_type>
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