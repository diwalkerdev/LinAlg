#ifndef LINALG_UTIL_H
#define LINALG_UTIL_H

#include "linalg/matrix.hpp"

#include <cmath>

//////////////////////////////////////////////////////////////////////////////

auto lrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{
    return {{{cosf(alpha), sinf(alpha)},
             {-sinf(alpha), cosf(alpha)}}};
}

//////////////////////////////////////////////////////////////////////////////

auto rrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{

    return {{{cosf(alpha), -sinf(alpha)},
             {sinf(alpha), cosf(alpha)}}};
}

//////////////////////////////////////////////////////////////////////////////

auto ltransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    return {{{cosf(alpha), sinf(alpha), x},
             {-sinf(alpha), cosf(alpha), y},
             {0, 0, 1}}};
}

//////////////////////////////////////////////////////////////////////////////

auto rtransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    return {{{cosf(alpha), -sinf(alpha), 0},
             {sinf(alpha), cosf(alpha), 0},
             {x, y, 1}}};
}

//////////////////////////////////////////////////////////////////////////////

#endif // LINALG_UTIL_H