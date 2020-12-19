#pragma once

#include "linalg/matrix.hpp"
#include <cmath>

//////////////////////////////////////////////////////////////////////////////

inline auto lrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{
    return {{{cosf(alpha), sinf(alpha)},
             {-sinf(alpha), cosf(alpha)}}};
}

//////////////////////////////////////////////////////////////////////////////

inline auto rrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{

    return {{{cosf(alpha), -sinf(alpha)},
             {sinf(alpha), cosf(alpha)}}};
}

//////////////////////////////////////////////////////////////////////////////

inline auto ltransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    return {{{cosf(alpha), sinf(alpha), x},
             {-sinf(alpha), cosf(alpha), y},
             {0, 0, 1}}};
}

//////////////////////////////////////////////////////////////////////////////

inline auto rtransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    return {{{cosf(alpha), -sinf(alpha), 0},
             {sinf(alpha), cosf(alpha), 0},
             {x, y, 1}}};
}

//////////////////////////////////////////////////////////////////////////////
