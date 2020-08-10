#ifndef LINALG_UTIL_H
#define LINALG_UTIL_H

#include "linalg/matrix.hpp"

#include <cmath>

//////////////////////////////////////////////////////////////////////////////

auto lrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{
    // clang-format off
    return {{
        cosf(alpha), sinf(alpha),
        -sinf(alpha), cosf(alpha)
    }};
    // clang-format on
}

//////////////////////////////////////////////////////////////////////////////

auto rrotzf(float alpha) -> linalg::Matrix<float, 2, 2>
{
    // clang-format off
    return {{
        cosf(alpha), -sinf(alpha),
        sinf(alpha), cosf(alpha)
    }};
    // clang-format on
}

//////////////////////////////////////////////////////////////////////////////

auto ltransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    // clang-format off
    return {{
         cosf(alpha), sinf(alpha), x,
        -sinf(alpha), cosf(alpha), y,
         0,                     0, 1
    }};
    // clang-format on
}

//////////////////////////////////////////////////////////////////////////////

auto rtransf(float alpha, float x, float y) -> linalg::Matrix<float, 3, 3>
{
    // clang-format off
    return {{
        cosf(alpha), -sinf(alpha), 0,
        sinf(alpha),  cosf(alpha), 0,
        x,            y          , 1
    }};
    // clang-format on
}

//////////////////////////////////////////////////////////////////////////////

#endif // LINALG_UTIL_H