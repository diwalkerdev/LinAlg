#ifndef LINALG_UTIL_H
#define LINALG_UTIL_H

#include "linalg/matrix.h"
#include <cmath>


linalg::Matrix<float, 2, 2> lrotzf(float alpha)
{
    // clang-format off
    return {{
        cosf(alpha), sinf(alpha), 
        -sinf(alpha), cosf(alpha)
    }};
    // clang-format on
}


linalg::Matrix<float, 2, 2> rrotzf(float alpha)
{
    // clang-format off
    return {{
        cosf(alpha), -sinf(alpha), 
        sinf(alpha), cosf(alpha)
    }};
    // clang-format on
}


linalg::Matrix<float, 2, 3> ltransf(float alpha, float x, float y)
{
    // clang-format off
    return {{
        cosf(alpha), sinf(alpha), x,
        -sinf(alpha), cosf(alpha) , y,
    }};
    // clang-format on
}

linalg::Matrix<float, 3, 2> rtransf(float alpha, float x, float y)
{
    // clang-format off
    return {{
        cosf(alpha), -sinf(alpha),
        sinf(alpha),  cosf(alpha) ,
        x,            y
    }};
    // clang-format on
}

#endif // LINALG_UTIL_H