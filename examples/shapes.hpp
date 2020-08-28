#ifndef SHAPE_ATTACK_SHAPES_HPP
#define SHAPE_ATTACK_SHAPES_HPP

#include <linalg/matrix.hpp>

//////////////////////////////////////////////////////////////////////////////

template <size_t NumSegments>
auto make_circle_points(float radius) -> linalg::Matrixf<NumSegments + 2, 3>
{
    static_assert(NumSegments >= 3);
    constexpr float seg_size = (2 * M_PI / NumSegments);

    linalg::Matrixf<NumSegments + 2, 3> points;

    float theta = 0;

    points[0][0] = 0;
    points[0][1] = 0;
    points[0][2] = 1;

    for (int i = 1; i < NumSegments + 2; ++i)
    {
        points[i][0] = cos(theta) * radius;
        points[i][1] = sin(theta) * radius;
        points[i][2] = 1;
        theta += seg_size;
    }

    return points;
}

//////////////////////////////////////////////////////////////////////////////

template <size_t NumSegments>
struct Circle {
    float radius;
    float offset_x;
    float offset_y;
    float theta;

    linalg::Matrixf<NumSegments + 2, 3> data;
    Circle(float radius, float offset_x = 0, float offset_y = 0)
        : radius(radius)
        , offset_x(offset_x)
        , offset_y(offset_y)
        , theta(0)
        , data(make_circle_points<NumSegments>(radius))
    {
    }
};

//////////////////////////////////////////////////////////////////////////////

template <int NumSegments>
auto make_circle(float circumference) -> linalg::Matrixf<NumSegments + 2, 3>
{
    return make_circle_points<NumSegments>(circumference / 2);
}

//////////////////////////////////////////////////////////////////////////////

auto make_square(float width) -> linalg::Matrixf<4 + 2, 3>
{
    return make_circle_points<4>(width / 2);
}

//////////////////////////////////////////////////////////////////////////////

auto make_triangle(float length) -> linalg::Matrixf<3 + 2, 3>
{
    return make_circle_points<3>(length / 2);
}

//////////////////////////////////////////////////////////////////////////////

#endif // SHAPE_ATTACK_SHAPES_HPP