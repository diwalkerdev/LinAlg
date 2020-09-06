#ifndef LINALG_MISC_HPP
#define LINALG_MISC_HPP

#include <array>

//////////////////////////////////////////////////////////////////////////////
// It is intented that this function should eventually be replaced with iota_view.
// This function is really bad from a performance perspective because you have to
// allocate Nxint amount of memory to perform an iteration.

template <size_t StartValue, size_t EndValue, size_t Size = EndValue - StartValue>
std::array<int, Size> irange()
{
    static_assert(Size > 0);

    std::array<int, Size> indices;

    for (int i = 0; i < Size; ++i)
    {
        indices[i] = StartValue + i;
    }
    return indices;
}

//////////////////////////////////////////////////////////////////////////////

// It is intented that this function should eventually be replaced with iota_view.
template <size_t EndValue>
std::array<int, EndValue> irange()
{
    return irange<0, EndValue>();
}

//////////////////////////////////////////////////////////////////////////////

// Performs a deep copy of "values" into the span.
// TODO: Rename to copy_from
// template <typename Span>
// void span_deepcopy(Span row, std::array<typename Span::value_type, Span::extent> values)
// {
//     static_assert(Span::extent != std::dynamic_extent);

//     std::copy(values.begin(), values.end(), row.begin());
// }


//////////////////////////////////////////////////////////////////////////////

#endif // LINALG_MISC_HPP