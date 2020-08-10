#ifndef LINALG_MISC_HPP
#define LINALG_MISC_HPP

#include <array>
#include <numeric>
#include <span>
#include <vector>

//////////////////////////////////////////////////////////////////////////////

// It is intented that this function should eventually be replaced with iota_view.
// This function is really bad from a performance perspective because you have to
// allocate Nxint amount of memory to perform an iteration.
template <size_t StartValue, size_t EndValue>
std::vector<size_t> irange()
{
    static_assert(EndValue > StartValue);

    constexpr size_t size = EndValue - StartValue;

    std::vector<size_t> v(size);
    std::iota(v.begin(), v.end(), StartValue);
    return v;
}

//////////////////////////////////////////////////////////////////////////////

// It is intented that this function should eventually be replaced with iota_view.
template <size_t EndValue>
std::vector<size_t> irange()
{
    return irange<0, EndValue>();
}

//////////////////////////////////////////////////////////////////////////////

// Performs a deep copy of "values" into the span.
template <typename Span>
void span_deepcopy(Span row, std::array<typename Span::value_type, Span::extent> values)
{
    static_assert(Span::extent != std::dynamic_extent);

    std::copy(values.begin(), values.end(), row.begin());
}

//////////////////////////////////////////////////////////////////////////////

#endif // LINALG_MISC_HPP