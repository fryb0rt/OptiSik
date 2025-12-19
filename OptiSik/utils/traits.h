#pragma once
#include <vector>

namespace OptiSik {

template<typename T, typename TContainer>
struct IsDynamicArray {
    static constexpr bool value = false;
};

template<typename T>
struct IsDynamicArray<T,std::vector<T>> {
    static constexpr bool value = true;
};

template<typename T, typename TContainer>
constexpr bool IsDynamicArrayV = IsDynamicArray<T,TContainer>::value;

} // namespace OptiSik