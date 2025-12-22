#pragma once
#include "utils/exception.h"
#include "utils/traits.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>


namespace OptiSik {

template <typename T, typename TUnderlying = std::vector<T>>
class Vector {
    TUnderlying mData;

    template <typename TOther, typename TUnderlyingOther>
    friend class Vector;

public:
    using Type                      = T;
    using UnderlyingType            = TUnderlying;
    static constexpr bool isDynamic = IsDynamicArrayV<T, TUnderlying>;

    explicit Vector() {
        static_assert(!isDynamic,
                      "The Vector() constructor can be used only for "
                      "underlying types with implicit non-zero size.");
        checkSizeNonZero();
    }
    explicit Vector(const size_t n) : mData(n, T(0)) {
        static_assert(isDynamic,
                      "The Vector (const size_t n) can be only used for "
                      "underlying dynamic types.");
        checkSizeNonZero();
    }

    explicit Vector(const TUnderlying& values) : mData(values) {
    }

    Vector(const Vector& other) : mData(other.mData) {
    }

    template <typename T2, typename U>
    Vector(const Vector<T2, U>& other) : mData(init(other.dimension())) {
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] = static_cast<T>(other.mData[i]);
        }
    }

    Vector& operator=(const Vector& other) {
        return operator= <TUnderlying>(other);
    }

    template <typename U>
    Vector& operator=(const Vector<T, U>& other) {
        checkSizeEqual(other);
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] = other.mData[i];
        }
        return *this;
    }

    constexpr size_t dimension() const {
        return mData.size();
    }

    T& operator[](const size_t i) {
        return mData[i];
    }
    const T& operator[](const size_t i) const {
        return mData[i];
    }

    template <typename U>
    T dot(const Vector<T, U>& other) const {
        checkSizeEqual(other);
        T result = 0;
        for (size_t i = 0; i < dimension(); ++i) {
            result += mData[i] * other.mData[i];
        }
        return result;
    }

    T squaredMagnitude() const {
        return dot(*this);
    }

    T magnitude() const {
        return sqrt(squaredMagnitude());
    }

    template <typename U>
    Vector operator+(const Vector<T, U>& other) const {
        checkSizeEqual(other);
        auto result = init();
        for (size_t i = 0; i < dimension(); ++i) {
            result.mData[i] = mData[i] + other.mData[i];
        }
        return result;
    }

    template <typename U>
    Vector& operator+=(const Vector<T, U>& other) {
        checkSizeEqual(other);
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] += other.mData[i];
        }
        return *this;
    }

    Vector operator*(const T scalar) const {
        auto result = init();
        for (size_t i = 0; i < dimension(); ++i) {
            result.mData[i] = mData[i] * scalar;
        }
        return result;
    }

    Vector& operator*=(const T scalar) {
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] *= scalar;
        }
        return *this;
    }

    Vector operator/(const T scalar) const {
        auto result = init();
        for (size_t i = 0; i < dimension(); ++i) {
            result.mData[i] = mData[i] / scalar;
        }
        return result;
    }

    Vector& operator/=(const T scalar) {
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] /= scalar;
        }
        return *this;
    }

    template <typename U>
    Vector operator-(const Vector<T, U>& other) const {
        checkSizeEqual(other);
        auto result = init();
        for (size_t i = 0; i < dimension(); ++i) {
            result.mData[i] = mData[i] - other.mData[i];
        }
        return result;
    }

    template <typename U>
    Vector& operator-=(const Vector<T, U>& other) {
        checkSizeEqual(other);
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] -= other.mData[i];
        }
        return *this;
    }

    Vector operator-() const {
        auto result = init();
        for (size_t i = 0; i < dimension(); ++i)
            result.mData[i] = -mData[i];
        return result;
    }

    template <typename U>
    bool operator==(const Vector<T, U>& other) const {
        if (dimension() != other.dimension())
            return false;
        for (size_t i = 0; i < dimension(); ++i)
            if (mData[i] != other.mData[i])
                return false;
        return true;
    }

    template <typename U>
    bool operator!=(const Vector<T, U>& other) const {
        return !(*this == other);
    }

    Vector normalized() const {
        T mag = magnitude();
        return (*this) / mag;
    }

    void normalize() {
        T mag = magnitude();
        (*this) /= mag;
    }

    auto begin() {
        return mData.begin();
    }
    auto end() {
        return mData.end();
    }
    auto begin() const {
        return mData.begin();
    }
    auto end() const {
        return mData.end();
    }

    void fill(const T value) {
        std::fill(mData.begin(), mData.end(), value);
    }

    // This overload is needed otherwise calling min with the same Vector types could be ambiguous.
    friend Vector min(const Vector& a, const Vector& b) {
        return min<TUnderlying>(a, b);
    }

    template <typename U>
    friend Vector min(const Vector& a, const Vector<T, U>& b) {
        a.checkSizeEqual(b);
        auto result = a.init();
        for (size_t i = 0; i < a.dimension(); ++i)
            result.mData[i] = std::min(a.mData[i], b.mData[i]);
        return result;
    }

    // This overload is needed otherwise calling max with the same Vector types could be ambiguous.
    friend Vector max(const Vector& a, const Vector& b) {
        return max<TUnderlying>(a, b);
    }

    template <typename U>
    friend Vector max(const Vector& a, const Vector<T, U>& b) {
        a.checkSizeEqual(b);
        auto result = a.init();
        for (size_t i = 0; i < a.dimension(); ++i)
            result.mData[i] = std::max(a.mData[i], b.mData[i]);
        return result;
    }

    T minElement() const {
        checkSizeNonZero();
        T m = mData[0];
        for (size_t i = 1; i < dimension(); ++i)
            if (mData[i] < m)
                m = mData[i];
        return m;
    }

    T maxElement() const {
        checkSizeNonZero();
        T m = mData[0];
        for (size_t i = 1; i < dimension(); ++i)
            if (mData[i] > m)
                m = mData[i];
        return m;
    }

    template <typename U>
    bool epsilonEquals(const Vector<T, U>& other, const T eps) const {
        checkSizeEqual(other);
        T sum = T(0);
        for (size_t i = 0; i < dimension(); ++i)
            sum += std::abs(mData[i] - other.mData[i]);
        return sum <= eps;
    }

    size_t minArg() const {
        checkSizeNonZero();
        size_t idx = 0;
        for (size_t i = 1; i < dimension(); ++i)
            if (mData[i] < mData[idx])
                idx = i;
        return idx;
    }

    size_t maxArg() const {
        checkSizeNonZero();
        size_t idx = 0;
        for (size_t i = 1; i < dimension(); ++i)
            if (mData[i] > mData[idx])
                idx = i;
        return idx;
    }

    T average() const {
        checkSizeNonZero();
        T sum = T(0);
        for (size_t i = 0; i < dimension(); ++i)
            sum += mData[i];
        return sum / static_cast<T>(dimension());
    }

    friend Vector operator*(const T scalar, const Vector& v) {
        return v * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.dimension(); ++i) {
            os << v.mData[i];
            if (i + 1 < v.dimension())
                os << ", ";
        }
        os << "]";
        return os;
    }

private:
    Vector init() const {
        if constexpr (isDynamic) {
            return Vector(dimension());
        } else {
            return Vector();
        }
    }

    TUnderlying init(const size_t size) const {
        if constexpr (isDynamic) {
            return TUnderlying(size);
        } else {
            if (dimension() != size) {
                throw invalidArgument("Dimension mismatch");
            }
            return TUnderlying();
        }
    }

    template <typename U>
    void checkSizeEqual(const Vector<T, U>& other) const {
        if constexpr (isDynamic || Vector<T, U>::isDynamic) {
            if (dimension() != other.dimension()) {
                throw invalidArgument("Dimension mismatch");
            }
        } else {
            static_assert(dimension() == other.dimension(), "Dimension mismatch");
        }
    }

    void checkSizeNonZero() const {
        if constexpr (isDynamic) {
            if (dimension() == 0) {
                throw invalidArgument("The Vector must have non-zero size");
            }
        } else {
            static_assert(dimension() > 0, "The Vector must have non-zero size");
        }
    }
};

template <typename T, size_t N>
using SVector = Vector<T, std::array<T, N>>;

template <typename TVector, typename = std::enable_if_t<!TVector::isDynamic>>
constexpr size_t vectorSize = std::tuple_size_v<typename TVector::UnderlyingType>;

template <typename T>
struct IsVector {
    static constexpr bool value = false;
};

template <typename T, typename U>
struct IsVector<Vector<T, U>> {
    static constexpr bool value = true;
};

} // namespace OptiSik