#pragma once
#include "utils/exception.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <array>
#include "utils/traits.h"


namespace OptiSik {

template <typename T, typename TUnderlying = std::vector<T>> class Vector {
    TUnderlying mData;

    template<typename TOther, typename TUnderlyingOther>
    friend class Vector;
    public:
    explicit Vector () {
        static_assert(!IsDynamicArray<T, TUnderlying>::value, "The Vector() constructor can be used only for underlying types with implicit non-zero size.");
        emptyVectorCheck();
    }
    explicit Vector (const size_t n) : mData (n, T (0)) {
        static_assert(IsDynamicArrayV<T, TUnderlying>, "The Vector (const size_t n) can be only used for underlying dynamic types.");
        emptyVectorCheck();
    }

    explicit Vector (const TUnderlying& values) : mData (values) {
    }

    Vector(const Vector& other):mData(other.mData) {}
    
    template<typename U>
    explicit Vector(const Vector<T,U>& other):mData(init().mData) {
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] = other.mData[i];
        }
    }

    
    Vector& operator=(const Vector & other) {
        return operator=<TUnderlying>(other);
    }

    template<typename U>
    Vector& operator=(const Vector<T,U> & other) {
        checkSize(other);
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] = other.mData[i];
        }
        return *this;
    }
    
    size_t dimension () const {
        return mData.size ();
    }

    T& operator[] (const size_t i) {
        return mData[i];
    }
    const T& operator[] (const size_t i) const {
        return mData[i];
    }

    template<typename U>
    T dot (const Vector<T,U>& other) const {
        checkSize(other);
        T result = 0;
        for (size_t i = 0; i < mData.size(); ++i) {
            result += mData[i] * other.mData[i];
        }
        return result;
    }

    T squaredMagnitude () const {
        return dot (*this);
    }

    T magnitude () const {
        return sqrt (squaredMagnitude ());
    }

    template<typename U>
    Vector operator+ (const Vector<T,U>& other) const {
        checkSize(other);
        auto result = init();
        for (size_t i = 0; i < mData.size(); ++i) {
            result.mData[i] = mData[i] + other.mData[i];
        }
        return result;
    }

    template<typename U>
    Vector& operator+= (const Vector<T,U>& other) {
        checkSize(other);
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] += other.mData[i];
        }
        return *this;
    }

    Vector operator* (const T scalar) const {
        auto result = init();
        for (size_t i = 0; i < mData.size(); ++i) {
            result.mData[i] = mData[i] * scalar;
        }
        return result;
    }

    Vector& operator*= (const T scalar) {
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] *= scalar;
        }
        return *this;
    }

    Vector operator/ (const T scalar) const {
        auto result = init();
        for (size_t i = 0; i < mData.size(); ++i) {
            result.mData[i] = mData[i] / scalar;
        }
        return result;
    }

    Vector& operator/= (const T scalar) {
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] /= scalar;
        }
        return *this;
    }

    template<typename U>
    Vector operator- (const Vector<T,U>& other) const {
        checkSize(other);
        auto result = init();
        for (size_t i = 0; i < mData.size(); ++i) {
            result.mData[i] = mData[i] - other.mData[i];
        }
        return result;
    }

    template<typename U>
    Vector& operator-= (const Vector<T,U>& other) {
        checkSize(other);
        for (size_t i = 0; i < mData.size(); ++i) {
            mData[i] -= other.mData[i];
        }
        return *this;
    }

    Vector operator- () const {
        auto result = init();
        for (size_t i = 0; i < mData.size(); ++i)
            result.mData[i] = -mData[i];
        return result;
    }

    template<typename U>
    bool operator== (const Vector<T,U>& other) const {
        if (dimension () != other.dimension ())
            return false;
        for (size_t i = 0; i < mData.size(); ++i)
            if (mData[i] != other.mData[i])
                return false;
        return true;
    }

    template<typename U>
    bool operator!= (const Vector<T,U>& other) const {
        return !(*this == other);
    }

    Vector normalized () const {
        T mag = magnitude ();
        return (*this) / mag;
    }

    void normalize () {
        T mag = magnitude ();
        (*this) /= mag;
    }

    auto begin () {
        return mData.begin ();
    }
    auto end () {
        return mData.end ();
    }
    auto begin () const {
        return mData.begin ();
    }
    auto end () const {
        return mData.end ();
    }

    void fill (const T value) {
        std::fill (mData.begin (), mData.end (), value);
    }

    // This overload is needed otherwise calling min with the same Vector types could be ambiguous.
    friend Vector min (const Vector& a, const Vector& b) {
        return min<TUnderlying>(a,b);
    }

    template<typename U>
    friend Vector min (const Vector& a, const Vector<T,U>& b) {
        a.checkSize(b);
        auto result = a.init();
        for (size_t i = 0; i < a.mData.size(); ++i)
            result.mData[i] = std::min (a.mData[i], b.mData[i]);
        return result;
    }

    // This overload is needed otherwise calling max with the same Vector types could be ambiguous.
    friend Vector max (const Vector& a, const Vector& b) {
        return max<TUnderlying>(a,b);
    }

    template<typename U>
    friend Vector max (const Vector& a, const Vector<T,U>& b) {
        a.checkSize(b);
        auto result = a.init();
        for (size_t i = 0; i < a.mData.size(); ++i)
            result.mData[i] = std::max (a.mData[i], b.mData[i]);
        return result;
    }

    T minElement () const {
        emptyVectorCheck();
        T m = mData[0];
        for (size_t i = 1; i < mData.size(); ++i)
            if (mData[i] < m)
                m = mData[i];
        return m;
    }

    T maxElement () const {
        emptyVectorCheck();
        T m = mData[0];
        for (size_t i = 1; i < mData.size(); ++i)
            if (mData[i] > m)
                m = mData[i];
        return m;
    }

    template<typename U>
    bool epsilonEquals (const Vector<T,U>& other, const T eps) const {
        checkSize(other);
        T sum = T (0);
        for (size_t i = 0; i < mData.size(); ++i)
            sum += std::abs (mData[i] - other.mData[i]);
        return sum <= eps;
    }

    size_t minArg () const {
        emptyVectorCheck();
        size_t idx = 0;
        for (size_t i = 1; i < mData.size(); ++i)
            if (mData[i] < mData[idx])
                idx = i;
        return idx;
    }

    size_t maxArg () const {
        emptyVectorCheck();
        size_t idx = 0;
        for (size_t i = 1; i < mData.size(); ++i)
            if (mData[i] > mData[idx])
                idx = i;
        return idx;
    }

    T average () const {
        emptyVectorCheck();
        T sum = T (0);
        for (size_t i = 0; i < mData.size(); ++i)
            sum += mData[i];
        return sum / static_cast<T> (mData.size());
    }

    friend Vector operator* (const T scalar, const Vector& v) {
        return v * scalar;
    }

    friend std::ostream& operator<< (std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.mData.size(); ++i) {
            os << v.mData[i];
            if (i + 1 < v.mData.size())
                os << ", ";
        }
        os << "]";
        return os;
    }

private:
    Vector init() const {
        if constexpr(IsDynamicArrayV<T, TUnderlying>) {
            return Vector(dimension());
        } else {
            return Vector();
        }
    }

    template<typename U>
    void checkSize(const Vector<T,U>& other) const {
        if constexpr(IsDynamicArrayV<T, TUnderlying> || IsDynamicArrayV<T, U>) {
            if (dimension () != other.dimension ()) {
                throw invalidArgument ("Dimension mismatch");
            }
        } else {
            static_assert(mData.size() == other.mData.size(), "Dimension mismatch");
        }
    }

    void emptyVectorCheck() const {
        if constexpr(IsDynamicArrayV<T, TUnderlying>) {
            if (dimension () == 0) {
                throw invalidArgument ("The Vector must have non-zero size");
            }
        } else {
            static_assert(mData.size() > 0, "The Vector must have non-zero size");
        }
    }
};

template<typename T, size_t size>
using SVector = Vector<T, std::array<T, size>>;

} // namespace OptiSik