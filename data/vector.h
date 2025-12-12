#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include <algorithm>
#include <limits>

template<typename T>
class Vector {
    std::vector<T> mData;
public:
    explicit Vector(const size_t n) : mData(n, T(0)) {}
    
    explicit Vector(const std::vector<T>& values) : mData(values) {}
    
    size_t dimension() const { return mData.size(); }

    T& operator[](const size_t i) { return mData[i]; }
    const T& operator[](const size_t i) const { return mData[i]; }

    T dot(const Vector& other) const {
        if (dimension() != other.dimension()) {
            throw std::invalid_argument("Dimension mismatch");
        }
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
    
    Vector operator+(const Vector& other) const {
        if (dimension() != other.dimension()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) {
            result[i] = mData[i] + other.mData[i];
        }
        return Vector(result);
    }
    
    Vector& operator+=(const Vector& other) {
        if (dimension() != other.dimension()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] += other.mData[i];
        }
        return *this;
    }
    
    Vector operator*(const T scalar) const {
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) {
            result[i] = mData[i] * scalar;
        }
        return Vector(result);
    }

    Vector& operator*=(const T scalar) {
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] *= scalar;
        }
        return *this;
    }

    Vector operator/(const T scalar) const {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) {
            result[i] = mData[i] / scalar;
        }
        return Vector(result);
    }

    Vector& operator/=(const T scalar) {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] /= scalar;
        }
        return *this;
    }

    Vector operator-(const Vector& other) const {
        if (dimension() != other.dimension()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) {
            result[i] = mData[i] - other.mData[i];
        }
        return Vector(result);
    }

    Vector& operator-=(const Vector& other) {
        if (dimension() != other.dimension()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        for (size_t i = 0; i < dimension(); ++i) {
            mData[i] -= other.mData[i];
        }
        return *this;
    }

    Vector operator-() const {
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) result[i] = -mData[i];
        return Vector(result);
    }

    bool operator==(const Vector& other) const {
        if (dimension() != other.dimension()) return false;
        for (size_t i = 0; i < dimension(); ++i) if (mData[i] != other.mData[i]) return false;
        return true;
    }

    bool operator!=(const Vector& other) const {
        return !(*this == other);
    }

    Vector normalized() const {
        T mag = magnitude();
        if (mag == T(0)) {
            throw std::invalid_argument("Cannot normalize zero-length vector");
        }
        return (*this) / mag;
    }

    void normalize() {
        T mag = magnitude();
        if (mag == T(0)) {
            throw std::invalid_argument("Cannot normalize zero-length vector");
        }
        (*this) /= mag;
    }

    auto begin() { return mData.begin(); }
    auto end() { return mData.end(); }
    auto begin() const { return mData.begin(); }
    auto end() const { return mData.end(); }

    void fill(const T value) { std::fill(mData.begin(), mData.end(), value); }

    Vector min(const Vector& other) const {
        if (dimension() != other.dimension()) throw std::invalid_argument("Dimension mismatch");
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) result[i] = std::min(mData[i], other.mData[i]);
        return Vector(result);
    }

    Vector max(const Vector& other) const {
        if (dimension() != other.dimension()) throw std::invalid_argument("Dimension mismatch");
        std::vector<T> result(dimension());
        for (size_t i = 0; i < dimension(); ++i) result[i] = std::max(mData[i], other.mData[i]);
        return Vector(result);
    }

    T minElement() const {
        if (dimension() == 0) throw std::invalid_argument("Empty vector");
        T m = mData[0];
        for (size_t i = 1; i < dimension(); ++i) if (mData[i] < m) m = mData[i];
        return m;
    }

    T maxElement() const {
        if (dimension() == 0) throw std::invalid_argument("Empty vector");
        T m = mData[0];
        for (size_t i = 1; i < dimension(); ++i) if (mData[i] > m) m = mData[i];
        return m;
    }

    bool epsilonEquals(const Vector& other, const T eps) const {
        if (dimension() != other.dimension()) return false;
        T sum = T(0);
        for (size_t i = 0; i < dimension(); ++i) sum += std::abs(mData[i] - other.mData[i]);
        return sum <= eps;
    }

    size_t minArg() const {
        if (dimension() == 0) throw std::invalid_argument("Empty vector");
        size_t idx = 0;
        for (size_t i = 1; i < dimension(); ++i) if (mData[i] < mData[idx]) idx = i;
        return idx;
    }

    size_t maxArg() const {
        if (dimension() == 0) throw std::invalid_argument("Empty vector");
        size_t idx = 0;
        for (size_t i = 1; i < dimension(); ++i) if (mData[i] > mData[idx]) idx = i;
        return idx;
    }

    T average() const {
        if (dimension() == 0) throw std::invalid_argument("Empty vector");
        T sum = T(0);
        for (size_t i = 0; i < dimension(); ++i) sum += mData[i];
        return sum / static_cast<T>(dimension());
    }

    friend Vector operator*(const T scalar, const Vector& v) {
        return v * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.dimension(); ++i) {
            os << v.mData[i];
            if (i + 1 < v.dimension()) os << ", ";
        }
        os << "]";
        return os;
    }
};