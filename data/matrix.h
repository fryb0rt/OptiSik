#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include "data/vector.h"

namespace OptiSik {

template<typename T>
class Matrix {
    std::vector<std::vector<T>> mData;
    size_t mRows;
    size_t mCols;

public:
    explicit Matrix(const size_t rows, const size_t cols) 
        : mRows(rows), mCols(cols) {
        mData.resize(rows, std::vector<T>(cols, T(0)));
    }

    explicit Matrix(const std::vector<std::vector<T>>& data) 
        : mData(data), mRows(data.size()), mCols(data.empty() ? 0 : data[0].size()) {
        if (mRows > 0) {
            for (const auto& row : mData) {
                if (row.size() != mCols) {
                    throw std::invalid_argument("Inconsistent row sizes");
                }
            }
        }
    }

    size_t rows() const { return mRows; }
    size_t cols() const { return mCols; }

    T& operator()(const size_t i, const size_t j) { return mData[i][j]; }
    const T& operator()(const size_t i, const size_t j) const { return mData[i][j]; }

    bool operator==(const Matrix& other) const {
        if (mRows != other.mRows || mCols != other.mCols) return false;
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                if (mData[i][j] != other.mData[i][j]) return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }

    Matrix operator+(const Matrix& other) const {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        Matrix result(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] + other.mData[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        Matrix result(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] - other.mData[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const T scalar) const {
        Matrix result(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix operator/(const T scalar) const {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        Matrix result(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] / scalar;
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (mCols != other.mRows) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
        Matrix result(mRows, other.mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < other.mCols; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < mCols; ++k) {
                    sum += mData[i][k] * other.mData[k][j];
                }
                result.mData[i][j] = sum;
            }
        }
        return result;
    }

    Vector<T> operator*(const Vector<T>& v) const {
        if (mCols != v.dimension()) {
            throw std::invalid_argument("Matrix columns must match vector dimension");
        }
        std::vector<T> result(mRows, T(0));
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result[i] += mData[i][j] * v[j];
            }
        }
        return Vector<T>(result);
    }

    Matrix transpose() const {
        Matrix result(mCols, mRows);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[j][i] = mData[i][j];
            }
        }
        return result;
    }

    T trace() const {
        if (mRows != mCols) {
            throw std::invalid_argument("Trace only defined for square matrices");
        }
        T sum = T(0);
        for (size_t i = 0; i < mRows; ++i) {
            sum += mData[i][i];
        }
        return sum;
    }

    void fill(const T value) {
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[i][j] = value;
            }
        }
    }

    static Matrix identity(const size_t n) {
        Matrix result(n, n);
        for (size_t i = 0; i < n; ++i) {
            result.mData[i][i] = T(1);
        }
        return result;
    }

    friend Matrix operator*(const T scalar, const Matrix& m) {
        return m * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "[";
        for (size_t i = 0; i < m.mRows; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < m.mCols; ++j) {
                os << m.mData[i][j];
                if (j + 1 < m.mCols) os << ", ";
            }
            os << "]";
            if (i + 1 < m.mRows) os << "\n";
        }
        os << "]";
        return os;
    }
};

} // namespace OptiSik
