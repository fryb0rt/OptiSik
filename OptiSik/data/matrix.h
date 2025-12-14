#pragma once
#include "data/vector.h"
#include "utils/exception.h"
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>


namespace OptiSik {

template <typename T> class Matrix {
    std::vector<Vector<T>> mData;
    size_t mCols;

    public:
    explicit Matrix (const size_t rows, const size_t cols)
    : mData (rows, Vector<T> (cols)), mCols (cols) {
    }

    explicit Matrix (const std::vector<std::vector<T>>& data)
    : mCols (data.empty () ? 0 : data[0].size ()) {
        mData.reserve (data.size ());
        for (const auto& row : data) {
            if (row.size () != mCols) {
                throw invalidArgument ("Inconsistent row sizes");
            }
            mData.emplace_back (row);
        }
    }

    explicit Matrix (const std::vector<Vector<T>>& rows)
    : mData (rows), mCols (rows.empty () ? 0 : rows[0].dimension ()) {
        if (!mData.empty ()) {
            for (const auto& row : mData) {
                if (row.dimension () != mCols) {
                    throw invalidArgument ("Inconsistent row sizes");
                }
            }
        }
    }

    size_t rows () const {
        return mData.size ();
    }
    size_t cols () const {
        return mCols;
    }

    T& operator() (const size_t i, const size_t j) {
        return mData[i][j];
    }
    const T& operator() (const size_t i, const size_t j) const {
        return mData[i][j];
    }

    bool operator== (const Matrix& other) const {
        if (rows () != other.rows () || mCols != other.mCols)
            return false;
        for (size_t i = 0; i < rows (); ++i) {
            if (mData[i] != other.mData[i])
                return false;
        }
        return true;
    }

    bool operator!= (const Matrix& other) const {
        return !(*this == other);
    }

    Matrix operator+ (const Matrix& other) const {
        if (rows () != other.rows () || mCols != other.mCols) {
            throw invalidArgument ("Matrix dimensions mismatch");
        }
        Matrix result (rows (), mCols);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] + other.mData[i][j];
            }
        }
        return result;
    }

    Matrix operator- (const Matrix& other) const {
        if (rows () != other.rows () || mCols != other.mCols) {
            throw invalidArgument ("Matrix dimensions mismatch");
        }
        Matrix result (rows (), mCols);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] - other.mData[i][j];
            }
        }
        return result;
    }

    Matrix operator* (const T scalar) const {
        Matrix result (rows (), mCols);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix operator/ (const T scalar) const {
        if (scalar == T (0)) {
            throw invalidArgument ("Division by zero");
        }
        Matrix result (rows (), mCols);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[i][j] = mData[i][j] / scalar;
            }
        }
        return result;
    }

    Matrix operator* (const Matrix& other) const {
        if (mCols != other.rows ()) {
            throw invalidArgument (
            "Matrix dimensions incompatible for multiplication");
        }
        Matrix result (rows (), other.mCols);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < other.mCols; ++j) {
                T sum = T (0);
                for (size_t k = 0; k < mCols; ++k) {
                    sum += mData[i][k] * other.mData[k][j];
                }
                result.mData[i][j] = sum;
            }
        }
        return result;
    }

    Vector<T> operator* (const Vector<T>& v) const {
        if (mCols != v.dimension ()) {
            throw invalidArgument (
            "Matrix columns must match vector dimension");
        }
        Vector<T> result (rows ());
        for (size_t i = 0; i < rows (); ++i) {
            result[i] = mData[i].dot (v);
        }
        return result;
    }

    Matrix transpose () const {
        Matrix result (mCols, rows ());
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                result.mData[j][i] = mData[i][j];
            }
        }
        return result;
    }

    T trace () const {
        if (rows () != mCols) {
            throw invalidArgument ("Trace only defined for square matrices");
        }
        T sum = T (0);
        for (size_t i = 0; i < rows (); ++i) {
            sum += mData[i][i];
        }
        return sum;
    }

    void fill (const T value) {
        for (size_t i = 0; i < rows (); ++i) {
            mData[i].fill (value);
        }
    }

    bool epsilonEquals (const Matrix& other, const T eps) const {
        if (rows () != other.rows () || mCols != other.mCols)
            return false;
        T sum = T (0);
        for (size_t i = 0; i < rows (); ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                sum += std::abs (mData[i][j] - other.mData[i][j]);
            }
        }
        return sum <= eps;
    }

    static Matrix identity (const size_t n) {
        Matrix result (n, n);
        for (size_t i = 0; i < n; ++i) {
            result.mData[i][i] = T (1);
        }
        return result;
    }

    friend Matrix operator* (const T scalar, const Matrix& m) {
        return m * scalar;
    }

    friend std::ostream& operator<< (std::ostream& os, const Matrix& m) {
        os << "[";
        for (size_t i = 0; i < m.rows (); ++i) {
            if (i > 0)
                os << " ";
            os << "[";
            for (size_t j = 0; j < m.mCols; ++j) {
                os << m.mData[i][j];
                if (j + 1 < m.mCols)
                    os << ", ";
            }
            os << "]";
            if (i + 1 < m.rows ())
                os << "\n";
        }
        os << "]";
        return os;
    }
};

} // namespace OptiSik
