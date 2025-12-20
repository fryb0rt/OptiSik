#pragma once
#include "data/vector.h"
#include "utils/exception.h"
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace OptiSik {

template <typename TVector,
          typename TUnderlying = std::vector<TVector>,
          typename = std::enable_if_t<TVector::isDynamic == IsDynamicArrayV<TVector, TUnderlying>>>
class Matrix {
    TUnderlying mData;

    template <typename TVectorOther, typename TUnderlyingOther, typename>
    friend class Matrix;

public:
    using Type                      = typename TVector::Type;
    using RowType                   = TVector;
    using UnderlyingType            = TUnderlying;
    static constexpr bool isDynamic = IsDynamicArrayV<TVector, TUnderlying>;

    explicit Matrix() {
        static_assert(!isDynamic,
                      "The Matrix() constructor can be used only for "
                      "underlying types with implicit non-zero size.");
        checkSizeNonZero();
    }
    explicit Matrix(const size_t rows, const size_t cols)
    : mData(rows, TVector(cols)) {
        static_assert(isDynamic,
                      "The Matrix (rows, cols) can be only used for "
                      "underlying dynamic types.");
        checkSizeNonZero();
    }

    Matrix(const TUnderlying& values) : mData(values) {
    }

    Matrix(const Matrix& other) : mData(other.mData) {
    }

    template <typename U1, typename U2>
    Matrix(const Matrix<Vector<Type, U1>, U2>& other)
    : mData(init(other.rows(), other.cols())) {
        for (size_t i = 0; i < rows(); ++i) {
            mData[i] = other.mData[i];
        }
    }

    Matrix& operator=(const Matrix& other) {
        return operator= <TUnderlying>(other);
    }

    template <typename U1, typename U2>
    Matrix& operator=(const Matrix<Vector<Type, U1>, U2>& other) {
        checkSizeEqual(other);
        for (size_t i = 0; i < rows(); ++i) {
            mData[i] = other.mData[i];
        }
        return *this;
    }

    constexpr size_t rows() const {
        return mData.size();
    }
    constexpr size_t cols() const {
        return mData[0].dimension();
    }

    Type& operator()(const size_t i, const size_t j) {
        return mData[i][j];
    }
    const Type& operator()(const size_t i, const size_t j) const {
        return mData[i][j];
    }

    template <typename U1, typename U2>
    bool operator==(const Matrix<Vector<Type, U1>, U2>& other) const {
        if (rows() != other.rows() || cols() != other.cols())
            return false;
        for (size_t i = 0; i < rows(); ++i) {
            if (mData[i] != other.mData[i])
                return false;
        }
        return true;
    }

    template <typename U1, typename U2>
    bool operator!=(const Matrix<Vector<Type, U1>, U2>& other) const {
        return !(*this == other);
    }

    template <typename U1, typename U2>
    Matrix operator+(const Matrix<Vector<Type, U1>, U2>& other) const {
        checkSizeEqual(other);
        auto result = init();
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result.mData[i][j] = mData[i][j] + other.mData[i][j];
            }
        }
        return result;
    }

    template <typename U1, typename U2>
    Matrix operator-(const Matrix<Vector<Type, U1>, U2>& other) const {
        checkSizeEqual(other);
        auto result = init();
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result.mData[i][j] = mData[i][j] - other.mData[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Type scalar) const {
        auto result = init();
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result.mData[i][j] = mData[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix operator/(const Type scalar) const {
        auto result = init();
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result.mData[i][j] = mData[i][j] / scalar;
            }
        }
        return result;
    }

    template <typename U1, typename U2>
    auto operator*(const Matrix<Vector<Type, U1>, U2>& other) const {
        if constexpr (isDynamic || Matrix<Vector<Type, U1>, U2>::isDynamic) {
            if (cols() != other.rows()) {
                throw invalidArgument("Dimension mismatch");
            }
            Matrix<Vector<Type>> result(rows(), other.cols());
            doMatrixMult(other, result);
            return result;
        } else {
            static_assert(cols() == other.rows(), "Dimension mismatch");
            using Vec = SVector<Type, other.cols()>;
            Matrix<Vec, std::array<Vec, rows()>> result;
            doMatrixMult(other, result);
            return result;
        }
    }

    template <typename U1>
    auto operator*(const Vector<Type, U1>& v) const {
        if constexpr (isDynamic || Vector<Type, U1>::isDynamic) {
            if (cols() != v.dimension()) {
                throw invalidArgument("Dimension mismatch");
            }
            Vector<Type> result(rows()); // Dynamic size
            doVectorMult(v, result);
            return result;
        } else {
            static_assert(cols() == v.dimension(), "Dimension mismatch");
            Vector<Type, std::array<Type, rows()>> result; // Static size
            doVectorMult(v, result);
            return result;
        }
    }

    auto transpose() const {
        if constexpr (isDynamic) {
            Matrix<Vector<Type>> result(cols(), rows()); // Dynamic
            doMatrixTranspose(result);
            return result;
        } else {
            using Vec = SVector<Type, rows()>;
            Matrix<Vec, std::array<Vec, cols()>> result; // Static
            doMatrixTranspose(result);
            return result;
        }
    }

    Type trace() const {
        checkSizeSquare();
        Type sum = Type(0);
        for (size_t i = 0; i < rows(); ++i) {
            sum += mData[i][i];
        }
        return sum;
    }

    void fill(const Type value) {
        for (size_t i = 0; i < rows(); ++i) {
            mData[i].fill(value);
        }
    }

    template <typename U1, typename U2>
    bool epsilonEquals(const Matrix<Vector<Type, U1>, U2>& other, const Type eps) const {
        if (rows() != other.rows() || cols() != other.cols())
            return false;
        Type sum = Type(0);
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                sum += std::abs(mData[i][j] - other.mData[i][j]);
            }
        }
        return sum <= eps;
    }

    static auto zeros(const size_t n) {
        Matrix<Vector<Type>> result(n, n); // Dynamic
        return result;
    }

    template <size_t N>
    static auto zeros() {
        using Vec = SVector<Type, N>;
        Matrix<Vec, std::array<Vec, N>> result; // Static
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result.mData[i][j] = Type(0);
            }
        }
        return result;
    }

    static auto identity(const size_t n) {
        Matrix<Vector<Type>> result(n, n); // Dynamic
        for (size_t i = 0; i < n; ++i) {
            result.mData[i][i] = Type(1);
        }
        return result;
    }

    template <size_t N>
    static auto identity() {
        using Vec = SVector<Type, N>;
        Matrix<Vec, std::array<Vec, N>> result; // Static
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result.mData[i][j] = Type(i == j);
            }
        }
        return result;
    }

    friend Matrix operator*(const Type scalar, const Matrix& m) {
        return m * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "[";
        for (size_t i = 0; i < m.rows(); ++i) {
            if (i > 0)
                os << " ";
            os << "[";
            for (size_t j = 0; j < m.cols(); ++j) {
                os << m.mData[i][j];
                if (j + 1 < m.cols())
                    os << ", ";
            }
            os << "]";
            if (i + 1 < m.rows())
                os << "\n";
        }
        os << "]";
        return os;
    }

private:
    Matrix init() const {
        if constexpr (isDynamic) {
            return Matrix(rows(), cols());
        } else {
            return Matrix();
        }
    }

    TUnderlying init(const size_t rows, const size_t cols) const {
        if constexpr (isDynamic) {
            return TUnderlying(rows, TVector(cols));
        } else {
            if (this->rows() != rows || this->cols() != cols) {
                throw invalidArgument("Dimension mismatch");
            }
            return TUnderlying();
        }
    }

    template <typename U1, typename U2>
    void checkSizeEqual(const Matrix<Vector<Type, U1>, U2>& other) const {
        if constexpr (isDynamic || Matrix<Vector<Type, U1>, U2>::isDynamic) {
            if (rows() != other.rows() || cols() != other.cols()) {
                throw invalidArgument("Dimension mismatch");
            }
        } else {
            static_assert(rows() == other.rows() && cols() == other.cols(), "Dimension mismatch");
        }
    }

    void checkSizeNonZero() const {
        if constexpr (isDynamic) {
            if (rows() == 0 || cols() == 0) {
                throw invalidArgument("The Matrix must have non-zero size");
            }
        } else {
            static_assert(rows() > 0 && cols() > 0, "The Matrix must have non-zero size");
        }
    }

    void checkSizeSquare() const {
        if constexpr (isDynamic) {
            if (rows() != cols()) {
                throw invalidArgument("The Matrix must be square");
            }
        } else {
            static_assert(rows() == cols(), "The Matrix must be square");
        }
    }

    template <typename TMatrix1, typename TMatrix2>
    void doMatrixMult(const TMatrix1& other, TMatrix2& out) const {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < other.cols(); ++j) {
                Type sum = Type(0);
                for (size_t k = 0; k < cols(); ++k) {
                    sum += mData[i][k] * other.mData[k][j];
                }
                out.mData[i][j] = sum;
            }
        }
    }

    template <typename TVector1, typename TVector2>
    void doVectorMult(const TVector1& v, TVector2& out) const {
        for (size_t i = 0; i < rows(); ++i) {
            out[i] = mData[i].dot(v);
        }
    }

    template <typename TMatrix>
    void doMatrixTranspose(TMatrix& out) const {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                out.mData[j][i] = mData[i][j];
            }
        }
    }
};

template <typename T, size_t ROWS, size_t COLS>
using SMatrix = Matrix<SVector<T, COLS>, std::array<SVector<T, COLS>, ROWS>>;

} // namespace OptiSik
