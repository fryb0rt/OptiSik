#pragma once
#include "data/matrix.h"
#include "utils/exception.h"
#include "utils/mathUtils.h"
#include <type_traits>

namespace OptiSik {

namespace {

template<typename TMatrix, bool isDynamic>
struct PermutationType {
    using Type = std::vector<size_t>;
};

template<typename TMatrix>
struct PermutationType<TMatrix, false> {
    using Type = std::array<size_t, matrixRows<TMatrix>>;
};

} // namespace

/// LU Decomposition with partial pivoting
template <typename TMatrix, typename TTolerance = Tolerance<typename TMatrix::Type>>
class LUMatrix {
    /// Contains the LU-decomposed matrix. Upper triangular part is stored
    /// completely, lower triangular part has implicit unit diagonal.
    TMatrix mLU;

    /// Permutation vector - represents identity matrix permuted according to
    /// row exchanges
    PermutationType<TMatrix, TMatrix::isDynamic>::Type mPermutations;

    /// Number of row exchanges performed during decomposition
    size_t mRowExchanges;

public:
    using T = typename TMatrix::Type;
    using TVector = typename TMatrix::RowType;

    /// Constructor that performs LU decomposition with partial pivoting
    LUMatrix(const TMatrix& A) : mLU(A), mRowExchanges(0) {
        if constexpr (TMatrix::isDynamic) {
            if (A.rows() != A.cols()) {
                throw invalidArgument(
                "LUMatrix: Matrix must be square for LU decomposition");
            }
        } else {
            static_assert(A.rows() == A.cols(), "LUMatrix: Matrix must be square for LU decomposition");
        }
        const size_t n = A.rows();
        if constexpr (TMatrix::isDynamic) {
            mPermutations.resize(n);
        }
        for (size_t i = 0; i < n; ++i) {
            mPermutations[i] = i;
        }
        for (size_t k = 0; k < n; ++k) {
            // Pivoting
            T maxVal      = std::abs(mLU(k, k));
            size_t maxRow = k;
            for (size_t i = k + 1; i < n; ++i) {
                if (std::abs(mLU(i, k)) > maxVal) {
                    maxVal = std::abs(mLU(i, k));
                    maxRow = i;
                }
            }
            if (maxVal < TTolerance::tolerance) {
                throw computationError(
                "LUMatrix: Matrix is singular or near-singular");
            }
            if (maxRow != k) {
                std::swap(mPermutations[k], mPermutations[maxRow]);
                for (size_t j = 0; j < n; ++j) {
                    std::swap(mLU(k, j), mLU(maxRow, j));
                }
                ++mRowExchanges;
            }
            // LU Decomposition
            for (size_t i = k + 1; i < n; ++i) {
                mLU(i, k) /= mLU(k, k);
                for (size_t j = k + 1; j < n; ++j) {
                    mLU(i, j) -= mLU(i, k) * mLU(k, j);
                }
            }
        }
    }

    /// Solves the system Ax = b using the LU decomposition
    TVector solve(const TVector& b) const {
        const size_t n = mLU.rows();
        if constexpr (TMatrix::isDynamic) {
            if (n != b.dimension()) {
                throw invalidArgument("LUMatrix::solve: Dimension mismatch "
                                      "between matrix and vector");
            }
        }

        TVector x = [n]() {
            if constexpr (TMatrix::isDynamic) {
                return TVector(n);
            } else {
                return TVector{};
            }
        }();
        // Apply permutations to b
        for (size_t i = 0; i < n; ++i) {
            x[i] = b[mPermutations[i]];
            // Forward substitution to solve Ly = Pb
            for (size_t j = 0; j < i; ++j) {
                x[i] -= mLU(i, j) * x[j];
            }
        }

        // Backward substitution to solve Ux = y
        for (size_t i1 = n; i1 > 0; --i1) {
            size_t i = i1 - 1; // Avoid underflow
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= mLU(i, j) * x[j];
            }
            x[i] /= mLU(i, i);
        }
        return x;
    }

    /// Computes the determinant of the original matrix
    T determinant() const {
        T det(1);
        for (size_t i = 0; i < mLU.rows(); ++i) {
            det *= mLU(i, i);
        }
        return mRowExchanges % 2 == 0 ? det : -det;
    }

    /// Computes the inverse of the original matrix
    TMatrix invert() const {
        const size_t n  = mLU.rows();
        TMatrix inverse = [n]() {
            if constexpr (TMatrix::isDynamic) {
                return TMatrix(n, n);
            } else {
                return TMatrix();
            }
        }();
        for (size_t i = 0; i < n; ++i) {
            TVector e = [n]() {
                if constexpr (TVector::isDynamic) {
                    return TVector(n);
                } else {
                    TVector e;
                    for (int j = 0; j < n; ++j) {
                        e[j] = T(0);
                    }
                    return e;
                }
            }();
            e[i]           = T(1);
            const auto col = solve(e);
            for (size_t j = 0; j < n; ++j) {
                inverse(j, i) = col[j];
            }
        }
        return inverse;
    }

    /// Returns the LU matrix
    const TMatrix& getLU() const {
        return mLU;
    }

    /// Returns the permutation vector
    const auto& getPermutations() const {
        return mPermutations;
    }

    /// Returns the number of row exchanges
    size_t getRowExchanges() const {
        return mRowExchanges;
    }
};
} // namespace OptiSik
