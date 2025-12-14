#pragma once
#include "data/matrix.h"
#include "utils/exception.h"
#include "utils/mathUtils.h"

namespace OptiSik {

/// LU Decomposition with partial pivoting
template <typename T, typename TMatrix = Matrix<T>, typename TTolerance = Tolerance<T>> class LUMatrix {
  /// Contains the LU-decomposed matrix. Upper triangular part is stored completely,
  /// lower triangular part has implicit unit diagonal.
  TMatrix mLU; 

  /// Permutation vector - represents identity matrix permuted according to row exchanges
  std::vector<size_t> mPermutations;

  /// Number of row exchanges performed during decomposition
  size_t mRowExchanges;
public:
  /// Constructor that performs LU decomposition with partial pivoting
  LUMatrix(const TMatrix &A) : mLU(A), mPermutations(A.rows()), mRowExchanges(0) {
    assert(A.rows() == A.cols());
    const size_t n = A.rows();
    for (size_t i = 0; i < n; ++i) {
      mPermutations[i] = i;
    }
    for (size_t k = 0; k < n; ++k) {
      // Pivoting
      T maxVal = std::abs(mLU(k, k));
      size_t maxRow = k;
      for (size_t i = k + 1; i < n; ++i) {
        if (std::abs(mLU(i, k)) > maxVal) {
          maxVal = std::abs(mLU(i, k));
          maxRow = i;
        }
      }
      if (maxVal < TTolerance::tolerance) {
        throw computationError("LUMatrix: Matrix is singular or near-singular");
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
    std::vector<T> solve(const std::vector<T> &b) const {
        const size_t n = mLU.rows();
        assert(n == b.size());
        
        std::vector<T> x(n);
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
};
} // namespace OptiSik
