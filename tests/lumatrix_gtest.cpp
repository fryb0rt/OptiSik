#include "data/lumatrix.h"
#include "data/matrix.h"
#include <gtest/gtest.h>


using namespace OptiSik;

TEST(LUMatrixTest, SolveSimpleSystem) {
    Matrix<double> A(std::vector<std::vector<double>>{ { 3, 1 }, { 1, 2 } });
    std::vector<double> b{ 5, 5 };
    LUMatrix<double> lu(A);
    auto x = lu.solve(b);
    EXPECT_NEAR(x[0], 1.0, 1e-9);
    EXPECT_NEAR(x[1], 2.0, 1e-9);
}

TEST(LUMatrixTest, DeterminantAndInverse) {
    Matrix<double> A(std::vector<std::vector<double>>{ { 4, 7 }, { 2, 6 } });
    LUMatrix<double> lu(A);
    // Determinant should be 10
    EXPECT_NEAR(lu.determinant(), 10.0, 1e-9);
    // Inverse reconstruction
    Matrix<double> inv  = lu.invert();
    Matrix<double> prod = A * inv;
    Matrix<double> id   = Matrix<double>::identity(2);
    EXPECT_TRUE(prod.epsilonEquals(id, 1e-8));
}

TEST(LUMatrixTest, SingularThrows) {
    Matrix<double> A(std::vector<std::vector<double>>{ { 1, 2 }, { 2, 4 } });
    EXPECT_THROW(LUMatrix<double> lu(A), computationError);
}
