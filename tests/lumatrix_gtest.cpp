#include "data/lumatrix.h"
#include "data/matrix.h"
#include <gtest/gtest.h>


using namespace OptiSik;

using Vec = Vector<double>;
using Mat = Matrix<Vec>;

TEST(LUMatrixTest, SolveSimpleSystem) {
    Mat A(std::vector<Vec>{ Vec({ 3, 1 }), Vec({ 1, 2 }) });
    Vec b({ 5, 5 });
    LUMatrix lu(A);
    auto x = lu.solve(b);
    EXPECT_NEAR(x[0], 1.0, 1e-9);
    EXPECT_NEAR(x[1], 2.0, 1e-9);
}

TEST(LUMatrixTest, DeterminantAndInverse) {
    Mat A(std::vector<Vec>{ Vec({ 4, 7 }), Vec({ 2, 6 }) });
    LUMatrix lu(A);
    // Determinant should be 10
    EXPECT_NEAR(lu.determinant(), 10.0, 1e-9);
    // Inverse reconstruction
    auto inv  = lu.invert();
    auto prod = A * inv;
    auto id   = Mat::identity(2);
    EXPECT_TRUE(prod.epsilonEquals(id, 1e-8));
}

TEST(LUMatrixTest, SingularThrows) {
    Mat A(std::vector<Vec>{ Vec({ 1, 2 }), Vec({ 2, 4 }) });
    EXPECT_THROW(LUMatrix lu(A), computationError);
}

TEST(LUMatrixTest, StaticSolveSimpleSystem) {
    SMatrix<double, 2, 2> A = Mat(std::vector<Vec>{ Vec({ 3, 1 }), Vec({ 1, 2 }) });
    SVector<double, 2> b = Vec({ 5, 5 });
    LUMatrix lu(A);
    auto x = lu.solve(b);
    EXPECT_NEAR(x[0], 1.0, 1e-9);
    EXPECT_NEAR(x[1], 2.0, 1e-9);
}

TEST(LUMatrixTest, StaticDeterminantAndInverse) {
    SMatrix<double, 2, 2> A = Mat(std::vector<Vec>{ Vec({ 4, 7 }), Vec({ 2, 6 }) });
    LUMatrix lu(A);
    // Determinant should be 10
    EXPECT_NEAR(lu.determinant(), 10.0, 1e-9);
    // Inverse reconstruction
    auto inv  = lu.invert();
    auto prod = A * inv;
    auto id   = SMatrix<double, 2, 2>::identity<2>();
    EXPECT_TRUE(prod.epsilonEquals(id, 1e-8));
}

TEST(LUMatrixTest, StaticSingularThrows) {
    SMatrix<double, 2, 2> A = Mat(std::vector<Vec>{ Vec({ 1, 2 }), Vec({ 2, 4 }) });
    EXPECT_THROW(LUMatrix lu(A), computationError);
}