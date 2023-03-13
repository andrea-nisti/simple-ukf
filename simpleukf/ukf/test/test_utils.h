#ifndef SIMPLEUKF_UKF_TEST_TEST_UTILS_H
#define SIMPLEUKF_UKF_TEST_TEST_UTILS_H

#include <cassert>

#include <gtest/gtest.h>

namespace simpleukf::testing
{

template <typename MatrixA, typename MatrixB>
void ExpectNearMatrixd(const MatrixA& matrix_a, const MatrixB& matrix_b, const double delta)
{
    if ((matrix_a.cols() != matrix_b.cols()) or (matrix_a.rows() != matrix_b.rows()))
    {
        ASSERT_TRUE(false);
    }

    for (int index = 0; index < matrix_a.cols(); ++index)
    {
        const auto matrix_b_element{matrix_b(index)};
        const auto matrix_a_element{matrix_a(index)};

        EXPECT_NEAR(matrix_b_element, matrix_a_element, delta);
    }
}

}  // namespace simpleukf::testing

#endif  // SIMPLEUKF_UKF_TEST_TEST_UTILS_H