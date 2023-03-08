#include <gtest/gtest.h>

#include "simpleukf/models/crtv_models.h"
#include "simpleukf/ukf/linear_update_strategy.h"
#include "test_utils.h"

namespace simpleukf::testing
{

struct MeasureMock
{
    using PredictedVector = Eigen::Vector<double, 2>;
    static constexpr int n = 2;

    PredictedVector Predict(const simpleukf::models::CRTVModel<>::PredictedVector& curr_state) const
    {
        PredictedVector ret{curr_state(0), curr_state(1)};
        return ret;
    }

    inline static const Eigen::Matrix<double, n, n> noise_matrix_squared =
        (Eigen::Matrix<double, n, n>() << 0.1f, 0, 0, 0.1f).finished();
};

TEST(LinearUpdateStrategy, GivenParameters_ExpectUpdate)
{
    using namespace simpleukf::models;

    Eigen::Vector2d measure;
    measure << 1.0f, 0.5f;
    // clang-format off
    Eigen::MatrixXd H(2, 5);
    H << 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0;
    // clang-format on

    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> current_state;
    current_state.mean = {0.5f, 0.3f, 0.0f, 0.0f, 0.0f};
    current_state.covariance = simpleukf::models::CRTVModel<>::StateCovMatrix::Identity();

    auto strategy = ukf::LinearUpdateStrategy<CRTVModel<>, MeasureMock>{H};
    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> new_state;

    strategy.Update(measure, current_state, new_state);
    CRTVModel<>::PredictedVector expected_state{0.954545, 0.481818, 0, 0, 0};

    ExpectNearMatrixd(expected_state, new_state.mean, 0.0001f);
}

}  // namespace simpleukf::testing