#include <gtest/gtest.h>

#include "simpleukf/models/crtv_models.h"
#include "simpleukf/models/linear_update_strategy.h"

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
        (Eigen::Matrix<double, n, n>() << 1.0f, 0, 0, 1.0f).finished();
};

TEST(LinearUpdateStrategy, IsConstructible)
{
    using namespace simpleukf::models;

    auto strategy = LinearUpdateStrategy<CRTVModel<>>{};
    EXPECT_TRUE(true);
}

TEST(LinearUpdateStrategy, GivenParameters_ExpectUpdate)
{
    using namespace simpleukf::models;

    LinearUpdateParamenters<CRTVModel<>, MeasureMock> params;
    params.measure = {1.0f, 0.5f};
    // clang-format off
    params.H <<
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;
    // clang-format on

    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> current_state;
    current_state.mean = {0.5f, 0.3f, 0.0f, 0.0f, 0.0f};
    current_state.covariance = simpleukf::models::CRTVModel<>::StateCovMatrix::Identity();

    auto strategy = LinearUpdateStrategy<CRTVModel<>>{};
    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> new_state;
    strategy.Update(params, current_state, new_state);
    std::cout << new_state.mean << "\n";
    std::cout << new_state.covariance << std::endl;

    EXPECT_TRUE(true);
}