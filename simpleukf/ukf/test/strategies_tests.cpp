#include <gtest/gtest.h>

#include "simpleukf/models/crtv_models.h"
#include "simpleukf/ukf/linear_update_strategy.h"
#include "simpleukf/ukf/unscented_update_strategy.h"
#include "test_utils.h"

namespace simpleukf::testing
{

struct MeasureMock
{
    static constexpr int n = 2;
    using PredictedVector = Eigen::Vector<double, n>;

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

TEST(UnscentedUpdateStrategy, GivenParameters_ExpectUpdate)
{
    using namespace simpleukf::models;

    // clang-format off
    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> current_state;
    current_state.mean << 5.93446, 1.48886, 2.2049, 0.53678, 0.3528;

    current_state.covariance <<  
        0.00548035,   -0.002499,  0.00340508, -0.00357408,  -0.0030908,
        -0.002499,   0.0110543,  0.00151778,  0.00990746,  0.00806631,
        0.00340508,  0.00151778,      0.0058,     0.00078,      0.0008,
        -0.00357408,  0.00990746,     0.00078,    0.011924,     0.01125,
        -0.0030908,  0.00806631,      0.0008,     0.01125,      0.0127;


    ukf_utils::PredictedSigmaMatrix<CRTVModel<>, CRTVModel<>::n_sigma_points> predicted_sigma_matrix;
    predicted_sigma_matrix <<
         5.93553,   6.0625,  5.92217,   5.9415,  5.92361,  5.93516,  5.93705,  5.93553,  5.80833,  5.94481,  5.92935,  5.94553,  5.93589,  5.93401,  5.93553,
         1.48939,  1.44673,  1.66483,  1.49719,    1.508,  1.49001,  1.49022,  1.48939,   1.5308,  1.31288,  1.48182,  1.46967,  1.48876,  1.48855,  1.48939,
          2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,  2.17026,   2.2049,
         0.53678, 0.473388, 0.678099, 0.554557, 0.643644, 0.543372,  0.53678, 0.538512, 0.600172, 0.395461, 0.519003, 0.429916, 0.530188,  0.53678, 0.535048,
          0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528, 0.31815;
    
    RadarModel<>::PredictedVector measure = {
        5.9214,  // rho in m
        0.2187,   // phi in rad
        2.0062   // rho_dot in m/s};
    };

    CRTVModel<>::PredictedVector expected_state = {
        5.92115,
        1.41666,
        2.15551,
        0.48931,
        0.31995
    };
    // clang-format on
    auto weights = CRTVModel<>::GenerateWeights();
    auto strategy = ukf::UnscentedUpdateStrategy<CRTVModel<>, RadarModel<>>{predicted_sigma_matrix, weights};
    simpleukf::ukf_utils::MeanAndCovariance<CRTVModel<>> new_state;

    strategy.Update(measure, current_state, new_state);
    ExpectNearMatrixd(expected_state, new_state.mean, 0.0001f);
}

}  // namespace simpleukf::testing