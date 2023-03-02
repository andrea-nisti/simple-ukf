#include <gtest/gtest.h>

#include "simpleukf/models/crtv_models.h"
#include "simpleukf/ukf/ukf.h"

TEST(SimpleTest, AssertTrue)
{
    EXPECT_TRUE(true);
}

TEST(UKFComponentTests, OnGivenInitialConditions_ExpectCorrectEstimation)
{
    using namespace simpleukf::ukf;
    using namespace simpleukf::models;

    using RadarModel = RadarModel<>;
    using CRTVModel = CRTVModel<>;

    // set example state
    auto x = CRTVModel::StateVector{5.7441, 1.3800, 2.2049, 0.5015, 0.3528};

    // set example covariance matrix
    // clang-format off
    auto P = CRTVModel::StateCovMatrix{};
    P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, 
        -0.0013,  0.0077, 0.0011,  0.0071,  0.0060, 
         0.0030,  0.0011, 0.0054,  0.0007,  0.0008, 
        -0.0022,  0.0071, 0.0007,  0.0098,  0.0100, 
        -0.0020,  0.0060, 0.0008,  0.0100,  0.0123;
    
    // create example vector for mean predicted measurement
    RadarModel::MeasurementVector z{};
    z << 5.9214,  // rho in m
        0.2187,   // phi in rad
        2.0062;   // rho_dot in m/s
    // clang-format on

    UKF<CRTVModel> instance{};
    instance.Init(x, P);

    instance.PredictProcessMeanAndCovariance(0.1f);
    instance.UpdateState<RadarModel>(z);

    // then
    auto expected = CRTVModel::PredictedVector{ 5.92115, 1.41666, 2.15551, 0.48931, 0.31995};
    const auto current_prediction = instance.GetCurrentStateVector();
    for (int index = 0; index < expected.ColsAtCompileTime; ++index)
    {
        const auto current_element{current_prediction(index)};
        const auto expected_element{expected(index)};

        EXPECT_NEAR(current_element, expected_element, 0.00001f);
    }
}