#include <gtest/gtest.h>

#include "simpleukf/models/ctrv_models.h"
#include "simpleukf/ukf/ukf.h"
#include "simpleukf/ukf/unscented_update_strategy.h"
#include "test_utils.h"

namespace simpleukf::testing
{

TEST(UKFComponentTests, OnGivenInitialConditions_ExpectCorrectEstimation)
{
    using namespace simpleukf::ukf;
    using namespace simpleukf::models;

    using RadarModel = RadarModel<>;
    using CTRVModel = CTRVModel<>;

    // set example state
    UKF<CTRVModel> instance{};

    // set initial state
    const auto x = CTRVModel::StateVector{5.7441, 1.3800, 2.2049, 0.5015, 0.3528};
    // clang-format off
    const auto P = (CTRVModel::StateCovMatrix{}
      << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, 
        -0.0013,  0.0077, 0.0011,  0.0071,  0.0060, 
         0.0030,  0.0011, 0.0054,  0.0007,  0.0008, 
        -0.0022,  0.0071, 0.0007,  0.0098,  0.0100, 
        -0.0020,  0.0060, 0.0008,  0.0100,  0.0123).finished();
        
    instance.Init(x, P);
    instance.PredictProcessMeanAndCovariance(0.1f);
    
    // create example vector for mean predicted measurement
    RadarModel::MeasurementVector z{};
    z << 5.9214,  // rho in m
        0.2187,   // phi in rad
        2.0062;   // rho_dot in m/s
    // clang-format on

    auto strategy = ukf::UnscentedUpdateStrategy<CTRVModel, RadarModel>{instance.GetCurrentPredictedSigmaMatrix()};
    instance.UpdateState<RadarModel>(z, strategy);

    // then
    auto expected = CTRVModel::PredictedVector{5.92115, 1.41666, 2.15551, 0.48931, 0.31995};
    const auto current_prediction = instance.GetCurrentStateVector();

    ExpectNearMatrixd(expected, current_prediction, 0.00001);
}

}  // namespace simpleukf::testing