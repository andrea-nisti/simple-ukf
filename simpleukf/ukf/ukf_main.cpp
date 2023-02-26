#include <iostream>

#include "simpleukf/models/crtv_models.h"
#include "ukf.h"

int main()
{
    using namespace simpleukf::ukf;
    using namespace simpleukf::models;

    using RadarModel = RadarModel<NoiseConstantDefault>;

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
    // clang-format on

    UKF<CRTVModel> instance{};
    instance.Init(x, P);

    CRTVModel::StateVector x_out{};
    CRTVModel::StateCovMatrix P_out{};
    instance.PredictMeanAndCovariance(0.1f);

    std::cout << "Predicted state" << std::endl;
    std::cout << instance.GetCurrentStateVector() << std::endl;
    std::cout << "Predicted covariance matrix" << std::endl;
    std::cout << instance.GetCurrentCovarianceMatrix() << std::endl;
    std::cout << "Predicted sigma matrix" << std::endl;

    std::cout << instance.GetCurrentPredictedSigmaMatrix() << std::endl;

    // create example vector for mean predicted measurement
    RadarModel::MeasurementVector z{};
    z << 5.9214,  // rho in m
        0.2187,   // phi in rad
        2.0062;   // rho_dot in m/s

    instance.UpdateState<RadarModel>(z, x_out, P_out);
    std::cout << "Updated state" << std::endl;
    std::cout << x_out << std::endl;
    std::cout << "Updated covariance matrix" << std::endl;
    std::cout << P_out << std::endl;

    return 0;
}