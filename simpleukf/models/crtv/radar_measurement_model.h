#ifndef SIMPLEUKF_MODELS_RADAR_MEASUREMENT_MODEL_H
#define SIMPLEUKF_MODELS_RADAR_MEASUREMENT_MODEL_H

#include "crtv_model.h"

#include <Eigen/Dense>

namespace simpleukf::models
{

// NoiseConstants example:
struct NoiseConstantDefault
{
    static constexpr double std_radr = 0.3;
    static constexpr double std_radphi = 0.0175;
    static constexpr double std_radrd = 0.1;
};

template <typename NoiseConstants = NoiseConstantDefault>
class RadarModel
{
  public:
    static constexpr int n = 3;

    // radar measurement noise standard deviation radius in m
    static constexpr double std_radr = NoiseConstants::std_radr;  // 0.3;

    // radar measurement noise standard deviation angle in rad
    static constexpr double std_radphi = NoiseConstants::std_radphi;  // 0.0175;

    // radar measurement noise standard deviation radius change in m/s
    static constexpr double std_radrd = NoiseConstants::std_radrd;  // 0.1;

    // clang-format off
    using MeasurementCovMatrix = Eigen::Matrix<double, n, n>;
    inline static const MeasurementCovMatrix measurement_cov_matrix =
        (MeasurementCovMatrix() << 
         std_radr * std_radr,        0, 0,
         0,    std_radphi * std_radphi, 0, 
         0,    0,    std_radrd * std_radrd).finished();
    // clang-format on

    using MeasurementVector = Eigen::Vector<double, n>;
    using PredictedSigmaMatrix = Eigen::Matrix<double, n, CRTVModel::n_sigma_points>;

    template <typename StateVector>
    static MeasurementVector Predict(const StateVector& current_state)
    {
        static_assert(StateVector::RowsAtCompileTime >= CRTVModel::n,
                      "The radar model works for CRTV normal or augmented state");

        // extract values for better readability
        double p_x = current_state(0);
        double p_y = current_state(1);
        double v = current_state(2);
        double yaw = current_state(3);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        MeasurementVector return_state;
        return_state(0) = sqrt(p_x * p_x + p_y * p_y);                          // r
        return_state(1) = atan2(p_y, p_x);                                      // phi
        return_state(2) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);  // r_dot

        return return_state;
    }

    static void AdjustMeasure(MeasurementVector& to_be_adjusted)
    {
        // angle normalization
        while (to_be_adjusted(1) > M_PI)
            to_be_adjusted(1) -= 2. * M_PI;
        while (to_be_adjusted(1) < -M_PI)
            to_be_adjusted(1) += 2. * M_PI;
    }
};

}  // namespace simpleukf::models

#endif  // SIMPLEUKF_MODELS_RADAR_MEASUREMENT_MODEL_H
