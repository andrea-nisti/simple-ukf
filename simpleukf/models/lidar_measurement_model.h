#ifndef SIMPLEUKF_MODELS_LIDAR_MEASUREMENT_MODEL_H
#define SIMPLEUKF_MODELS_LIDAR_MEASUREMENT_MODEL_H

#include <Eigen/Dense>

namespace simpleukf::models
{

struct LidarNoiseConstantDefault
{
    // Laser measurement noise standard deviation position1 in m
    static constexpr double std_laspx = 0.15;

    // Laser measurement noise standard deviation position2 in m
    static constexpr double std_laspy = 0.15;
};

template <typename NoiseConstants = LidarNoiseConstantDefault>
struct LidarModel
{
  private:
    using CTRVModelInt = CTRVModel<>;

  public:
    static constexpr double std_laspx = LidarNoiseConstantDefault::std_laspx;
    static constexpr double std_laspy = LidarNoiseConstantDefault::std_laspy;

    static constexpr int n = 2;
    using PredictedVector = Eigen::Vector<double, n>;
    using MeasurementVector = PredictedVector;

    // clang-format off
    using PredictedCovMatrix = Eigen::Matrix<double, n, n>;
    inline static const PredictedCovMatrix noise_matrix_squared =
        (PredictedCovMatrix() << 
         std_laspx * std_laspx,   0,
         0,    std_laspy * std_laspy).finished();
    // clang-format on
    using MeasurementCovMatrix = PredictedCovMatrix;

    PredictedVector Predict(const CTRVModelInt::PredictedVector& curr_state) const
    {
        PredictedVector ret{curr_state(0), curr_state(1)};
        return ret;
    }
};

}  // namespace simpleukf::models

#endif  // SIMPLEUKF_MODELS_LIDAR_MEASUREMENT_MODEL_H
