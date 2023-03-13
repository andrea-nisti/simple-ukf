#include "fusion.h"

#include <iostream>

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fusion
{

/**
 * Initializes Unscented Kalman filter
 */
Fusion::Fusion()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // time when the state is true, in us
    time_us_ = 0.0;

    // PLEASE Check src/simpleukf_tools.h for noise constants.
    // The main logic is encapsulated in the submodule src/simpleukf [https://github.com/andrea-nisti/simple-ukf]
}

void Fusion::ProcessMeasurement(MeasurementPackage meas_package)
{
    if (not is_initialized_)
    {
        // Initialize the filter with a laser measure
        CTRVFusionFilter::StateVector_t predicted_state;
        predicted_state.fill(0.0f);
        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            predicted_state.head(2) = meas_package.raw_measurements_;
            ctrv_ukf_filter_.Init(predicted_state, CTRVFusionFilter::StateCovMatrix_t::Identity());
            // Prediction(0.0);
            UpdateLidar(meas_package);
            is_initialized_ = true;
        }
    }
    else
    {
        const double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
        Prediction(delta_t);

        // RUN this logic if skipping predictions when observations with same timestamp arrives
        // unfortunately, for some reason it makes the rmse test fail
        // if (delta_t > 0.0)
        // {
        //     /* code */
        //     Prediction(delta_t);
        // }

        // fuse correct measurement
        if ((meas_package.sensor_type_ == MeasurementPackage::LASER) and use_laser_)
        {
            UpdateLidar(meas_package);
        }
        else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) and use_radar_)
        {
            UpdateRadar(meas_package);
        }
        else
        {
            std::cout << "Received wrong sensor type, skipping package" << std::endl;
        }
    }

    // update last fusion timestamp and states
    x_ = ctrv_ukf_filter_.GetCurrentStateVector();
    P_ = ctrv_ukf_filter_.GetCurrentCovarianceMatrix();
    time_us_ = meas_package.timestamp_;
}

void Fusion::Prediction(double delta_t)
{
    ctrv_ukf_filter_.PredictProcessMeanAndCovariance(delta_t);

    x_ = ctrv_ukf_filter_.GetCurrentStateVector();
    P_ = ctrv_ukf_filter_.GetCurrentCovarianceMatrix();
}

void Fusion::UpdateLidar(MeasurementPackage meas_package)
{
    // clang-format off
    static const auto H = ( Eigen::Matrix<double, 2, 5>() << 
                                                        1, 0, 0, 0, 0,
                                                        0, 1, 0, 0, 0).finished();
    // clang-format on
    LidarMeasurementModel::MeasurementVector measure;
    measure.head(LidarMeasurementModel::n) = meas_package.raw_measurements_;

    auto strategy = simpleukf::ukf::LinearUpdateStrategy<CTRVProcessModel, LidarMeasurementModel>{H};
    ctrv_ukf_filter_.UpdateState<LidarMeasurementModel>(meas_package.raw_measurements_, strategy);
}

void Fusion::UpdateRadar(MeasurementPackage meas_package)
{
    RadarMeasurementModel::MeasurementVector measure;
    measure.head(RadarMeasurementModel::n) = meas_package.raw_measurements_;

    auto strategy = simpleukf::ukf::UnscentedUpdateStrategy<CTRVProcessModel, RadarMeasurementModel>{
        ctrv_ukf_filter_.GetCurrentPredictedSigmaMatrix()};
    ctrv_ukf_filter_.UpdateState<RadarMeasurementModel>(meas_package.raw_measurements_, strategy);
}

}  // namespace fusion