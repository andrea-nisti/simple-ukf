#ifndef SIMPLEUKF_EXAMPLES_FUSION_H
#define SIMPLEUKF_EXAMPLES_FUSION_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include "simpleukf_tools.h"
namespace fusion
{
class Fusion
{
  public:
    /**
     * Constructor
     */
    Fusion();
    void PrintStates()
    {
        std::cout << "Mean\n" << x_ << "\n";
        std::cout << "Cova\n" << P_ << std::endl;
    }

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

    // state covariance matrix
    Eigen::MatrixXd P_;

  private:
    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);

    // Main fusion logic based on the submodule src/simpleukf
    CTRVFusionFilter ctrv_ukf_filter_{};

    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_{false};

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // time when the state is true, in us
    long long time_us_;
};

}  // namespace fusion

#endif // SIMPLEUKF_EXAMPLES_FUSION_H
