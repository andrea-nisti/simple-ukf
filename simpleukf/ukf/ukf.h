#ifndef SIMPLEUKF_UKF_UKF_H
#define SIMPLEUKF_UKF_UKF_H

#include <iostream>

#include "ukf.h"
#include "ukf_utils.h"

#include <Eigen/Dense>

/* TODO:
- fix naming convention

impovement
- cache last fusion timestamp
*/
namespace simpleukf::ukf
{

template <typename ProcessModel>
class UKF
{

  public:
    UKF() {}
    virtual ~UKF() {}

    using StateVector_t = typename ProcessModel::StateVector;
    using StateCovMatrix_t = typename ProcessModel::StateCovMatrix;
    using PredictedSigmaMatrix_t = typename ProcessModel::PredictedSigmaMatrix;

    /**
     * Init Initializes Unscented Kalman filter
     */
    void Init(const StateVector_t& init_state, const StateCovMatrix_t& init_cov_matrix)
    {
        // set example state
        current_state_ = init_state;
        current_cov_ = init_cov_matrix;
    }

    // TODO: extract a common function to be used withing the measurement prediction too.
    template <typename... PredictionArgs>
    void PredictProcessMeanAndCovariance(PredictionArgs&&... args)
    {

        using SigmaMatrixAugmented = typename ProcessModel::SigmaMatrixAugmented;

        SigmaMatrixAugmented augmented_sigma_points =
            ukf_utils::AugmentedSigmaPoints(current_state_, current_cov_, lambda_, ProcessModel::noise_matrix_squared);

        // Predict sigma points
        current_predicted_sigma_points_ = ukf_utils::SigmaPointPrediction<ProcessModel>(
            augmented_sigma_points, std::forward<const PredictionArgs>(args)...);

        current_state_ = ukf_utils::ComputeMeanFromSigmaPoints(weights_, current_predicted_sigma_points_);
        current_cov_ = ukf_utils::ComputeCovarianceFromSigmaPoints(
            weights_, current_predicted_sigma_points_, current_state_, &ProcessModel::AdjustState);
    }

  private:
    template <typename MeasurementModel>
    typename MeasurementModel::PredictedSigmaMatrix PredictMeasurement(
        typename MeasurementModel::MeasurementVector& measure_out,
        typename MeasurementModel::MeasurementCovMatrix& S_out)
    {
        using MeasurementVector_t = typename MeasurementModel::MeasurementVector;
        using PredictedMeasurementSigmaPoints_t = typename MeasurementModel::PredictedSigmaMatrix;

        // define spreading parameter and generate weights

        // mean predicted measurement
        PredictedMeasurementSigmaPoints_t predicted_measurement_sigma_points =
            ukf_utils::SigmaPointPrediction<MeasurementModel>(current_predicted_sigma_points_);

        // mean predicted measurement
        MeasurementVector_t predicted_measurement =
            ukf_utils::ComputeMeanFromSigmaPoints(weights_, predicted_measurement_sigma_points);

        typename MeasurementModel::MeasurementCovMatrix S = ukf_utils::ComputeCovarianceFromSigmaPoints(
            weights_, predicted_measurement_sigma_points, predicted_measurement, &MeasurementModel::AdjustMeasure);

        // add measurement noise covariance matrix
        S = S + MeasurementModel::measurement_cov_matrix;

        // write result
        measure_out = predicted_measurement;
        S_out = S;

        return predicted_measurement_sigma_points;
    }

  public:
    template <typename MeasurementModel>
    void UpdateState(typename MeasurementModel::MeasurementVector measure,
                     typename ProcessModel::StateVector& x_out,
                     typename ProcessModel::StateCovMatrix& P_out)
    {
        using MeasurementVector_t = typename MeasurementModel::MeasurementVector;
        using MeasurementCovarianceMatrix_t = typename MeasurementModel::MeasurementCovMatrix;

        using PredictedMeasurementSigmaPoints_t =
            typename Eigen::Matrix<double, MeasurementModel::n, PredictedSigmaMatrix_t::ColsAtCompileTime>;

        // create matrix for cross correlation Tc, predicted measurement measure_out and pred covariance S_out
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> cross_correlation_matrix{};
        cross_correlation_matrix.fill(0.0f);

        MeasurementVector_t measure_pred{};
        MeasurementCovarianceMatrix_t S{};

        const PredictedMeasurementSigmaPoints_t predicted_measurement_sigma_points =
            PredictMeasurement<MeasurementModel>(measure_pred, S);

        // calculate cross correlation matrix
        // this computation gives the correlation between real measure and predicted
        for (int i = 0; i < ProcessModel::n_sigma_points; ++i)
        {
            MeasurementVector_t measure_diff = predicted_measurement_sigma_points.col(i) - measure_pred;
            MeasurementModel::AdjustMeasure(measure_diff);

            StateVector_t x_diff = current_predicted_sigma_points_.col(i) - current_state_;
            ProcessModel::AdjustState(x_diff);

            cross_correlation_matrix = cross_correlation_matrix + weights_(i) * x_diff * measure_diff.transpose();
        }

        // Kalman gain K;
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> K = cross_correlation_matrix * S.inverse();

        // residual
        MeasurementVector_t measure_diff = measure - measure_pred;

        // angle normalization
        MeasurementModel::AdjustMeasure(measure_diff);

        // update state mean and covariance matrix
        current_state_ = current_state_ + K * measure_diff;
        current_cov_ = current_cov_ - K * S * K.transpose();
        x_out = current_state_;
        P_out = current_cov_;
    }

    const PredictedSigmaMatrix_t& GetCurrentPredictedSigmaMatrix() const { return current_predicted_sigma_points_; }
    const StateVector_t& GetCurrentStateVector() const { return current_state_; }
    const StateCovMatrix_t& GetCurrentCovarianceMatrix() const { return current_cov_; }

  private:
    static constexpr double lambda_ = ProcessModel::GetLambda();
    const Eigen::Vector<double, ProcessModel::n_sigma_points> weights_ = ProcessModel::GenerateWeights();

    StateVector_t current_state_{};
    StateCovMatrix_t current_cov_{};
    PredictedSigmaMatrix_t current_predicted_sigma_points_{};
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_UKF_H
