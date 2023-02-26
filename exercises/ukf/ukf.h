#ifndef EXERCISES_UKF_UKF_H
#define EXERCISES_UKF_UKF_H

#include <iostream>

#include "exercises/models/models.h"
#include "ukf.h"
#include "ukf_utils.h"

#include <Eigen/Dense>

/* TODO:
- fix naming convention

impovement
- cache last fusion timestamp
*/

template <typename ProcessModel>
class UKF
{
  public:
    UKF() {}
    virtual ~UKF() {}

    using StateVector_t = typename ProcessModel::StateVector;
    using StateVectorAugmented_t = typename ProcessModel::StateVectorAugmented;
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

    void PredictMeanAndCovariance(const double delta_t)
    {
        using SigmaMatrixAugmented = typename ProcessModel::SigmaMatrixAugmented;
        using ProcessNoiseVector = Eigen::Vector<double, ProcessModel::n_process_noise>;

        SigmaMatrixAugmented augmented_sigma_points =
            AugmentedSigmaPoints(current_state_, current_cov_, ProcessNoiseVector{{0.2f, 0.2f}});

        // Predict sigma points
        current_predicted_sigma_points_ = SigmaPointPrediction<ProcessModel>(augmented_sigma_points, delta_t);

        const auto weights = GenerateWeights<ProcessModel::n_aug>(lambda_);

        current_state_ = ComputeMeanFromSigmaPoints(weights, current_predicted_sigma_points_);
        current_cov_ = ComputeCovarianceFromSigmaPoints(
            weights, current_predicted_sigma_points_, current_state_, &ProcessModel::AdjustState);
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
        const auto weights = GenerateWeights<ProcessModel::n_aug>(lambda_);

        // mean predicted measurement
        PredictedMeasurementSigmaPoints_t predicted_measurement_sigma_points =
            SigmaPointPrediction<MeasurementModel>(current_predicted_sigma_points_);

        // mean predicted measurement
        MeasurementVector_t predicted_measurement =
            ComputeMeanFromSigmaPoints(weights, predicted_measurement_sigma_points);

        typename MeasurementModel::MeasurementCovMatrix S = ComputeCovarianceFromSigmaPoints(
            weights, predicted_measurement_sigma_points, predicted_measurement, &MeasurementModel::AdjustMeasure);

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
        using PredictedMeasurementSigmaPoints_t =
            typename Eigen::Matrix<double, MeasurementModel::n, PredictedSigmaMatrix_t::ColsAtCompileTime>;

        // create matrix for cross correlation Tc, predicted measurement measure_out and pred covariance S_out
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> cross_correlation_matrix{};
        cross_correlation_matrix.fill(0.0f);

        RadarModel::MeasurementVector measure_pred{};
        RadarModel::MeasurementCovMatrix S{};

        const PredictedMeasurementSigmaPoints_t predicted_measurement_sigma_points =
            PredictMeasurement<MeasurementModel>(measure_pred, S);

        // calculate cross correlation matrix
        // this computation gives the correlation between real measure and predicted
        const auto weights = GenerateWeights<ProcessModel::n_aug>(lambda_);
        for (int i = 0; i < ProcessModel::n_sigma_points; ++i)
        {
            MeasurementVector_t measure_diff = predicted_measurement_sigma_points.col(i) - measure_pred;
            MeasurementModel::AdjustMeasure(measure_diff);

            StateVector_t x_diff = current_predicted_sigma_points_.col(i) - current_state_;
            ProcessModel::AdjustState(x_diff);

            cross_correlation_matrix = cross_correlation_matrix + weights(i) * x_diff * measure_diff.transpose();
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
    static constexpr double lambda_ = 3 - ProcessModel::n_aug;
    StateVector_t current_state_{};
    StateCovMatrix_t current_cov_{};
    PredictedSigmaMatrix_t current_predicted_sigma_points_{};
};

#endif  // EXERCISES_UKF_UKF_H
