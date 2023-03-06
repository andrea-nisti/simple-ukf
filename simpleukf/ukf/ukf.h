#ifndef SIMPLEUKF_UKF_UKF_H
#define SIMPLEUKF_UKF_UKF_H

#include <iostream>

#include "ukf.h"
#include "ukf_utils.h"

#include <Eigen/Dense>

/* TODO:
- make prediction independent from augmentation

impovement
- cache last fusion timestamp
*/
namespace simpleukf::ukf
{

template <typename ProcessModel>
class UKF
{

  public:
    using StateVector_t = typename ProcessModel::StateVector;
    using StateCovMatrix_t = typename ProcessModel::StateCovMatrix;
    using PredictedProcessSigmaMatrix_t = ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points>;

    /**
     * Init Initializes Unscented Kalman filter
     */
    void Init(const StateVector_t& init_state, const StateCovMatrix_t& init_cov_matrix)
    {
        // set example state
        current_state_ = init_state;
        current_cov_ = init_cov_matrix;
    }

    template <typename... PredictionArgs>
    void PredictProcessMeanAndCovariance(PredictionArgs&&... args)
    {
        auto augmented_sigma_points =
            ukf_utils::AugmentedSigmaPoints(current_state_, current_cov_, lambda_, ProcessModel::noise_matrix_squared);

        const auto prediction = ukf_utils::PredictMeanAndCovarianceFromSigmaPoints<ProcessModel>(
            current_predicted_sigma_points_,
            augmented_sigma_points,
            weights_,
            std::forward<const PredictionArgs>(args)...);

        current_state_ = prediction.mean;
        current_cov_ = prediction.covariance;
    }

    template <typename MeasurementModel, typename... MeasurementPredictionArgs>
    void UpdateState(Eigen::Vector<double, MeasurementModel::n>& measure, MeasurementPredictionArgs&&... args)
    {
        using MeasurementVector_t = typename Eigen::Vector<double, MeasurementModel::n>;
        using PredictedMeasurementSigmaPoints_t =
            ukf_utils::PredictedSigmaMatrix<MeasurementModel, ProcessModel::n_sigma_points>;

        PredictedMeasurementSigmaPoints_t predicted_measurement_sigma_points{};

        const auto measurement_prediction = PredictMeasurement<MeasurementModel>(
            predicted_measurement_sigma_points, std::forward<const MeasurementPredictionArgs>(args)...);

        // create matrix for cross correlation: predicted measurement `measurement_prediction.mean` and pred covariance
        // `measurement_prediction.covariance`
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> cross_correlation_matrix{};
        cross_correlation_matrix.fill(0.0f);
        for (int i = 0; i < ProcessModel::n_sigma_points; ++i)
        {
            MeasurementVector_t measure_diff = predicted_measurement_sigma_points.col(i) - measurement_prediction.mean;
            MeasurementModel::Adjust(measure_diff);

            StateVector_t x_diff = current_predicted_sigma_points_.col(i) - current_state_;
            ProcessModel::Adjust(x_diff);

            cross_correlation_matrix = cross_correlation_matrix + weights_(i) * x_diff * measure_diff.transpose();
        }

        // Kalman gain K;
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> K =
            cross_correlation_matrix * measurement_prediction.covariance.inverse();

        // residual
        MeasurementVector_t measure_diff = measure - measurement_prediction.mean;

        // angle normalization
        MeasurementModel::Adjust(measure_diff);

        // update state mean and covariance matrix
        current_state_ = current_state_ + K * measure_diff;
        current_cov_ = current_cov_ - K * measurement_prediction.covariance * K.transpose();
    }

    const PredictedProcessSigmaMatrix_t& GetCurrentPredictedSigmaMatrix() const
    {
        return current_predicted_sigma_points_;
    }
    const StateVector_t& GetCurrentStateVector() const { return current_state_; }
    const StateCovMatrix_t& GetCurrentCovarianceMatrix() const { return current_cov_; }

  private:
    // TODO: extract a general "predict mean and covariance" to be used both for measurements and process
    template <typename MeasurementModel, typename... MeasurementPredictionArgs>
    ukf_utils::MeanAndCovariance<MeasurementModel> PredictMeasurement(
        ukf_utils::PredictedSigmaMatrix<MeasurementModel, ProcessModel::n_sigma_points>& predicted_sigma_matrix_out,
        MeasurementPredictionArgs&&... args)
    {
        auto measurement_prediction = ukf_utils::PredictMeanAndCovarianceFromSigmaPoints<MeasurementModel>(
            predicted_sigma_matrix_out,
            current_predicted_sigma_points_,
            weights_,
            std::forward<const MeasurementPredictionArgs>(args)...);

        // add measurement noise covariance matrix
        measurement_prediction.covariance += MeasurementModel::noise_matrix_squared;

        return measurement_prediction;
    }

    static constexpr double lambda_ = ProcessModel::GetLambda();
    const Eigen::Vector<double, ProcessModel::n_sigma_points> weights_ = ProcessModel::GenerateWeights();

    StateVector_t current_state_{};
    StateCovMatrix_t current_cov_{};
    PredictedProcessSigmaMatrix_t current_predicted_sigma_points_{};
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_UKF_H
