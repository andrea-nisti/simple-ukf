#ifndef SIMPLEUKF_UKF_UNSCENTED_UPDATE_STRATEGY_H
#define SIMPLEUKF_UKF_UNSCENTED_UPDATE_STRATEGY_H

#include "simpleukf/models/models_utils.h"
#include "simpleukf/ukf/ukf_utils.h"

namespace simpleukf::ukf
{

template <typename ProcessModel, typename MeasurementModel>
class UnscentedUpdateStrategy
{
  public:
    UnscentedUpdateStrategy(
        const Eigen::Ref<const ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points>>&
            current_predicted_sigma_points,
        const Eigen::Ref<const Eigen::Vector<double, ProcessModel::n_sigma_points>>& weights)
        : current_predicted_sigma_points_{current_predicted_sigma_points}, weights_{weights}
    {
    }

    void Update(const Eigen::Ref<const typename MeasurementModel::PredictedVector>& measure,
                const simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& current_hypotesis,
                simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& mean_and_cov_out) const
    {
        simpleukf::ukf_utils::PredictedSigmaMatrix<MeasurementModel, ProcessModel::n_sigma_points>
            measurement_predicted_sigma_matrix_out;

        auto measurement_prediction = ukf_utils::PredictMeanAndCovarianceFromSigmaPoints<MeasurementModel>(
            measurement_predicted_sigma_matrix_out, current_predicted_sigma_points_, weights_);
        // std::forward<const MeasurementPredictionArgs>(args)...);

        // add measurement noise covariance matrix
        measurement_prediction.covariance += MeasurementModel::noise_matrix_squared;

        // create matrix for cross correlation: predicted measurement `measurement_prediction.mean` and pred covariance
        // `measurement_prediction.covariance`
        const auto cross_correlation_matrix =
            ukf_utils::ComputeCrossCorrelation<ProcessModel, MeasurementModel, ProcessModel::n_sigma_points>(
                current_predicted_sigma_points_,
                current_hypotesis.mean,
                measurement_predicted_sigma_matrix_out,
                measurement_prediction.mean,
                weights_);

        // Kalman gain K;
        Eigen::Matrix<double, ProcessModel::n, MeasurementModel::n> K =
            cross_correlation_matrix * measurement_prediction.covariance.inverse();

        // residual
        typename MeasurementModel::PredictedVector measure_diff = measure - measurement_prediction.mean;

        // angle normalization
        models_utils::AdjustIfNeeded<MeasurementModel>(measure_diff);

        // update state mean and covariance matrix
        mean_and_cov_out.mean = current_hypotesis.mean + K * measure_diff;
        mean_and_cov_out.covariance =
            current_hypotesis.covariance - K * measurement_prediction.covariance * K.transpose();
    }

  private:
    // TODO: should we keep a copy or not?
    const ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points> current_predicted_sigma_points_;
    const Eigen::Vector<double, ProcessModel::n_sigma_points> weights_;
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_UNSCENTED_UPDATE_STRATEGY_H