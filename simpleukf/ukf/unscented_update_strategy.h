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
        const Eigen::Ref<ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points>>&
            current_predicted_sigma_points,
        const Eigen::Ref<Eigen::Vector<double, ProcessModel::n_sigma_points>>& weights)
        : current_predicted_sigma_points_{current_predicted_sigma_points}, weights_{weights}
    {
    }

    void Update(const Eigen::Ref<typename MeasurementModel::PredictedVector>& measure,
                const simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& current_hypotesis,
                simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& mean_and_cov_out)
    {
        simpleukf::ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points>
            measurement_predicted_sigma_matrix_out;
        auto measurement_prediction = ukf_utils::PredictMeanAndCovarianceFromSigmaPoints<MeasurementModel>(
            measurement_predicted_sigma_matrix_out, current_predicted_sigma_points_, weights)
            // std::forward<const MeasurementPredictionArgs>(args)...);

            // add measurement noise covariance matrix
            if constexpr (not models_utils::is_augmented<MeasurementModel>)
        {
            measurement_prediction.covariance += MeasurementModel::noise_matrix_squared;
        }
    }

  private:
    MeasurementModel measurement_model_{};

    const Eigen::Vector<double, ProcessModel::n_sigma_points>& weights_;
    const ukf_utils::PredictedSigmaMatrix<ProcessModel, ProcessModel::n_sigma_points>& current_predicted_sigma_points_;
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_UNSCENTED_UPDATE_STRATEGY_H