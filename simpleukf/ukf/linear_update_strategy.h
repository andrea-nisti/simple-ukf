#ifndef SIMPLEUKF_UKF_LINEAR_UPDATE_STRATEGY_H
#define SIMPLEUKF_UKF_LINEAR_UPDATE_STRATEGY_H

#include "simpleukf/models/models_utils.h"
#include "simpleukf/ukf/ukf_utils.h"

namespace simpleukf::ukf
{
template <typename ProcessModel, typename MeasurementModel>
class LinearUpdateStrategy
{
  public:
    LinearUpdateStrategy(Eigen::Ref<Eigen::Matrix<double, MeasurementModel::n, ProcessModel::n>> H) : H_{H} {}

    void Update(const Eigen::Ref<typename MeasurementModel::PredictedVector> measure,
                const simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& current_hypotesis,
                simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& mean_and_cov_out)
    {
        const auto measurement_prediction{measurement_model_.Predict(current_hypotesis.mean)};
        typename MeasurementModel::PredictedVector measurement_diff = measure - measurement_prediction;

        models_utils::AdjustIfNeeded<MeasurementModel>(measurement_diff);

        const auto PHt = current_hypotesis.covariance * H_.transpose();
        const auto S = H_ * PHt + decltype(measurement_model_)::noise_matrix_squared;
        const auto K = PHt * S.inverse();

        const auto I = Eigen::Matrix<double, ProcessModel::n, ProcessModel::n>::Identity();
        mean_and_cov_out.mean = current_hypotesis.mean + K * measurement_diff;
        mean_and_cov_out.covariance = (I - K * H_) * current_hypotesis.covariance;
    }

  private:
    MeasurementModel measurement_model_{};
    Eigen::Matrix<double, MeasurementModel::n, ProcessModel::n> H_;
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_LINEAR_UPDATE_STRATEGY_H