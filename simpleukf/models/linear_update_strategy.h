#ifndef SRC_SIMPLEUKF_SIMPLEUKF_MODELS_LINEAR_UPDATE_STRATEGY_H
#define SRC_SIMPLEUKF_SIMPLEUKF_MODELS_LINEAR_UPDATE_STRATEGY_H

#include "simpleukf/ukf/ukf_utils.h"

template <typename ProcessModel, typename MeasurementModel>
struct LinearUpdateParamenters
{
    // H matrix for measurement prediction
    typename MeasurementModel::PredictedVector measure;
    Eigen::Matrix<double, MeasurementModel::n, ProcessModel::n> H;
    MeasurementModel measurement_model;
};

template <typename ProcessModel>
class LinearUpdateStrategy
{
  public:
    template <typename UpdateParametersStruct>
    void Update(UpdateParametersStruct& parameters,
                const simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& current_hypotesis,
                simpleukf::ukf_utils::MeanAndCovariance<ProcessModel>& mean_and_cov_out)
    {
        const auto measurement_prediction{parameters.measurement_model.Predict(current_hypotesis.mean)};
        const auto measurement_diff = parameters.measure - measurement_prediction;

        const auto PHt = current_hypotesis.covariance * parameters.H.transpose();
        const auto S = parameters.H * PHt + decltype(parameters.measurement_model)::noise_matrix_squared;
        const auto K = PHt * S.inverse();

        const auto I = Eigen::Matrix<double, ProcessModel::n, ProcessModel::n>::Identity();
        mean_and_cov_out.mean = mean_and_cov_out.mean + K * measurement_diff;
        mean_and_cov_out.covariance = (I - K * parameters.H) * mean_and_cov_out.covariance;
    }
};

#endif  // SRC_SIMPLEUKF_SIMPLEUKF_MODELS_LINEAR_UPDATE_STRATEGY_H

// void KalmanFilter::Update(const VectorXd &z) {
//   VectorXd z_pred = H_ * x_;
//   VectorXd y = z - z_pred;
//   MatrixXd Ht = H_.transpose();
//   MatrixXd S = H_ * P_ * Ht + R_;
//   MatrixXd Si = S.inverse();
//   MatrixXd PHt = P_ * Ht;
//   MatrixXd K = PHt * Si;

//   //new estimate
//   x_ = x_ + (K * y);
//   long x_size = x_.size();
//   MatrixXd I = MatrixXd::Identity(x_size, x_size);
//   P_ = (I - K * H_) * P_;
// }