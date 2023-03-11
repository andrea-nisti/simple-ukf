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
    void Init(const Eigen::Ref<const StateVector_t>& init_state, const Eigen::Ref<const StateCovMatrix_t>& init_cov_matrix)
    {
        // set example state
        current_hypotesis_.mean = init_state;
        current_hypotesis_.covariance = init_cov_matrix;
    }

    template <typename... PredictionArgs>
    void PredictProcessMeanAndCovariance(PredictionArgs... args)
    {
        auto augmented_sigma_points = ukf_utils::AugmentedSigmaPoints(
            current_hypotesis_.mean, current_hypotesis_.covariance, lambda_, ProcessModel::noise_matrix_squared);

        const auto prediction = ukf_utils::PredictMeanAndCovarianceFromSigmaPoints<ProcessModel>(
            current_predicted_sigma_points_,
            augmented_sigma_points,
            weights_,
            std::forward<PredictionArgs>(args)...);

        current_hypotesis_.mean = prediction.mean;
        current_hypotesis_.covariance = prediction.covariance;
    }

    template <typename MeasurementModel, typename MeasureUpdateStrategy, typename... MeasurementPredictionArgs>
    void UpdateState(const Eigen::Ref<const Eigen::Vector<double, MeasurementModel::n>>& measure,
                     MeasureUpdateStrategy&& measurement_update_strategy)
    {
        measurement_update_strategy.Update(measure, current_hypotesis_, current_hypotesis_);
    }

    const PredictedProcessSigmaMatrix_t& GetCurrentPredictedSigmaMatrix() const
    {
        return current_predicted_sigma_points_;
    }

    const StateVector_t& GetCurrentStateVector() const { return current_hypotesis_.mean; }
    const StateCovMatrix_t& GetCurrentCovarianceMatrix() const { return current_hypotesis_.covariance; }

    UKF<ProcessModel>& operator=(const UKF<ProcessModel>& right){
        current_hypotesis_.mean = right.GetCurrentStateVector();
        current_hypotesis_.covariance = right.GetCurrentCovarianceMatrix();
        current_predicted_sigma_points_ = right.GetCurrentPredictedSigmaMatrix();

        return *this;
    };

  private:
    static constexpr double lambda_{ProcessModel::GetLambda()};
    const Eigen::Vector<double, ProcessModel::n_sigma_points> weights_{ProcessModel::GenerateWeights()};

    ukf_utils::MeanAndCovariance<ProcessModel> current_hypotesis_{};
    PredictedProcessSigmaMatrix_t current_predicted_sigma_points_{};
};

}  // namespace simpleukf::ukf

#endif  // SIMPLEUKF_UKF_UKF_H
