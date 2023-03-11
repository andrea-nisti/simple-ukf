#ifndef SIMPLEUKF_UKF_UKF_UTILS_H
#define SIMPLEUKF_UKF_UKF_UTILS_H

#include <iostream>
#include <optional>

#include <Eigen/Dense>

namespace simpleukf::ukf_utils
{

template <typename PredictionModel, int n_sigma_points>
using PredictedSigmaMatrix = Eigen::Matrix<double, PredictionModel::n, n_sigma_points>;

template <typename PredictionModel>
struct MeanAndCovariance
{
    typename PredictionModel::PredictedVector mean;
    typename PredictionModel::PredictedCovMatrix covariance;
};

template <int n, int n_aug, int n_process_noise>
void AugmentStates(const Eigen::Vector<double, n>& state_mean,
                   const Eigen::Matrix<double, n, n>& P,
                   const Eigen::Vector<double, n_process_noise> process_noise,
                   Eigen::Vector<double, n_aug>& x_aug,
                   Eigen::Matrix<double, n_aug, n_aug>& P_aug)
{
    static_assert(n + n_process_noise == n_aug, "State dimensions not matching");
    x_aug.fill(0.0f);
    x_aug.head(n) = state_mean;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(n, n) = P;

    // ---calculate sigma square
    P_aug.bottomRightCorner(n_process_noise, n_process_noise) =
        Eigen::Vector<double, n_process_noise>{process_noise.array()}.asDiagonal();
}

template <int n_state>
Eigen::Matrix<double, n_state, 2 * n_state + 1> GenerateSigmaPoints(const Eigen::Vector<double, n_state>& state_mean,
                                                                    const Eigen::Matrix<double, n_state, n_state>& P,
                                                                    const double lambda)
{
    // create sigma point matrix
    Eigen::Matrix<double, n_state, 2 * n_state + 1> sigma_matrix{};

    // calculate square root of P
    // TODO: make matrix A compile time
    const Eigen::MatrixXd& A = P.llt().matrixL();

    // set first column of sigma point matrix
    sigma_matrix.col(0) = state_mean;

    // set remaining sigma points
    for (int i = 0; i < n_state; ++i)
    {
        sigma_matrix.col(i + 1) = state_mean + sqrt(lambda + n_state) * A.col(i);
        sigma_matrix.col(i + 1 + n_state) = state_mean - sqrt(lambda + n_state) * A.col(i);
    }

    // write result
    return sigma_matrix;
}

template <int n, int n_process_noise>
auto AugmentedSigmaPoints(const Eigen::Vector<double, n>& state_mean,
                          const Eigen::Matrix<double, n, n>& P,
                          const double lambda,
                          Eigen::Vector<double, n_process_noise> process_noise)
{
    // create augmented mean state
    // set augmented dimension
    constexpr int n_aug = n + process_noise.RowsAtCompileTime;
    Eigen::Vector<double, n_aug> state_aug{};
    state_aug.fill(0.0);
    // create augmented covariance matrix
    Eigen::Matrix<double, n_aug, n_aug> P_aug;

    AugmentStates(state_mean, P, std::move(process_noise), state_aug, P_aug);
    return GenerateSigmaPoints(state_aug, P_aug, lambda);
}

template <typename PredictionModel, typename InputSigmaMatrix, typename... PredictionArgs>
PredictedSigmaMatrix<PredictionModel, InputSigmaMatrix::ColsAtCompileTime> SigmaPointPrediction(
    const InputSigmaMatrix& sigma_points,
    const PredictionArgs&&... args)
{
    using StateVector_t = Eigen::Vector<double, PredictionModel::n>;

    constexpr int n_sigma_points{InputSigmaMatrix::ColsAtCompileTime};
    PredictedSigmaMatrix<PredictionModel, InputSigmaMatrix::ColsAtCompileTime> predicted_points{};
    predicted_points.fill(0.0f);
    // this could be parallelized
    for (int i = 0; i < n_sigma_points; ++i)
    {
        const Eigen::Vector<double, InputSigmaMatrix::RowsAtCompileTime>& sigma_state = sigma_points.col(i);

        StateVector_t predicted_sigma_state =
            PredictionModel{}.Predict(sigma_state, std::forward<const PredictionArgs>(args)...);

        // Fill predicted points matrix
        for (int state_index = 0; state_index < PredictionModel::n; ++state_index)
        {
            predicted_points(state_index, i) = predicted_sigma_state(state_index);
        }
    }

    return predicted_points;
}

template <int n_state, int n_sigma_points>
Eigen::Vector<double, n_state> ComputeMeanFromSigmaPoints(
    const Eigen::Vector<double, n_sigma_points>& weights,
    const Eigen::Matrix<double, n_state, n_sigma_points>& current_predicted_sigma_points)
{
    Eigen::Vector<double, n_state> computed_mean{};
    computed_mean.fill(0.0);
    for (int i = 0; i < n_sigma_points; ++i)
    {  // iterate over sigma points
        computed_mean = computed_mean + weights(i) * current_predicted_sigma_points.col(i);
    }

    return computed_mean;
}

template <int n_state, int n_sigma_points, typename DifferenceAdjuster>
Eigen::Matrix<double, n_state, n_state> ComputeCovarianceFromSigmaPoints(
    const Eigen::Vector<double, n_sigma_points>& weights,
    const Eigen::Matrix<double, n_state, n_sigma_points>& current_predicted_sigma_points,
    const Eigen::Vector<double, n_state>& predicted_state_mean,
    const DifferenceAdjuster difference_adjuster = nullptr)
{
    // create covariance matrix for prediction
    Eigen::Matrix<double, n_state, n_state> P{};
    P.fill(0.0);

    for (int i = 0; i < n_sigma_points; ++i)
    {
        // state difference
        Eigen::Vector<double, n_state> diff = current_predicted_sigma_points.col(i) - predicted_state_mean;

        // Adjust state values (e.g. angles wrapping)
        if (difference_adjuster)
            difference_adjuster(diff);

        P = P + weights(i) * diff * diff.transpose();
    }
    return P;
}

template <typename PredictionModel, typename InputSigmaMatrix, typename... PredictionArgs>
MeanAndCovariance<PredictionModel> PredictMeanAndCovarianceFromSigmaPoints(
    PredictedSigmaMatrix<PredictionModel, InputSigmaMatrix::ColsAtCompileTime>& predicted_sigma_matrix_out,
    const InputSigmaMatrix& current_sigma_points,
    const Eigen::Vector<double, InputSigmaMatrix::ColsAtCompileTime> weights,
    const PredictionArgs... args)
{
    // Predict sigma points
    predicted_sigma_matrix_out.fill(0.0f);
    predicted_sigma_matrix_out = ukf_utils::SigmaPointPrediction<PredictionModel>(
        current_sigma_points, std::forward<const PredictionArgs>(args)...);

    const typename PredictionModel::PredictedVector& predicted_state =
        ukf_utils::ComputeMeanFromSigmaPoints(weights, predicted_sigma_matrix_out);

    const typename PredictionModel::PredictedCovMatrix& predicted_cov = ukf_utils::ComputeCovarianceFromSigmaPoints(
        weights, predicted_sigma_matrix_out, predicted_state, &PredictionModel::Adjust);

    return {predicted_state, predicted_cov};
}

template <typename ModelA, typename ModelB, int n_sigma_points>
Eigen::Matrix<double, ModelA::n, ModelB::n> ComputeCrossCorrelation(
    const Eigen::Matrix<double, ModelA::n, n_sigma_points>& matrix_a,
    const typename ModelA::PredictedVector& mean_a,
    const Eigen::Matrix<double, ModelB::n, n_sigma_points>& matrix_b,
    const typename ModelB::PredictedVector& mean_b,
    const Eigen::Vector<double, n_sigma_points> weights)
{
    Eigen::Matrix<double, ModelA::n, ModelB::n> cross_correlation_matrix{};
    cross_correlation_matrix.fill(0.0f);
    for (int i = 0; i < n_sigma_points; ++i)
    {
        typename ModelA::PredictedVector diff_a = matrix_a.col(i) - mean_a;
        ModelA::Adjust(diff_a);

        typename ModelB::PredictedVector diff_b = matrix_b.col(i) - mean_b;
        ModelB::Adjust(diff_b);

        cross_correlation_matrix = cross_correlation_matrix + weights(i) * diff_a * diff_b.transpose();
    }
    return cross_correlation_matrix;
}

}  // namespace simpleukf::ukf_utils

#endif  // SIMPLEUKF_UKF_UKF_UTILS_H
