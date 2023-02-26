#ifndef SIMPLEUKF_UKF_UKF_UTILS_H
#define SIMPLEUKF_UKF_UKF_UTILS_H

#include <iostream>

#include <Eigen/Dense>

namespace simpleukf::ukf_utils
{

constexpr inline double GetLamda(int n_state)
{
    return 3 - n_state;
}

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
        Eigen::Vector<double, n_process_noise>{process_noise.array().pow(2)}.asDiagonal();
}

template <int n_state>
Eigen::Matrix<double, n_state, 2 * n_state + 1> GenerateSigmaPoints(const Eigen::Vector<double, n_state>& state_mean,
                                                                    const Eigen::Matrix<double, n_state, n_state>& P)
{
    // create sigma point matrix
    Eigen::Matrix<double, n_state, 2 * n_state + 1> sigma_matrix{};

    // define spreading parameter
    double lambda = GetLamda(n_state);

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
    return GenerateSigmaPoints(state_aug, P_aug);
}

template <typename PredictionModel, typename InputSigmaMatrix, typename... PredictionArgs>
typename PredictionModel::PredictedSigmaMatrix SigmaPointPrediction(const InputSigmaMatrix& sigma_points,
                                                                    const PredictionArgs&&... args)
{
    using StateVector_t = Eigen::Vector<double, PredictionModel::n>;

    constexpr int n_sigma_points{InputSigmaMatrix::ColsAtCompileTime};
    typename PredictionModel::PredictedSigmaMatrix predicted_points{};
    predicted_points.fill(0.0f);
    // this could be parallelized
    for (int i = 0; i < n_sigma_points; ++i)
    {
        const Eigen::Vector<double, InputSigmaMatrix::RowsAtCompileTime>& sigma_state = sigma_points.col(i);

        StateVector_t predicted_sigma_state = PredictionModel{}.Predict(sigma_state, std::forward<const PredictionArgs>(args)...);

        // Fill predicted points matrix
        for (int state_index = 0; state_index < PredictionModel::n; ++state_index)
        {
            predicted_points(state_index, i) = predicted_sigma_state(state_index);
        }
    }

    return predicted_points;
}

template <int state_dim>
constexpr auto GenerateWeights(double lambda)
{
    constexpr int n_sigma_points = 2 * state_dim + 1;
    // set vector for weights
    Eigen::Vector<double, n_sigma_points> weights{};
    double weight_0 = lambda / (lambda + state_dim);
    double weight = 0.5 / (lambda + state_dim);
    weights(0) = weight_0;

    for (int i = 1; i < n_sigma_points; ++i)
    {
        weights(i) = weight;
    }
    return weights;
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

}  // namespace simpleukf::utils

#endif  // SIMPLEUKF_UKF_UKF_UTILS_H