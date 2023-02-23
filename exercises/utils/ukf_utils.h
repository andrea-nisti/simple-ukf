#ifndef EXERCISES_UTILS_UKF_UTILS_H
#define EXERCISES_UTILS_UKF_UTILS_H

#include <iostream>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr inline double GetLamda(int n_state)
{
    return 3 - n_state;
}

template <int n_x, int n_aug, int n_process_noise>
void AugmentStates(const Eigen::Vector<double, n_x>& state_mean,
                   const Eigen::Matrix<double, n_x, n_x>& P,
                   const Eigen::Vector<double, n_process_noise> process_noise,
                   Eigen::Vector<double, n_aug>& x_aug,
                   Eigen::Matrix<double, n_aug, n_aug>& P_aug)
{
    static_assert(n_x + n_process_noise == n_aug, "State dimensions not matching");
    x_aug.head(n_x) = state_mean;

    int index = n_x;
    while (index < n_aug)
    {
        // set rest to zero
        x_aug(index) = 0;
        ++index;
    }

    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x, n_x) = P;

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
    const MatrixXd& A = P.llt().matrixL();

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

template <int n_x, int n_process_noise>
auto AugmentedSigmaPoints(const Eigen::Vector<double, n_x>& state_mean,
                          const Eigen::Matrix<double, n_x, n_x>& P,
                          const Eigen::Vector<double, n_process_noise> process_noise)
{
    // create augmented mean state
    // set augmented dimension
    constexpr int n_aug = n_x + process_noise.RowsAtCompileTime;
    Eigen::Vector<double, n_aug> state_aug{};
    // create augmented covariance matrix
    Eigen::Matrix<double, n_aug, n_aug> P_aug;

    AugmentStates(state_mean, P, std::move(process_noise), state_aug, P_aug);
    return GenerateSigmaPoints(state_aug, P_aug);
}

// TODO: this can be fused with measurement sigma prediction
template <typename PredictionModel, typename InputSigmaMatrix>
typename PredictionModel::PredictedSigmaMatrix SigmaPointPrediction(const InputSigmaMatrix& sigma_points,
                                                                    double delta_t)
{
    using StateVector_t = typename PredictionModel::StateVector;

    constexpr int n_sigma_points{InputSigmaMatrix::ColsAtCompileTime};
    typename PredictionModel::PredictedSigmaMatrix predicted_points{};

    // this could be parallelized
    for (int i = 0; i < n_sigma_points; ++i)
    {
        const Eigen::Vector<double, sigma_points.RowsAtCompileTime>& prev_sigma_state = sigma_points.col(i);
        StateVector_t predicted_sigma_state = PredictionModel{}.PredictState(prev_sigma_state, delta_t);

        // Fill predicted points matrix
        for (int state_index = 0; state_index < PredictionModel::n_x; ++state_index)
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

template <int n_state, typename SigmaMatrix, int n_sigma_points>
void PredictMeanFromSigmaPoints(const Eigen::Vector<double, n_sigma_points>& weights,
                                     const SigmaMatrix& current_predicted_sigma_points,
                                     Eigen::Vector<double, n_state>& computed_mean)
{
    computed_mean.fill(0.0);
    for (int i = 0; i < n_sigma_points; ++i)
    {  // iterate over sigma points
        computed_mean = computed_mean + weights(i) * current_predicted_sigma_points.col(i);
    }
}

#endif  // EXERCISES_UTILS_UKF_UTILS_H
