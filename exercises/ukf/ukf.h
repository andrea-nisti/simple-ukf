#ifndef EXERCISES_UKF_UKF_H
#define EXERCISES_UKF_UKF_H

#include <iostream>

#include "exercises/models/models.h"
#include "exercises/utils/ukf_utils.h"
#include "ukf.h"

#include <Eigen/Dense>

constexpr double delta_t = 0.1f;

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

    void PredictMeanAndCovariance(StateVector_t& x_pred, StateCovMatrix_t& P_pred)
    {
        using SigmaMatrixAugmented = typename ProcessModel::SigmaMatrixAugmented;
        using ProcessNoiseVector = Eigen::Vector<double, ProcessModel::n_process_noise>;

        auto augmented_sigma_points =
            AugmentedSigmaPoints(current_state_, current_cov_, ProcessNoiseVector{{0.2f, 0.2f}});

        // Predict sigma points
        current_predicted_sigma_points_ = SigmaPointPrediction<ProcessModel>(augmented_sigma_points, delta_t);

        // define spreading parameter and generate weights
        constexpr double lambda = 3 - ProcessModel::n_aug;
        const auto weights = GenerateWeights<ProcessModel::n_aug>(lambda);
        constexpr int n_sigma_points = SigmaMatrixAugmented::ColsAtCompileTime;

        // create vector for predicted state.
        StateVector_t predicted_state_mean{};
        PredictStateMeanFromSigmaPoints(weights, current_predicted_sigma_points_, predicted_state_mean);

        // create covariance matrix for prediction
        StateCovMatrix_t P{};

        // predicted state covariance matrix
        P.fill(0.0);
        for (int i = 0; i < n_sigma_points; ++i)
        {
            // state difference
            StateVector_t x_diff = current_predicted_sigma_points_.col(i) - predicted_state_mean;

            // Adjust state values (e.g. angles wrapping)
            ProcessModel::AdjustState(x_diff);

            P = P + weights(i) * x_diff * x_diff.transpose();
        }

        // write result
        x_pred = predicted_state_mean;
        P_pred = P;
    }

    template <typename MeasurementModel>
    void PredictMeasurement(
        typename MeasurementModel::MeasurementVector& z_out,
        typename MeasurementModel::MeasurementCovMatrix& S_out,
        Eigen::Matrix<double, MeasurementModel::n_z, PredictedSigmaMatrix_t::ColsAtCompileTime>& predicted_sigma_points)
    {
        using MeasurementVector_t = typename MeasurementModel::MeasurementVector;

        // define spreading parameter and generate weights
        constexpr double lambda = 3 - ProcessModel::n_aug;
        const auto weights = GenerateWeights<ProcessModel::n_aug>(lambda);
        constexpr int n_sigma_points = PredictedSigmaMatrix_t::ColsAtCompileTime;

        // mean predicted measurement
        MeasurementVector_t predicted_measurement{};

        // transform sigma points into measurement space
        for (int i = 0; i < n_sigma_points; ++i)
        {
            auto predicted_measure = MeasurementModel{}.PredictMeasure(current_predicted_sigma_points_.col(i), delta_t);

            // measurement model
            predicted_sigma_points(0, i) = predicted_measure(0);  // r
            predicted_sigma_points(1, i) = predicted_measure(1);  // phi
            predicted_sigma_points(2, i) = predicted_measure(2);  // r_dot
        }

        // mean predicted measurement
        PredictStateMeanFromSigmaPoints(weights, predicted_sigma_points, predicted_measurement);

        // innovation covariance matrix S
        typename MeasurementModel::MeasurementCovMatrix S{};
        for (int i = 0; i < n_sigma_points; ++i)
        {  // 2n+1 simga points
            // residual
            MeasurementVector_t z_diff = predicted_sigma_points.col(i) - predicted_measurement;

            MeasurementModel::AdjustMeasure(z_diff);

            S = S + weights(i) * z_diff * z_diff.transpose();
        }

        // add measurement noise covariance matrix
        S = S + MeasurementModel::measurement_cov_matrix;

        // write result
        z_out = predicted_measurement;
        S_out = S;
    }

    template <typename MeasurementModel>
    void UpdateState(typename MeasurementModel::MeasurementVector measure,
                     Eigen::VectorXd& x_out,
                     Eigen::MatrixXd& P_out)
    {
        // using MeasurementVector_t = typename MeasurementModel::MeasurementVector;

        // // create matrix for cross correlation Tc, predicted measurement z_out and pred covariance S_out
        // Eigen::Matrix<double, StateVector_t::n_x, MeasurementVector_t::n_z> Tc{};
        // RadarModel::MeasurementVector z_pred{};
        // RadarModel::MeasurementCovMatrix{};

        // PredictMeasurement<MeasurementModel>(z_pred, S_out);

        // // calculate cross correlation matrix
        // for (int i = 0; i < 2 * n_aug + 1; ++i)
        // {  // 2n+1 simga points
        //     // residual
        //     MeasurementVector_t z_diff = Zsig.col(i) - z_pred;
        //     // angle normalization
        //     MeasurementModel::AdjustMeasure(z_diff);

        //     StateVector_t x_diff = Xsig_pred.col(i) - x;
        //     // angle normalization
        //     while (x_diff(3) > M_PI)
        //         x_diff(3) -= 2. * M_PI;
        //     while (x_diff(3) < -M_PI)
        //         x_diff(3) += 2. * M_PI;

        //     Tc = Tc + weights(i) * x_diff * z_diff.transpose();
        // }

        // // Kalman gain K;
        // MatrixXd K = Tc * S.inverse();

        // // residual
        // VectorXd z_diff = z - z_pred;

        // // angle normalization
        // while (z_diff(1) > M_PI)
        //     z_diff(1) -= 2. * M_PI;
        // while (z_diff(1) < -M_PI)
        //     z_diff(1) += 2. * M_PI;

        // // update state mean and covariance matrix
        // x = x + K * z_diff;
        // P = P - K * S * K.transpose();
    }

  private:
    StateVector_t current_state_{};
    StateCovMatrix_t current_cov_{};
    PredictedSigmaMatrix_t current_predicted_sigma_points_{};
};

#endif // EXERCISES_UKF_UKF_H
