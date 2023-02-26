#ifndef SIMPLEUKF_MODELS_CRTV_CRTV_MODEL_H
#define SIMPLEUKF_MODELS_CRTV_CRTV_MODEL_H

#include <Eigen/Dense>
namespace simpleukf::models
{

class CRTVModel
{
  public:
    static constexpr int n = 5;
    static constexpr int n_process_noise = 2;
    static constexpr int n_aug = n + n_process_noise;
    static constexpr int n_sigma_points = 2 * n_aug + 1;

    using StateVector = Eigen::Vector<double, n>;
    using StateCovMatrix = Eigen::Matrix<double, n, n>;

    using SigmaMatrix = Eigen::Matrix<double, n, 2 * n + 1>;
    using SigmaMatrixAugmented = Eigen::Matrix<double, n_aug, n_sigma_points>;
    using PredictedSigmaMatrix = Eigen::Matrix<double, n, n_sigma_points>;

    template <int N>
    static StateVector Predict(const Eigen::Vector<double, N>& current_state, const double delta_t)
    {
        static_assert((N == n) or (N == n_aug), "State dimension must be 5 (normal state) or 7 (augmented state)");

        double p_x = current_state(0);
        double p_y = current_state(1);
        double v = current_state(2);
        double yaw = current_state(3);
        double yawd = current_state(4);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001)
        {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else
        {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // add noise if necessary
        if constexpr (current_state.RowsAtCompileTime == n_aug)
        {
            double nu_a = current_state(5);
            double nu_yawdd = current_state(6);

            px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
            py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
            v_p = v_p + nu_a * delta_t;

            yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
            yawd_p = yawd_p + nu_yawdd * delta_t;
        }

        // write predicted sigma point into right column
        StateVector return_state;

        return_state(0) = px_p;
        return_state(1) = py_p;
        return_state(2) = v_p;
        return_state(3) = yaw_p;
        return_state(4) = yawd_p;

        return return_state;
    }

    static void AdjustState(StateVector& to_be_adjusted)
    {
        while (to_be_adjusted(3) > M_PI)
            to_be_adjusted(3) -= 2. * M_PI;
        while (to_be_adjusted(3) < -M_PI)
            to_be_adjusted(3) += 2. * M_PI;
    }
};

}  // namespace simpleukf::models

#endif  // SIMPLEUKF_MODELS_CRTV_CRTV_MODEL_H
