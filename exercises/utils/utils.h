#ifndef EXERCISES_UTILS_UTILS_H
#define EXERCISES_UTILS_UTILS_H

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd CalculateJacobian(const VectorXd& x_state);
VectorXd CalculateRMSE(const std::vector<VectorXd>& estimations,
                       const std::vector<VectorXd>& ground_truth);

#endif  // EXERCISES_UTILS_UTILS_H
