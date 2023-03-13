#ifndef SIMPLEUKF_UTILS_UTILS_H
#define SIMPLEUKF_UTILS_UTILS_H

#include <vector>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd CalculateJacobian(const VectorXd& x_state);
VectorXd CalculateRMSE(const std::vector<VectorXd>& estimations, const std::vector<VectorXd>& ground_truth);

#endif  // SIMPLEUKF_UTILS_UTILS_H
