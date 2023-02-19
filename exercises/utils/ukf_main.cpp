#include <iostream>

#include "exercises/models/prediction_models.h"
#include "ukf_utils.h"

int main()
{

    // set example state
    auto x = Eigen::VectorXd(5);
    x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

    // set example covariance matrix
    auto P = Eigen::MatrixXd(5, 5);
    P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020, -0.0013, 0.0077, 0.0011, 0.0071, 0.0060, 0.0030, 0.0011, 0.0054,
        0.0007, 0.0008, -0.0022, 0.0071, 0.0007, 0.0098, 0.0100, -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

    auto ret = GenerateSigmaPoints(x, P);
    std::cout << "sigma points: " << std::endl;
    std::cout << ret << std::endl;

    auto ret_aug = AugmentedSigmaPoints(x, P, Eigen::VectorXd{{0.2f, 0.2f}});
    std::cout << "augmented sigma points: " << std::endl;
    std::cout << ret_aug << std::endl;

    // Predict sigma points
    auto predicted_points = SigmaPointPrediction<CRTVModel>(ret_aug);
    std::cout << "predicted sigma points: " << std::endl;
    std::cout << predicted_points << std::endl;

    return 0;
}