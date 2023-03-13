#include <iostream>

#include "utils.h"

int main()
{
    /**
     * Compute RMSE
     */
    std::vector<VectorXd> estimations;
    std::vector<VectorXd> ground_truth;

    // the input list of estimations
    VectorXd e(4);
    e << 1, 1, 0.2, 0.1;
    estimations.push_back(e);
    e << 2, 2, 0.3, 0.2;
    estimations.push_back(e);
    e << 3, 3, 0.4, 0.3;
    estimations.push_back(e);

    // the corresponding list of ground truth values
    VectorXd g(4);
    g << 1.1, 1.1, 0.3, 0.2;
    ground_truth.push_back(g);
    g << 2.1, 2.1, 0.4, 0.3;
    ground_truth.push_back(g);
    g << 3.1, 3.1, 0.5, 0.4;
    ground_truth.push_back(g);

    // call the CalculateRMSE and print out the result
    std::cout << CalculateRMSE(estimations, ground_truth) << std::endl;

    return 0;
}