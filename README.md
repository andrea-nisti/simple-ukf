# SIMPLEUKF - a simple ukf implementation for learning purposes
This projects implements a library that can be used in Unscented Kalman Filtering problems. The Unscented Kalman filter is an estimation algorithm that it is widely used in non linear motion tracking problems in the automotive and robotics fiels. It updates its belief of the state, also called hypothesis, through a prediction step and an update step.

The **prediction** step consists in predicting the state of the model from time k to time k+1 using the mathematical stochastic model that maps the current state to the next one, this is called Process Model. 

The **update** step integrates measures (or observations) provided by external components, such as sensors, with the predicted state. The result of this step is a new hypothesis representing the estimated state of the system.

## Common terminology

Here a list of some common terms that are used in the library code:

- `StateVector`: structure that represents the state of the system
  - `PredictedVector` is an alias top this type
- `MeasurementVector`: structure representing the values of a single measure.
  - `PredictedVector` is an alias to this type
- `ProcessModel`: used in generic programming, this template represents a class that describes the mathematical motion model of the estimated system
- `MeasurementModel`: this template represent the sensor model, namely the mapping between a system state and the measurement vector that ideally should arrive in that particular state
- `PredictedSigmaMatrix`: matrix containing the value of sigma points after being transformed through the process/measurement models
- `State/MeasurementCovMatrix` these two types represents the covariance matrix associated to a particular model. 
  - `PredictedCovMatrix` is an alias to this type
- `n`, `n_aug` and `n_sigma_points` contains respectively the dimension of a process or measurement state, the dimension of the process augmented state, and the number of generated sigma points

## Models conventions

Each model is a template class that must adhere to a set of design constraints to be correct. 

Here an example of a simple model:
```Cpp
template <typename NoiseConstants = LidarNoiseConstantDefault>
struct LidarModel
{
  private:
    using CTRVModelInt = CTRVModel<>;

  public:
    static constexpr double std_laspx = NoiseConstants::std_laspx;
    static constexpr double std_laspy = NoiseConstants::std_laspy;

    static constexpr int n = 2;
    using PredictedVector = Eigen::Vector<double, n>;
    using MeasurementVector = PredictedVector;

    // clang-format off
    using PredictedCovMatrix = Eigen::Matrix<double, n, n>;
    inline static const PredictedCovMatrix noise_matrix_squared =
        (PredictedCovMatrix() << 
         std_laspx * std_laspx,   0,
         0,    std_laspy * std_laspy).finished();
    // clang-format on
    using MeasurementCovMatrix = PredictedCovMatrix;

    PredictedVector Predict(const CTRVModelInt::PredictedVector& curr_state) const
    {
        PredictedVector ret{curr_state(0), curr_state(1)};
        return ret;
    }
};
```
The minimum viable set of public members that a model should expose is:

- `static constexpr int n` -- the measure dimension (e.g. the number of elements in the measurement vector)
-  `using PredictedVector` -- the vector representing a measure
-  `using MeasurementVector` -- same as above, this alias is for learning purposes
- `using PredictedCovMatrix` -- the matrix representing the covariance
- `using MeasurementCovMatrix` -- alias to `PredictedCovMatrix` for learing and clarity
-  `inline static const PredictedCovMatrix noise_matrix_squared` -- covariance matrix that contains sigma squared values
-  `PredictedVector Predict(const CTRVModelInt::PredictedVector& curr_state) const` -- function that maps a current process state to the measure that would be 'ideally' detected in that particular state

Additionally, a Process Model should expose the variable `n_sigma_points` that represents the number of computed sigma points.

If a process model exposes `n_aug` (the dimension of the augmented state), the augmentation process takes place. In this particular case, the member `noise_matrix_squared` should be a Vector containing the noise standard deviations.

# Usage

Here you can find some instructions on how to build and run examples or tests. The supported build system is `bazel` but a CMakeLists is also provided, though it does not include every availabe target.

## Building and running


## Code description

The main logic is encapsulated in the LinearUpdateStrategy and the UnscentedUpdateStrategy. Support functions can be found in models_utils.h and ukf_utils.

Please check the class `Fusion` (simpleukf/examples/fusion.h) for an example on how to use the UKF class. This object is intended to be a lightweight element where every Matrix has compile time dimensions that are computed from the models public members. This aspect should increase readability and imrove learning, as well as performances.

