#ifndef SIMPLEUKF_MODELS_MODELS_UTILS_H
#define SIMPLEUKF_MODELS_MODELS_UTILS_H

#include <type_traits>

namespace simpleukf::models_utils
{

template <typename T, typename = void>
struct is_augmented : std::false_type
{
};

template <typename T>
struct is_augmented<T, std::void_t<decltype(T::n_aug)> > : std::true_type
{
    static_assert(T::n_aug > T::n, "Augmented state must have higher dimension than normal state.");
};

template <typename T, typename = void>
struct has_adjust_method : std::false_type
{
};

template <typename T>
struct has_adjust_method<T, std::void_t< decltype(T::Adjust(std::declval<T::PredictedVector>()))> > : std::true_type
{
};

}  // namespace simpleukf::models_utils

#endif  // SIMPLEUKF_MODELS_MODELS_UTILS_H
