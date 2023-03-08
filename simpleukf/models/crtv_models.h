#ifndef SIMPLEUKF_MODELS_CRTV_MODELS_H
#define SIMPLEUKF_MODELS_CRTV_MODELS_H

#include "crtv_model.h"
#include "models_utils.h"
#include "radar_measurement_model.h"

static_assert(simpleukf::models_utils::is_augmented<simpleukf::models::CRTVModel<>>::value);
static_assert(not simpleukf::models_utils::is_augmented<simpleukf::models::RadarModel<>>::value);
static_assert(simpleukf::models_utils::has_adjust_method<simpleukf::models::CRTVModel<>>::value);

#endif  // SIMPLEUKF_MODELS_CRTV_MODELS_Hi
