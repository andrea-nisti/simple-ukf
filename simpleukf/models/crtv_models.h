#ifndef SIMPLEUKF_MODELS_CRTV_MODELS_H
#define SIMPLEUKF_MODELS_CRTV_MODELS_H

#include "crtv/crtv_model.h"
#include "crtv/radar_measurement_model.h"
#include "models_utils.h"

static_assert(simpleukf::models_utils::is_augmented<simpleukf::models::CRTVModel<>>::value);
static_assert(not simpleukf::models_utils::is_augmented<simpleukf::models::RadarModel<> >::value);

#endif  // SIMPLEUKF_MODELS_CRTV_MODELS_H
