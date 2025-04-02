///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_PCH_
#define CROCODDYL_CORE_PCH_

#include <boost/core/demangle.hpp>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include <pinocchio/utils/static-if.hpp>

#include "crocoddyl/config.hpp"
#include "crocoddyl/core/macros.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/core/utils/scalar.hpp"

#endif  // CROCODDYL_CORE_PCH_
