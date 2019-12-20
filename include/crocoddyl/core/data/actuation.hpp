///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_ACTUATION_HPP_
#define CROCODDYL_CORE_DATA_ACTUATION_HPP_

#include <boost/shared_ptr.hpp>
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

struct DataCollectorActuation : virtual DataCollectorAbstract {
  DataCollectorActuation(boost::shared_ptr<ActuationDataAbstract> actuation)
      : DataCollectorAbstract(), actuation(actuation) {}
  virtual ~DataCollectorActuation() {}

  boost::shared_ptr<ActuationDataAbstract> actuation;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_ACTUATION_HPP_
