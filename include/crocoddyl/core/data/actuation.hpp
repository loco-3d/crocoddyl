///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_ACTUATION_HPP_
#define CROCODDYL_CORE_DATA_ACTUATION_HPP_
#include "crocoddyl/core/fwd.hpp"
#include <boost/shared_ptr.hpp>
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

template<typename Scalar>
struct DataCollectorActuationTpl : virtual DataCollectorAbstractTpl<Scalar> {
  DataCollectorActuationTpl(boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation)
    : DataCollectorAbstractTpl<Scalar>(), actuation(actuation) {}
  virtual ~DataCollectorActuationTpl() {}

  boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation;
};

  
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_ACTUATION_HPP_
