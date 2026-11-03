///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_ACTUATION_HPP_
#define CROCODDYL_CORE_DATA_ACTUATION_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorActuationTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActuationTpl(
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation)
      : DataCollectorAbstractTpl<Scalar>(), actuation(actuation) {}
  virtual ~DataCollectorActuationTpl() {}

  std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation;
};

template <typename Scalar>
struct DataCollectorActuationParamsTpl : DataCollectorActuationTpl<Scalar>,
                                         DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActuationParamsTpl(
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorActuationTpl<Scalar>(actuation),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorActuationParamsTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorActuationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActuationParamsTpl)

#endif  // CROCODDYL_CORE_DATA_ACTUATION_HPP_
