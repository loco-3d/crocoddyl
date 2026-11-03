///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_OBSERVER_HPP_
#define CROCODDYL_CORE_DATA_OBSERVER_HPP_

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorObserverTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DataCollectorObserverTpl()
      : DataCollectorAbstractTpl<Scalar>(),
        xnext(nullptr),
        int_Fx(nullptr),
        int_Fu(nullptr),
        int_Fp(nullptr),
        dissipative_E(nullptr),
        Ex(nullptr),
        Eu(nullptr),
        Ep(nullptr) {}
  virtual ~DataCollectorObserverTpl() {}

  template <class ObserverData>
  void shareObserverData(ObserverData* const data) {
    if (data == nullptr) {
      throw_pretty("Invalid argument: data is null");
    }
    xnext = &data->xnext;
    int_Fx = &data->Fx;
    int_Fu = &data->Fu;
    int_Fp = &data->Fp;
    dissipative_E = &data->dissipative_E;
    Ex = &data->Ex;
    Eu = &data->Eu;
    Ep = &data->Ep;
  }

  bool hasObserverData() const {
    return xnext != nullptr && int_Fx != nullptr && int_Fu != nullptr &&
           int_Fp != nullptr && dissipative_E != nullptr && Ex != nullptr &&
           Eu != nullptr && Ep != nullptr;
  }

  VectorXs* xnext;
  MatrixXs* int_Fx;
  MatrixXs* int_Fu;
  MatrixXs* int_Fp;
  VectorXs* dissipative_E;
  MatrixXs* Ex;
  MatrixXs* Eu;
  MatrixXs* Ep;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorObserverTpl)

#endif  // CROCODDYL_CORE_DATA_OBSERVER_HPP_
