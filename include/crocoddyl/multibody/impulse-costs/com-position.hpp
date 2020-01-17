///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_COSTS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/impulse-cost-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

class ImpulseCostModelCoM : public ImpulseCostModelAbstract {
 public:
  ImpulseCostModelCoM(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation);
  ImpulseCostModelCoM(boost::shared_ptr<StateMultibody> state);
  ~ImpulseCostModelCoM();

  void calc(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  boost::shared_ptr<ImpulseCostDataAbstract> createData(DataCollectorAbstract* const data);
};

struct ImpulseCostDataCoM : public ImpulseCostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseCostDataCoM(Model* const model, DataCollectorAbstract* const data)
      : ImpulseCostDataAbstract(model, data),
        Arr_Rx(3, model->get_state()->get_nv()),
        dvc_dq(3, model->get_state()->get_nv()),
        ddv_dv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        pinocchio_dv(pinocchio::Data(model->get_state()->get_pinocchio())) {
    Arr_Rx.fill(0);
    dvc_dq.fill(0);
    ddv_dv.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }
    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  pinocchio::Data::Matrix3x Arr_Rx;
  pinocchio::Data::Matrix3x dvc_dq;
  pinocchio::Data::Matrix3x ddv_dv;
  pinocchio::Data pinocchio_dv;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
