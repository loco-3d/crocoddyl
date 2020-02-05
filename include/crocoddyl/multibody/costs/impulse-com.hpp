///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

class CostModelImpulseCoM : public CostModelAbstract {
 public:
  CostModelImpulseCoM(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation);
  CostModelImpulseCoM(boost::shared_ptr<StateMultibody> state);
  ~CostModelImpulseCoM();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);
};

struct CostDataImpulseCoM : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataImpulseCoM(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        Arr_Rx(3, model->get_state()->get_nv()),
        dvc_dq(3, model->get_state()->get_nv()),
        ddv_dv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        pinocchio_dv(pinocchio::Data(model->get_state()->get_pinocchio())) {
    Arr_Rx.fill(0);
    dvc_dq.fill(0);
    ddv_dv.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibodyInImpulse* d = dynamic_cast<DataCollectorMultibodyInImpulse*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibodyInImpulse");
    }
    pinocchio = d->pinocchio;
    impulses = d->impulses;
  }

  pinocchio::Data* pinocchio;
  boost::shared_ptr<crocoddyl::ImpulseDataMultiple> impulses;
  Eigen::Matrix3Xd Arr_Rx;
  Eigen::Matrix3Xd dvc_dq;
  Eigen::MatrixXd ddv_dv;
  pinocchio::Data pinocchio_dv;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
