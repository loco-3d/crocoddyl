///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class DifferentialActionModelContactFwdDynamics : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelContactFwdDynamics(StateMultibody& state, ActuationModelFloatingBase& actuation,
                                            ContactModelMultiple& contacts, CostModelSum& costs);
  ~DifferentialActionModelContactFwdDynamics();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  ActuationModelFloatingBase& get_actuation() const;
  ContactModelMultiple& get_contacts() const;
  CostModelSum& get_costs() const;
  pinocchio::Model& get_pinocchio() const;
  const Eigen::VectorXd& get_armature() const;
  void set_armature(const Eigen::VectorXd& armature);

 private:
  ActuationModelFloatingBase& actuation_;
  ContactModelMultiple& contacts_;
  CostModelSum& costs_;
  pinocchio::Model& pinocchio_;
  bool force_aba_;
  Eigen::VectorXd armature_;
  double JMinvJt_damping_;
};

struct DifferentialActionDataContactFwdDynamics : public DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit DifferentialActionDataContactFwdDynamics(Model* const model)
      : DifferentialActionDataAbstract(model), pinocchio(pinocchio::Data(model->get_pinocchio())) {
    actuation = model->get_actuation().createData();
    contacts = model->get_contacts().createData(&pinocchio);
    costs = model->get_costs().createData(&pinocchio);
    shareCostMemory(costs);
  }

  pinocchio::Data pinocchio;
  boost::shared_ptr<ActuationDataAbstract> actuation;
  boost::shared_ptr<ContactDataMultiple> contacts;
  boost::shared_ptr<CostDataSum> costs;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
