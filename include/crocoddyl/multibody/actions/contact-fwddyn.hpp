///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

#include <stdexcept>
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {

class DifferentialActionModelContactFwdDynamics : public DifferentialActionModelAbstract {
 public:
  DifferentialActionModelContactFwdDynamics(boost::shared_ptr<StateMultibody> state,
                                            boost::shared_ptr<ActuationModelFloatingBase> actuation,
                                            boost::shared_ptr<ContactModelMultiple> contacts,
                                            boost::shared_ptr<CostModelSum> costs, const double& JMinvJt_damping = 0.,
                                            const bool& enable_force = false);
  ~DifferentialActionModelContactFwdDynamics();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  const boost::shared_ptr<ActuationModelFloatingBase>& get_actuation() const;
  const boost::shared_ptr<ContactModelMultiple>& get_contacts() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  pinocchio::Model& get_pinocchio() const;
  const Eigen::VectorXd& get_armature() const;
  const double& get_damping_factor() const;

  void set_armature(const Eigen::VectorXd& armature);
  void set_damping_factor(const double& damping);

 private:
  boost::shared_ptr<ActuationModelFloatingBase> actuation_;
  boost::shared_ptr<ContactModelMultiple> contacts_;
  boost::shared_ptr<CostModelSum> costs_;
  pinocchio::Model& pinocchio_;
  bool with_armature_;
  Eigen::VectorXd armature_;
  double JMinvJt_damping_;
  bool enable_force_;
};

struct DifferentialActionDataContactFwdDynamics : public DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit DifferentialActionDataContactFwdDynamics(Model* const model)
      : DifferentialActionDataAbstract(model),
        pinocchio(pinocchio::Data(model->get_pinocchio())),
        multibody(&pinocchio, model->get_contacts()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        Kinv(model->get_state()->get_nv() + model->get_contacts()->get_nc(),
             model->get_state()->get_nv() + model->get_contacts()->get_nc()),
        df_dx(model->get_contacts()->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_contacts()->get_nc(), model->get_nu()) {
    actuation = model->get_actuation()->createData();
    costs->shareMemory(this);
    Kinv.fill(0);
    df_dx.fill(0);
    df_du.fill(0);
  }

  pinocchio::Data pinocchio;
  boost::shared_ptr<ActuationDataAbstract> actuation;
  DataCollectorMultibodyInContact multibody;
  boost::shared_ptr<CostDataSum> costs;
  Eigen::MatrixXd Kinv;
  Eigen::MatrixXd df_dx;
  Eigen::MatrixXd df_du;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
