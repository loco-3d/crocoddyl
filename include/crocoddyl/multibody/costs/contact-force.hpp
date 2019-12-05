///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/data/multibody-in-contact.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class CostModelContactForce : public CostModelAbstract {
 public:
  CostModelContactForce(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                        const FrameForce& fref, const std::size_t& nu);
  CostModelContactForce(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                        const FrameForce& fref);
  CostModelContactForce(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nu);
  CostModelContactForce(boost::shared_ptr<StateMultibody> state, const FrameForce& fref);
  ~CostModelContactForce();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrameForce& get_fref() const;
  void set_fref(const FrameForce& fref);

 protected:
  FrameForce fref_;
};

struct CostDataContactForce : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataContactForce(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.fill(0);

    // Check that proper shared data has been passed
    DataCollectorMultibodyInContact* d = dynamic_cast<DataCollectorMultibodyInContact*>(shared);
    if (d == NULL) {
      throw std::invalid_argument("The shared data should be derived from DataCollectorMultibodyInContact");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  boost::shared_ptr<ContactDataAbstract> contact;
  Eigen::MatrixXd Arr_Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
