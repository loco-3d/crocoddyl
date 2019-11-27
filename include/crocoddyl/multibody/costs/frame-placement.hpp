///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class CostModelFramePlacement : public CostModelAbstract {
 public:
  CostModelFramePlacement(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const FramePlacement& Fref,
                          const std::size_t& nu);
  CostModelFramePlacement(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const FramePlacement& Fref);
  CostModelFramePlacement(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref, const std::size_t& nu);
  CostModelFramePlacement(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref);
  ~CostModelFramePlacement();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const FramePlacement& get_Mref() const;

 private:
  FramePlacement Mref_;
  pinocchio::SE3 oMf_inv_;
};

struct CostDataFramePlacement : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFramePlacement(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data),
        J(6, model->get_state()->get_nv()),
        rJf(6, 6),
        fJf(6, model->get_state()->get_nv()),
        Arr_J(6, model->get_state()->get_nv()) {
    r.fill(0);
    J.fill(0);
    rJf.fill(0);
    fJf.fill(0);
    Arr_J.fill(0);
  }

  pinocchio::Motion::Vector6 r;
  pinocchio::SE3 rMf;
  pinocchio::Data::Matrix6x J;
  pinocchio::Data::Matrix6 rJf;
  pinocchio::Data::Matrix6x fJf;
  pinocchio::Data::Matrix6x Arr_J;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
