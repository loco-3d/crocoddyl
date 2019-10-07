///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelCoMPosition : public CostModelAbstract {
 public:
  CostModelCoMPosition(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::Vector3d& cref,
                       const std::size_t& nu);
  CostModelCoMPosition(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::Vector3d& cref);
  CostModelCoMPosition(StateMultibody& state, const Eigen::Vector3d& cref, const std::size_t& nu);
  CostModelCoMPosition(StateMultibody& state, const Eigen::Vector3d& cref);
  ~CostModelCoMPosition();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const Eigen::VectorXd& get_cref() const;

 private:
  Eigen::VectorXd cref_;
};

struct CostDataCoMPosition : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataCoMPosition(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data), Arr_Jcom(3, model->get_state().get_nv()) {
    Arr_Jcom.fill(0);
  }

  pinocchio::Data::Matrix3x Arr_Jcom;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
