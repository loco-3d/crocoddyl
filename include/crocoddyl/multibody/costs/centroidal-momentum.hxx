///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const Vector6s& href, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelCentroidalMomentum>(state, href, nu)), href_(href) {
  std::cerr << "Deprecated CostModelCentroidalMomentum: Use ResidualModelCentroidalMomentum with "
               "CostModelResidual class"
            << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const Vector6s& href)
    : Base(state, activation, boost::make_shared<ResidualModelCentroidalMomentum>(state, href)), href_(href) {
  std::cerr << "Deprecated CostModelCentroidalMomentum: Use ResidualModelCentroidalMomentum with "
               "CostModelResidual class"
            << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelCentroidalMomentum>(state, href, nu)), href_(href) {
  std::cerr << "Deprecated CostModelCentroidalMomentum: Use ResidualModelCentroidalMomentum with "
               "CostModelResidual class"
            << std::endl;
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href)
    : Base(state, boost::make_shared<ResidualModelCentroidalMomentum>(state, href)), href_(href) {
  std::cerr << "Deprecated CostModelCentroidalMomentum: Use ResidualModelCentroidalMomentum with "
               "CostModelResidual class"
            << std::endl;
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::~CostModelCentroidalMomentumTpl() {}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(Vector6s)) {
    href_ = *static_cast<const Vector6s*>(pv);
    ResidualModelCentroidalMomentum* residual = static_cast<ResidualModelCentroidalMomentum*>(residual_.get());
    residual->set_reference(href_);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector6s)");
  }
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(Vector6s)) {
    Eigen::Map<Vector6s> ref_map(static_cast<Vector6s*>(pv)->data());
    ResidualModelCentroidalMomentum* residual = static_cast<ResidualModelCentroidalMomentum*>(residual_.get());
    href_ = residual->get_reference();
    ref_map[0] = href_[0];
    ref_map[1] = href_[1];
    ref_map[2] = href_[2];
    ref_map[3] = href_[3];
    ref_map[4] = href_[4];
    ref_map[5] = href_[5];
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector6s)");
  }
}

}  // namespace crocoddyl
