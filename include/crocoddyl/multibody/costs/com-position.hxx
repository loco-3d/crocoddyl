///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const Vector3s& cref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelCoMPosition>(state, cref, nu)), cref_(cref) {
  std::cerr << "Deprecated CostModelCoMPosition: Use ResidualModelCoMPosition with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const Vector3s& cref)
    : Base(state, activation, boost::make_shared<ResidualModelCoMPosition>(state, cref)), cref_(cref) {
  std::cerr << "Deprecated CostModelCoMPosition: Use ResidualModelCoMPosition with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref,
                                                         const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelCoMPosition>(state, cref, nu)), cref_(cref) {
  std::cerr << "Deprecated CostModelCoMPosition: Use ResidualModelCoMPosition with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref)
    : Base(state, boost::make_shared<ResidualModelCoMPosition>(state, cref)), cref_(cref) {
  std::cerr << "Deprecated CostModelCoMPosition: Use ResidualModelCoMPosition with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::~CostModelCoMPositionTpl() {}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(Vector3s)) {
    cref_ = *static_cast<const Vector3s*>(pv);
    ResidualModelCoMPosition* residual = static_cast<ResidualModelCoMPosition*>(residual_.get());
    residual->set_reference(cref_);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(Vector3s)) {
    Eigen::Map<Vector3s> ref_map(static_cast<Vector3s*>(pv)->data());
    ResidualModelCoMPosition* residual = static_cast<ResidualModelCoMPosition*>(residual_.get());
    cref_ = residual->get_reference();
    ref_map[0] = cref_[0];
    ref_map[1] = cref_[1];
    ref_map[2] = cref_[2];
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

}  // namespace crocoddyl
