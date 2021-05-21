///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref, const std::size_t nu)
    : Base(state, activation,
           boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference, nu)),
      vref_(vref) {
  std::cerr << "Deprecated CostModelFrameVelocity: Use ResidualModelFrameVelocity with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameMotion& vref)
    : Base(state, activation,
           boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference)),
      vref_(vref) {
  std::cerr << "Deprecated CostModelFrameVelocity: Use ResidualModelFrameVelocity with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference, nu)),
      vref_(vref) {
  std::cerr << "Deprecated CostModelFrameVelocity: Use ResidualModelFrameVelocity with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameMotion& vref)
    : Base(state, boost::make_shared<ResidualModelFrameVelocity>(state, vref.id, vref.motion, vref.reference)),
      vref_(vref) {
  std::cerr << "Deprecated CostModelFrameVelocity: Use ResidualModelFrameVelocity with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelFrameVelocityTpl<Scalar>::~CostModelFrameVelocityTpl() {}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameMotion)) {
    FrameMotion& ref_map = *static_cast<FrameMotion*>(pv);
    ResidualModelFrameVelocity* residual = static_cast<ResidualModelFrameVelocity*>(residual_.get());
    vref_.id = residual->get_id();
    vref_.motion = residual->get_reference();
    vref_.reference = residual->get_type();
    ref_map = vref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

template <typename Scalar>
void CostModelFrameVelocityTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameMotion)) {
    vref_ = *static_cast<const FrameMotion*>(pv);
    ResidualModelFrameVelocity* residual = static_cast<ResidualModelFrameVelocity*>(residual_.get());
    residual->set_id(vref_.id);
    residual->set_reference(vref_.motion);
    residual->set_type(vref_.reference);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameMotion)");
  }
}

}  // namespace crocoddyl
