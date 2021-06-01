///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation, nu)),
      Rref_(Rref) {
  std::cerr << "Deprecated CostModelFrameRotation: Use ResidualModelFrameRotation with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref)
    : Base(state, activation, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation)),
      Rref_(Rref) {
  std::cerr << "Deprecated CostModelFrameRotation: Use ResidualModelFrameRotation with CostModelResidual" << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation, nu)), Rref_(Rref) {
  std::cerr << "Deprecated CostModelFrameRotation: Use ResidualModelFrameRotation with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref)
    : Base(state, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation)), Rref_(Rref) {
  std::cerr << "Deprecated CostModelFrameRotation: Use ResidualModelFrameRotation with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::~CostModelFrameRotationTpl() {}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameRotation)) {
    FrameRotation& ref_map = *static_cast<FrameRotation*>(pv);
    ResidualModelFrameRotation* residual = static_cast<ResidualModelFrameRotation*>(residual_.get());
    Rref_.id = residual->get_id();
    Rref_.rotation = residual->get_reference();
    ref_map = Rref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameRotation)) {
    Rref_ = *static_cast<const FrameRotation*>(pv);
    ResidualModelFrameRotation* residual = static_cast<ResidualModelFrameRotation*>(residual_.get());
    residual->set_id(Rref_.id);
    residual->set_reference(Rref_.rotation);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

}  // namespace crocoddyl
