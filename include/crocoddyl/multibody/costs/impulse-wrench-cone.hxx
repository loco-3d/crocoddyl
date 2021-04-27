///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameWrenchCone& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, 0)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelImpulseWrenchCone: Use ResidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
  if (activation_->get_nr() != fref_.cone.get_nf() + 13) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::CostModelImpulseWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const FrameWrenchCone& fref)
    : Base(state, boost::make_shared<ResidualModelContactWrenchCone>(state, fref.id, fref.cone, 0)), fref_(fref) {
  std::cerr << "Deprecated CostModelImpulseWrenchCone: Use ResidualModelContactWrenchCone with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelImpulseWrenchConeTpl<Scalar>::~CostModelImpulseWrenchConeTpl() {}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    fref_ = *static_cast<const FrameWrenchCone*>(pv);
    ResidualModelContactWrenchCone* residual = static_cast<ResidualModelContactWrenchCone*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.cone);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

template <typename Scalar>
void CostModelImpulseWrenchConeTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameWrenchCone)) {
    FrameWrenchCone& ref_map = *static_cast<FrameWrenchCone*>(pv);
    ResidualModelContactWrenchCone* residual = static_cast<ResidualModelContactWrenchCone*>(residual_.get());
    fref_.id = residual->get_id();
    fref_.cone = residual->get_reference();
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameWrenchCone)");
  }
}

}  // namespace crocoddyl
