///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/contact-impulse.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FrameForce& fref)
    : Base(state, activation,
           boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, activation->get_nr(), 0)),
      fref_(fref) {
  std::cerr << "Deprecated CostModelContactImpulse: Use ResidualModelContactForce with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref, const std::size_t nr)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, nr, 0)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactImpulse: Use ResidualModelContactForce with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, 6, 0)), fref_(fref) {
  std::cerr << "Deprecated CostModelContactImpulse: Use ResidualModelContactForce with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::~CostModelContactImpulseTpl() {}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameForce)) {
    fref_ = *static_cast<const FrameForce*>(pv);
    ResidualModelContactForce* residual = static_cast<ResidualModelContactForce*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.force);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameForce)) {
    FrameForce& ref_map = *static_cast<FrameForce*>(pv);
    ResidualModelContactForce* residual = static_cast<ResidualModelContactForce*>(residual_.get());
    fref_.id = residual->get_id();
    fref_.force = residual->get_reference();
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

}  // namespace crocoddyl
