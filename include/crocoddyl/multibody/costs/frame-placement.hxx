///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement, nu)),
      Mref_(Mref) {
  std::cerr << "Deprecated CostModelFramePlacement: Use ResidualModelFramePlacement with CostModelResidual"
            << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref)
    : Base(state, activation, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement)),
      Mref_(Mref) {
  std::cerr << "Deprecated CostModelFramePlacement: Use ResidualModelFramePlacement with CostModelResidual"
            << std::endl;
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement, nu)), Mref_(Mref) {
  std::cerr << "Deprecated CostModelFramePlacement: Use ResidualModelFramePlacement with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref)
    : Base(state, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement)), Mref_(Mref) {
  std::cerr << "Deprecated CostModelFramePlacement: Use ResidualModelFramePlacement with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::~CostModelFramePlacementTpl() {}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FramePlacement)) {
    FramePlacement& ref_map = *static_cast<FramePlacement*>(pv);
    ResidualModelFramePlacement* residual = static_cast<ResidualModelFramePlacement*>(residual_.get());
    Mref_.id = residual->get_id();
    Mref_.placement = residual->get_reference();
    ref_map = Mref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FramePlacement)) {
    Mref_ = *static_cast<const FramePlacement*>(pv);
    ResidualModelFramePlacement* residual = static_cast<ResidualModelFramePlacement*>(residual_.get());
    residual->set_id(Mref_.id);
    residual->set_reference(Mref_.placement);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

}  // namespace crocoddyl
