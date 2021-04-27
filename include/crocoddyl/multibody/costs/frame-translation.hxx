///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation, nu)),
      xref_(xref) {
  std::cerr << "Deprecated CostModelFrameTranslation: Use ResidualModelFrameTranslation with CostModelResidual"
            << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref)
    : Base(state, activation, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation)),
      xref_(xref) {
  std::cerr << "Deprecated CostModelFrameTranslation: Use ResidualModelFrameTranslation with CostModelResidual"
            << std::endl;
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation, nu)),
      xref_(xref) {
  std::cerr << "Deprecated CostModelFrameTranslation: Use ResidualModelFrameTranslation with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref)
    : Base(state, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation)), xref_(xref) {
  std::cerr << "Deprecated CostModelFrameTranslation: Use ResidualModelFrameTranslation with CostModelResidual"
            << std::endl;
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::~CostModelFrameTranslationTpl() {}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameTranslation)) {
    FrameTranslation& ref_map = *static_cast<FrameTranslation*>(pv);
    ResidualModelFrameTranslation* residual = static_cast<ResidualModelFrameTranslation*>(residual_.get());
    xref_.id = residual->get_id();
    xref_.translation = residual->get_reference();
    ref_map = xref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameTranslation)");
  }
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameTranslation)) {
    xref_ = *static_cast<const FrameTranslation*>(pv);
    ResidualModelFrameTranslation* residual = static_cast<ResidualModelFrameTranslation*>(residual_.get());
    residual->set_id(xref_.id);
    residual->set_reference(xref_.translation);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameTranslation)");
  }
}

}  // namespace crocoddyl
