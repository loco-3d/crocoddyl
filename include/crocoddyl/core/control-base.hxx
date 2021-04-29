///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {
template <typename Scalar>
ControlAbstractTpl<Scalar>::ControlAbstractTpl(const std::size_t nu, const std::size_t np)
    : nu_(nu),
      np_(np)
{}

template <typename Scalar>
ControlAbstractTpl<Scalar>::ControlAbstractTpl()
    : nu_(0),
      np_(0)
      {}

template <typename Scalar>
ControlAbstractTpl<Scalar>::~ControlAbstractTpl() {}

template <typename Scalar>
std::size_t ControlAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ControlAbstractTpl<Scalar>::get_np() const {
  return np_;
}

}  // namespace crocoddyl
