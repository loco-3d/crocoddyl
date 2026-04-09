///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef COLMPC_STATES_MULTIBODY_HPP_
#define COLMPC_STATES_MULTIBODY_HPP_

#include <colmpc/fwd.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include "crocoddyl/multibody/states/multibody.hpp"

namespace colmpc {

template <typename _Scalar>
class StateMultibodyTpl : public crocoddyl::StateMultibodyTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::StateMultibodyTpl<Scalar> Base;
  typedef pinocchio::GeometryModel GeometryModel;

  explicit StateMultibodyTpl(
      std::shared_ptr<typename Base::PinocchioModel> model,
      std::shared_ptr<GeometryModel> gmodel)
      : Base(model), geometry_(gmodel) {}
  StateMultibodyTpl() = default;
  virtual ~StateMultibodyTpl() = default;

  const std::shared_ptr<GeometryModel>& get_geometry() const {
    return geometry_;
  }

 private:
  std::shared_ptr<GeometryModel> geometry_;
};

extern template class StateMultibodyTpl<double>;

}  // namespace colmpc

#endif  // COLMPC_STATES_MULTIBODY_HPP_
