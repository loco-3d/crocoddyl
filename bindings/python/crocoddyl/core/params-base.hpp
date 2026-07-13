///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_PARAMS_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_PARAMS_BASE_HPP_

#include "crocoddyl/core/params-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class ParamsAbstractTpl_wrap : public ParamsAbstractTpl<_Scalar>,
                               public bp::wrapper<ParamsAbstractTpl<_Scalar> > {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsAbstractTpl<Scalar> ParamsModel;
  typedef typename ParamsModel::ParamsDataAbstract ParamsDataAbstract;
  typedef typename ParamsModel::StateAbstract State;
  typedef typename ParamsModel::VectorXs VectorXs;

  ParamsAbstractTpl_wrap(std::shared_ptr<State> state, const std::size_t np = 0)
      : ParamsModel(state, np), bp::wrapper<ParamsModel>() {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (boost::python::override update = this->get_override("update")) {
      return bp::call<void>(update.ptr(), data, (VectorXs)p);
    }
    return ParamsModel::update(data, p);
  }

  void default_update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) {
    return this->ParamsModel::update(data, p);
  }

  std::shared_ptr<ParamsDataAbstract> createData() override {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ParamsDataAbstract> >(createData.ptr());
    }
    return ParamsModel::createData();
  }

  std::shared_ptr<ParamsDataAbstract> default_createData() {
    return this->ParamsModel::createData();
  }
};

template <typename _Scalar>
class ActionModelParamsAbstractTpl_wrap
    : public ActionModelParamsAbstractTpl<_Scalar>,
      public bp::wrapper<ActionModelParamsAbstractTpl<_Scalar> > {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelParamsAbstractTpl<Scalar> ParamsModel;
  typedef typename ParamsModel::ParamsDataAbstract ParamsDataAbstract;
  typedef typename ParamsModel::ActionDataAbstract ActionDataAbstract;
  typedef typename ParamsModel::StateAbstract State;
  typedef typename ParamsModel::VectorXs VectorXs;

  ActionModelParamsAbstractTpl_wrap(std::shared_ptr<State> state,
                                    const std::size_t np = 0)
      : ParamsModel(state, np), bp::wrapper<ParamsModel>() {}

  void update(const std::shared_ptr<ParamsDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& p) override {
    if (boost::python::override update = this->get_override("update")) {
      return bp::call<void>(update.ptr(), data, (VectorXs)p);
    }
    return ParamsModel::update(data, p);
  }

  void default_update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) {
    return this->ParamsModel::update(data, p);
  }

  void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override {
    return bp::call<void>(this->get_override("computeParamSensitivity").ptr(),
                          data, params, (VectorXs)x, (VectorXs)u);
  }

  std::shared_ptr<ParamsDataAbstract> createData() override {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ParamsDataAbstract> >(createData.ptr());
    }
    return ParamsModel::createData();
  }

  std::shared_ptr<ParamsDataAbstract> default_createData() {
    return this->ParamsModel::createData();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_PARAMS_BASE_HPP_
