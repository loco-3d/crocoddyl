
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ControlParametrizationModelAbstractTpl_wrap
    : public ControlParametrizationModelAbstractTpl<Scalar>,
      public bp::wrapper<ControlParametrizationModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ControlParametrizationModelBase,
                         ControlParametrizationModelAbstractTpl_wrap)

  typedef typename crocoddyl::ControlParametrizationModelAbstractTpl<Scalar>
      ControlParamModel;
  typedef typename crocoddyl::ControlParametrizationDataAbstractTpl<Scalar>
      ControlParamData;
  typedef typename ControlParamModel::VectorXs VectorXs;
  typedef typename ControlParamModel::MatrixXs MatrixXs;
  using ControlParamModel::nu_;
  using ControlParamModel::nw_;

  ControlParametrizationModelAbstractTpl_wrap(std::size_t nw, std::size_t nu)
      : ControlParamModel(nw, nu), bp::wrapper<ControlParamModel>() {}

  void calc(const std::shared_ptr<ControlParamData>& data, const Scalar t,
            const Eigen::Ref<const VectorXs>& u) const override {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, t,
                          (VectorXs)u);
  }

  void calcDiff(const std::shared_ptr<ControlParamData>& data, const Scalar t,
                const Eigen::Ref<const VectorXs>& u) const override {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, t,
                          (VectorXs)u);
  }

  void params(const std::shared_ptr<ControlParamData>& data, const Scalar t,
              const Eigen::Ref<const VectorXs>& w) const override {
    if (static_cast<std::size_t>(w.size()) != nw_) {
      throw_pretty(
          "Invalid argument: " << "w has wrong dimension (it should be " +
                                      std::to_string(nw_) + ")");
    }
    return bp::call<void>(this->get_override("params").ptr(), data, t,
                          (VectorXs)w);
  }

  std::shared_ptr<ControlParamData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ControlParamData>>(createData.ptr());
    }
    return ControlParamModel::createData();
  }

  std::shared_ptr<ControlParamData> default_createData() {
    return this->ControlParamModel::createData();
  }

  void convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
                     const Eigen::Ref<const VectorXs>& w_ub,
                     Eigen::Ref<VectorXs> u_lb,
                     Eigen::Ref<VectorXs> u_ub) const override {
    bp::list res = convertBounds_wrap(w_lb, w_ub);
    u_lb.derived() = bp::extract<VectorXs>(res[0])();
    u_ub.derived() = bp::extract<VectorXs>(res[1])();
  }

  bp::list convertBounds_wrap(const Eigen::Ref<const VectorXs>& w_lb,
                              const Eigen::Ref<const VectorXs>& w_ub) const {
    bp::list p_bounds =
        bp::call<bp::list>(this->get_override("convertBounds").ptr(),
                           (VectorXs)w_lb, (VectorXs)w_ub);
    return p_bounds;
  }

  void multiplyByJacobian(const std::shared_ptr<ControlParamData>& data,
                          const Eigen::Ref<const MatrixXs>& A,
                          Eigen::Ref<MatrixXs> out,
                          const AssignmentOp op) const override {
    switch (op) {
      case setto: {
        out = multiplyByJacobian_wrap(data, A);
        break;
      }
      case addto: {
        out += multiplyByJacobian_wrap(data, A);
        break;
      }
      case rmfrom: {
        out -= multiplyByJacobian_wrap(data, A);
        break;
      }
      default: {
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  MatrixXs multiplyByJacobian_wrap(
      const std::shared_ptr<ControlParamData>& data,
      const Eigen::Ref<const MatrixXs>& A) const {
    return bp::call<MatrixXs>(this->get_override("multiplyByJacobian").ptr(),
                              data, (MatrixXs)A);
  }

  void multiplyJacobianTransposeBy(
      const std::shared_ptr<ControlParamData>& data,
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
      const AssignmentOp op) const override {
    switch (op) {
      case setto: {
        out = multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      case addto: {
        out += multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      case rmfrom: {
        out -= multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      default: {
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  MatrixXs multiplyJacobianTransposeBy_wrap(
      const std::shared_ptr<ControlParamData>& data,
      const Eigen::Ref<const MatrixXs>& A) const {
    return bp::call<MatrixXs>(
        this->get_override("multiplyJacobianTransposeBy").ptr(), data,
        (MatrixXs)A);
  }

  template <typename NewScalar>
  ControlParametrizationModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ControlParametrizationModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(nw_, nu_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
