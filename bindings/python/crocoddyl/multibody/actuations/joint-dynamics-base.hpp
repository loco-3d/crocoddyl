///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_

#include "crocoddyl/multibody/actuations/joint-dynamics-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class JointDynamicsModelAbstractTpl_wrap
    : public JointDynamicsModelAbstractTpl<Scalar>,
      public bp::wrapper<JointDynamicsModelAbstractTpl<Scalar> > {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(JointDynamicsModelBase,
                         JointDynamicsModelAbstractTpl_wrap)

  typedef JointDynamicsModelAbstractTpl<Scalar> JointDynamicsModelAbstract;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef typename JointDynamicsModelAbstract::VectorXs VectorXs;
  typedef typename JointDynamicsModelAbstract::MatrixXs MatrixXs;
  using JointDynamicsModelAbstract::id_;
  using JointDynamicsModelAbstract::nq_;
  using JointDynamicsModelAbstract::nu_;
  using JointDynamicsModelAbstract::nv_;

  JointDynamicsModelAbstractTpl_wrap(const pinocchio::JointIndex id,
                                     const std::size_t nq, const std::size_t nv,
                                     const std::size_t nu = 0)
      : JointDynamicsModelAbstract(id, nq, nv, nu),
        bp::wrapper<JointDynamicsModelAbstract>() {}

  void calc(const std::shared_ptr<JointDynamicsDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& q,
            const Eigen::Ref<const VectorXs>& v,
            const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)q,
                          (VectorXs)v, (VectorXs)u);
  }

  void calcDiff(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& q,
                const Eigen::Ref<const VectorXs>& v,
                const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)q, (VectorXs)v, (VectorXs)u);
  }

  void commands(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& q,
                const Eigen::Ref<const VectorXs>& v,
                const Eigen::Ref<const VectorXs>& tau) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(tau.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    return bp::call<void>(this->get_override("commands").ptr(), data,
                          (VectorXs)q, (VectorXs)v, (VectorXs)tau);
  }

  std::size_t get_np() const override {
    if (boost::python::override get_np = this->get_override("get_np")) {
      return bp::call<std::size_t>(get_np.ptr());
    }
    return JointDynamicsModelAbstract::get_np();
  }

  std::size_t default_get_np() const {
    return this->JointDynamicsModelAbstract::get_np();
  }

  void set_parameters(const Eigen::Ref<const VectorXs>& p) override {
    if (boost::python::override set_parameters =
            this->get_override("set_parameters")) {
      return bp::call<void>(set_parameters.ptr(), (VectorXs)p);
    }
    return JointDynamicsModelAbstract::set_parameters(p);
  }

  void default_set_parameters(const Eigen::Ref<const VectorXs>& p) {
    return this->JointDynamicsModelAbstract::set_parameters(p);
  }

  VectorXs get_parameters() const override {
    if (boost::python::override get_parameters =
            this->get_override("get_parameters")) {
      VectorXs p = bp::call<VectorXs>(get_parameters.ptr());
      if (static_cast<std::size_t>(p.size()) != get_np()) {
        throw_pretty(
            "Invalid argument: get_parameters returned a vector with "
            "wrong dimension");
      }
      return p;
    }
    return JointDynamicsModelAbstract::get_parameters();
  }

  VectorXs default_get_parameters() const {
    return this->JointDynamicsModelAbstract::get_parameters();
  }

  VectorXs get_parametrization() const override {
    if (boost::python::override get_parametrization =
            this->get_override("get_parametrization")) {
      VectorXs p = bp::call<VectorXs>(get_parametrization.ptr());
      if (static_cast<std::size_t>(p.size()) != get_np()) {
        throw_pretty(
            "Invalid argument: get_parametrization returned a vector with "
            "wrong dimension");
      }
      return p;
    }
    return JointDynamicsModelAbstract::get_parametrization();
  }

  VectorXs default_get_parametrization() const {
    return this->JointDynamicsModelAbstract::get_parametrization();
  }

  void updateParametrizationDerivative(
      Eigen::Ref<MatrixXs> dgamma_dp) const override {
    boost::python::override update_parametrization_derivative =
        this->get_override("updateParametrizationDerivative");
    if (update_parametrization_derivative) {
      bp::object result = bp::call<bp::object>(
          update_parametrization_derivative.ptr(), (MatrixXs)dgamma_dp);
      if (result.ptr() != Py_None) {
        MatrixXs value = bp::extract<MatrixXs>(result)();
        if (value.rows() != dgamma_dp.rows() ||
            value.cols() != dgamma_dp.cols()) {
          throw_pretty(
              "Invalid argument: updateParametrizationDerivative returned a "
              "matrix with wrong dimensions");
        }
        dgamma_dp = value;
      }
      return;
    }
    return JointDynamicsModelAbstract::updateParametrizationDerivative(
        dgamma_dp);
  }

  void default_updateParametrizationDerivative(
      Eigen::Ref<MatrixXs> dgamma_dp) const {
    return this->JointDynamicsModelAbstract::updateParametrizationDerivative(
        dgamma_dp);
  }

  void computeJointTorqueRegressor(
      Eigen::Ref<MatrixXs> joint_dtau_dp, const Eigen::Ref<const VectorXs>& q,
      const Eigen::Ref<const VectorXs>& v,
      const Eigen::Ref<const VectorXs>& u) const override {
    if (boost::python::override compute_joint_torque_regressor =
            this->get_override("computeJointTorqueRegressor")) {
      bp::object result = bp::call<bp::object>(
          compute_joint_torque_regressor.ptr(), (MatrixXs)joint_dtau_dp,
          (VectorXs)q, (VectorXs)v, (VectorXs)u);
      if (result.ptr() != Py_None) {
        MatrixXs value = bp::extract<MatrixXs>(result)();
        if (value.rows() != joint_dtau_dp.rows() ||
            value.cols() != joint_dtau_dp.cols()) {
          throw_pretty(
              "Invalid argument: computeJointTorqueRegressor returned a "
              "matrix with wrong dimensions");
        }
        joint_dtau_dp = value;
      }
      return;
    }
    return JointDynamicsModelAbstract::computeJointTorqueRegressor(
        joint_dtau_dp, q, v, u);
  }

  void default_computeJointTorqueRegressor(
      Eigen::Ref<MatrixXs> joint_dtau_dp, const Eigen::Ref<const VectorXs>& q,
      const Eigen::Ref<const VectorXs>& v,
      const Eigen::Ref<const VectorXs>& u) const {
    return this->JointDynamicsModelAbstract::computeJointTorqueRegressor(
        joint_dtau_dp, q, v, u);
  }

  std::shared_ptr<JointDynamicsDataAbstract> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<JointDynamicsDataAbstract> >(
          createData.ptr());
    }
    return JointDynamicsModelAbstract::createData();
  }

  std::shared_ptr<JointDynamicsDataAbstract> default_createData() {
    return this->JointDynamicsModelAbstract::createData();
  }

  template <typename NewScalar>
  JointDynamicsModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef JointDynamicsModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(id_, nq_, nv_, nu_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_
