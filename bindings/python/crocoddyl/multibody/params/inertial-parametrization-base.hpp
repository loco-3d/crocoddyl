///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_

#include "crocoddyl/multibody/params/inertial-parametrization-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class InertialParametrizationAbstractTpl_wrap
    : public InertialParametrizationAbstractTpl<_Scalar>,
      public bp::wrapper<InertialParametrizationAbstractTpl<_Scalar> > {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef InertialParametrizationAbstractTpl<Scalar> Model;
  typedef InertialParametrizationDataAbstractTpl<Scalar> Data;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;

  InertialParametrizationAbstractTpl_wrap() : Model(), bp::wrapper<Model>() {}

  void fromParametrization(const std::shared_ptr<Data>& data,
                           Eigen::Ref<VectorXs> psi,
                           const Eigen::Ref<const VectorXs>& p) override {
    return bp::call<void>(this->get_override("fromParametrization").ptr(), data,
                          boost::ref(psi), (VectorXs)p);
  }

  void toParametrization(Eigen::Ref<VectorXs> p,
                         const Eigen::Ref<const VectorXs>& psi) override {
    return bp::call<void>(this->get_override("toParametrization").ptr(),
                          boost::ref(p), (VectorXs)psi);
  }

  void updateParametrizationDerivative(
      const std::shared_ptr<Data>& data, Eigen::Ref<MatrixXs> dpsi_dp,
      const Eigen::Ref<const VectorXs>& p,
      const Eigen::Ref<const VectorXs>& psi) override {
    return bp::call<void>(
        this->get_override("updateParametrizationDerivative").ptr(), data,
        boost::ref(dpsi_dp), (VectorXs)p, (VectorXs)psi);
  }

  std::shared_ptr<Data> createData() override {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<Data> >(createData.ptr());
    }
    return Model::createData();
  }

  std::shared_ptr<Data> default_createData() {
    return this->Model::createData();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_
