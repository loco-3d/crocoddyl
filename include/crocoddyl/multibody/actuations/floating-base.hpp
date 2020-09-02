///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
template <typename _Scalar>
class ActuationModelFloatingBaseTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActuationModelFloatingBaseTpl(boost::shared_ptr<StateMultibody> state)
      : Base(state, state->get_nv() - state->get_pinocchio()->joints[1].nv()){};

  virtual ~ActuationModelFloatingBaseTpl(){};

  virtual void calc(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    data->tau.tail(nu_) = u;
  };

#ifndef NDEBUG
  virtual void calcDiff(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>& /*u*/) {
#else
  virtual void calcDiff(const boost::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>& /*u*/) {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(data->dtau_dx == MatrixXs::Zero(state_->get_nv(), state_->get_ndx()), "dtau_dx has wrong value");
    assert_pretty(data->dtau_du == dtau_du_, "dtau_du has wrong value");
  };
  boost::shared_ptr<StateMultibody> state = boost::dynamic_pointer_cast<StateMultibody> (state_);
  virtual boost::shared_ptr<Data> createData() {
    boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    std::size_t fb_nv = state->get_pinocchio()->joints[1].nv();
    data->dtau_du.diagonal(-fb_nv).fill((Scalar)1.);
#ifndef NDEBUG
    dtau_du_ = data->dtau_du;
#endif
    return data;
  };

 protected:
  using Base::nu_;
  using Base::state_;

#ifndef NDEBUG
 private:
  MatrixXs dtau_du_;
#endif
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
