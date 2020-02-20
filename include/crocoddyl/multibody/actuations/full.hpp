///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActuationModelFullTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  explicit ActuationModelFullTpl(boost::shared_ptr<StateMultibody> state) : Base(state, state->get_nv()) {
    pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
    if (state->get_pinocchio().joints[1].shortname() == ff_joint.shortname()) {
      throw_pretty("Invalid argument: "
                   << "the first joint cannot be free-flyer");
    }
  };

  ~ActuationModelFullTpl(){};

  void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& /*x*/,
            const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    data->tau = u;
  };

  void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                const Eigen::Ref<const VectorXs>& /*u*/) {
    // The derivatives has constant values which were set in createData.
    assert_pretty(data->dtau_dx == MatrixXs::Zero(state_->get_nv(), state_->get_ndx()), "dtau_dx has wrong value");
    assert_pretty(data->dtau_du == MatrixXs::Identity(state_->get_nv(), nu_), "dtau_du has wrong value");
  };

  boost::shared_ptr<ActuationDataAbstract> createData() {
    boost::shared_ptr<ActuationDataAbstract> data = boost::make_shared<ActuationDataAbstract>(this);
    data->dtau_du.diagonal().fill(1);
    return data;
  };

 protected:
  using Base::nu_;
  using Base::state_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
