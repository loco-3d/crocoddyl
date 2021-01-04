///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/actuation.hpp"

namespace crocoddyl {

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::ActuationModelNumDiffTpl(boost::shared_ptr<Base> model)
    : Base(model->get_state(), model->get_nu()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ActuationModelNumDiffTpl<Scalar>::~ActuationModelNumDiffTpl() {}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);
  model_->calc(data_nd->data_0, x, u);
  data->tau = data_nd->data_0->tau;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x,
                                                const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != model_->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(model_->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);

  const VectorXs& tau0 = data_nd->data_0->tau;

  // Computing the d Actuation(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < model_->get_state()->get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);
    data_nd->dtau_dx.col(ix) = (data_nd->data_x[ix]->tau - tau0) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d Actuation(x,u) / du
  data_nd->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    data_nd->du(iu) = disturbance_;
    model_->calc(data_nd->data_u[iu], x, u + data_nd->du);
    data_nd->dtau_du.col(iu) = (data_nd->data_u[iu]->tau - tau0) / disturbance_;
    data_nd->du(iu) = 0.0;
  }
}

template <typename Scalar>
boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > ActuationModelNumDiffTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >& ActuationModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ActuationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ActuationModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
