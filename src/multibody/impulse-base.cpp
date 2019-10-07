///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

ImpulseModelAbstract::ImpulseModelAbstract(StateMultibody& state, const std::size_t& ni) : state_(state), ni_(ni) {}

ImpulseModelAbstract::~ImpulseModelAbstract() {}

void ImpulseModelAbstract::updateForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                           const Eigen::MatrixXd& df_dq) const {
  assert((df_dq.rows() == ni_ || df_dq.cols() == state_.get_nv()) && "df_dq has wrong dimension");
  data->df_dq = df_dq;
}

boost::shared_ptr<ImpulseDataAbstract> ImpulseModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseDataAbstract>(this, data);
}

StateMultibody& ImpulseModelAbstract::get_state() const { return state_; }

const std::size_t& ImpulseModelAbstract::get_ni() const { return ni_; }

}  // namespace crocoddyl
