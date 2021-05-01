///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-force.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactForceTpl<Scalar>::ResidualModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const pinocchio::FrameIndex id, const Force& fref,
                                                                   const std::size_t nc, const std::size_t nu)
    : Base(state, nc, nu), id_(id), fref_(fref) {
  if (nc > 6) {
    throw_pretty("Invalid argument in ResidualModelContactForce: nc is less than 6");
  }
}

template <typename Scalar>
ResidualModelContactForceTpl<Scalar>::ResidualModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const pinocchio::FrameIndex id, const Force& fref,
                                                                   const std::size_t nc)
    : Base(state, nc), id_(id), fref_(fref) {
  if (nc > 6) {
    throw_pretty("Invalid argument in ResidualModelContactForce: nc is less than 6");
  }
}

template <typename Scalar>
ResidualModelContactForceTpl<Scalar>::~ResidualModelContactForceTpl() {}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // We transform the force to the contact frame
  switch (d->contact_type) {
    case Contact3D:
      data->r = (d->contact->jMf.actInv(d->contact->f) - fref_).linear();
      break;
    case Contact6D:
      data->r = (d->contact->jMf.actInv(d->contact->f) - fref_).toVector();
      break;
    default:
      break;
  }
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                    const Eigen::Ref<const VectorXs>&,
                                                    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  switch (d->contact_type) {
    case Contact3D:
      data->Rx = df_dx.template topRows<3>();
      data->Ru = df_du.template topRows<3>();
      break;
    case Contact6D:
      data->Rx = df_dx;
      data->Ru = df_du;
      break;
    default:
      break;
  }
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelContactForceTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactForceTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::ForceTpl<Scalar>& ResidualModelContactForceTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::set_reference(const Force& reference) {
  fref_ = reference;
}

}  // namespace crocoddyl
