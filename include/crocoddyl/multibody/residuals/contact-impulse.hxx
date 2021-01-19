///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/residuals/contact-impulse.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactImpulseTpl<Scalar>::ResidualModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id,
                                                                       const Force& fref)
    : Base(state, 6, 0), id_(id), fref_(fref) {}

template <typename Scalar>
ResidualModelContactImpulseTpl<Scalar>::~ResidualModelContactImpulseTpl() {}

template <typename Scalar>
void ResidualModelContactImpulseTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // We transform the impulse to the contact frame
  switch (d->impulse_type) {
    case Impulse3D:
      data->r = (d->impulse->jMf.actInv(d->impulse->f) - fref_).linear();
      break;
    case Impulse6D:
      data->r = (d->impulse->jMf.actInv(d->impulse->f) - fref_).toVector();
      break;
    default:
      break;
  }
}

template <typename Scalar>
void ResidualModelContactImpulseTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->impulse->df_dx;
  switch (d->impulse_type) {
    case Impulse3D:
      data->Rx = df_dx.template topRows<3>();
      break;
    case Impulse6D:
      data->Rx = df_dx;
      break;
    default:
      break;
  }
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelContactImpulseTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactImpulseTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::ForceTpl<Scalar>& ResidualModelContactImpulseTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactImpulseTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactImpulseTpl<Scalar>::set_reference(const Force& reference) {
  fref_ = reference;
}

}  // namespace crocoddyl
