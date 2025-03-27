///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelAbstractTpl<Scalar>::ResidualModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const std::size_t nu, const bool q_dependent, const bool v_dependent,
    const bool u_dependent)
    : state_(state),
      nr_(nr),
      nu_(nu),
      unone_(VectorXs::Zero(nu)),
      q_dependent_(q_dependent),
      v_dependent_(v_dependent),
      u_dependent_(u_dependent) {}

template <typename Scalar>
ResidualModelAbstractTpl<Scalar>::ResidualModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const bool q_dependent, const bool v_dependent, const bool u_dependent)
    : state_(state),
      nr_(nr),
      nu_(state->get_nv()),
      unone_(VectorXs::Zero(state->get_nv())),
      q_dependent_(q_dependent),
      v_dependent_(v_dependent),
      u_dependent_(u_dependent) {}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>&,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>&,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelAbstractTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<ResidualDataAbstract>(
      Eigen::aligned_allocator<ResidualDataAbstract>(), this, data);
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calcCostDiff(
    const std::shared_ptr<CostDataAbstract>& cdata,
    const std::shared_ptr<ResidualDataAbstract>& rdata,
    const std::shared_ptr<ActivationDataAbstract>& adata, const bool update_u) {
  // This function computes the derivatives of the cost function based on a
  // Gauss-Newton approximation
  const bool is_ru = u_dependent_ && nu_ != 0 && update_u;
  const std::size_t nv = state_->get_nv();
  if (is_ru) {
    cdata->Lu.noalias() = rdata->Ru.transpose() * adata->Ar;
    rdata->Arr_Ru.noalias() = adata->Arr.diagonal().asDiagonal() * rdata->Ru;
    cdata->Luu.noalias() = rdata->Ru.transpose() * rdata->Arr_Ru;
  }
  if (q_dependent_ && v_dependent_) {
    cdata->Lx.noalias() = rdata->Rx.transpose() * adata->Ar;
    rdata->Arr_Rx.noalias() = adata->Arr.diagonal().asDiagonal() * rdata->Rx;
    cdata->Lxx.noalias() = rdata->Rx.transpose() * rdata->Arr_Rx;
    if (is_ru) {
      cdata->Lxu.noalias() = rdata->Rx.transpose() * rdata->Arr_Ru;
    }
  } else if (q_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
        rdata->Rx.leftCols(nv);
    cdata->Lx.head(nv).noalias() = Rq.transpose() * adata->Ar;
    rdata->Arr_Rx.leftCols(nv).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rq;
    cdata->Lxx.topLeftCorner(nv, nv).noalias() =
        Rq.transpose() * rdata->Arr_Rx.leftCols(nv);
    if (is_ru) {
      cdata->Lxu.topRows(nv).noalias() = Rq.transpose() * rdata->Arr_Ru;
    }
  } else if (v_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv =
        rdata->Rx.rightCols(nv);
    cdata->Lx.tail(nv).noalias() = Rv.transpose() * adata->Ar;
    rdata->Arr_Rx.rightCols(nv).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rv;
    cdata->Lxx.bottomRightCorner(nv, nv).noalias() =
        Rv.transpose() * rdata->Arr_Rx.rightCols(nv);
    if (is_ru) {
      cdata->Lxu.bottomRows(nv).noalias() = Rv.transpose() * rdata->Arr_Ru;
    }
  }
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
ResidualModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ResidualModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
std::size_t ResidualModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_q_dependent() const {
  return q_dependent_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_v_dependent() const {
  return v_dependent_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_u_dependent() const {
  return u_dependent_;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ResidualModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
