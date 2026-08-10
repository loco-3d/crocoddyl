///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelAbstractTpl<Scalar>::TaskModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const std::size_t nu, const bool q_dependent, const bool v_dependent,
    const bool u_dependent, const bool has_acceleration)
    : state_(state),
      nr_(nr),
      nu_(nu),
      q_dependent_(q_dependent),
      v_dependent_(v_dependent),
      u_dependent_(u_dependent),
      has_acceleration_(has_acceleration) {
  if (nr_ == 0) {
    throw_pretty("Invalid argument: task dimension must be positive");
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelAbstractTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<TaskDataAbstract>(
      Eigen::aligned_allocator<TaskDataAbstract>(), this, data);
}

template <typename Scalar>
const std::shared_ptr<typename TaskModelAbstractTpl<Scalar>::StateAbstract>&
TaskModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t TaskModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
std::size_t TaskModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
bool TaskModelAbstractTpl<Scalar>::get_q_dependent() const {
  return q_dependent_;
}

template <typename Scalar>
bool TaskModelAbstractTpl<Scalar>::get_v_dependent() const {
  return v_dependent_;
}

template <typename Scalar>
bool TaskModelAbstractTpl<Scalar>::get_u_dependent() const {
  return u_dependent_;
}

template <typename Scalar>
bool TaskModelAbstractTpl<Scalar>::get_has_acceleration() const {
  return has_acceleration_;
}

template <typename Scalar>
void TaskModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << "TaskModelAbstract {nr=" << nr_ << ", nu=" << nu_ << "}";
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const TaskModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
