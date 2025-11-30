namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::
    DifferentialActionModelFreeThrustFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<CostModelSum> costs, std::vector<Rotor> rotors,
        std::shared_ptr<ConstraintModelManager> constraints)
    : DifferentialActionModelFreeFwdDynamicsTpl<Scalar>(state, actuation, costs,
                                                        constraints),
      actuation_(actuation),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()),
      rotors_(rotors),
      n_thrusts_(rotors.size()) {
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }

  // Set control limits
  // thrust part
  VectorXs u_lb = VectorXs::Zero(nu_);
  VectorXs u_ub = VectorXs::Zero(nu_);
  for (int i = 0; i < n_thrusts_; ++i) {
    u_lb(i) = rotors[i].min_thrust_;
    u_ub(i) = rotors[i].max_thrust_;
  }

  // joint torque part
  u_lb.tail(nu_ - n_thrusts_) =
      Scalar(-1.) * pinocchio_->effortLimit.tail(nu_ - n_thrusts_);
  u_ub.tail(nu_ - n_thrusts_) =
      Scalar(+1.) * pinocchio_->effortLimit.tail(nu_ - n_thrusts_);

  Base::set_u_lb(u_lb);
  Base::set_u_ub(u_ub);
}

template <typename Scalar>
void DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  // calculate the effect of thrusts on the system
  computeFExtByThrusts(u, d->fext);

  // Computing the dynamics using ABA
  VectorXs tau = VectorXs::Zero(state_->get_nv());
  tau.tail(nu_ - n_thrusts_) = u.tail(nu_ - n_thrusts_);
  d->xout = pinocchio::aba(*pinocchio_, d->pinocchio, q, v, tau, d->fext);
  pinocchio::updateGlobalPlacements(*pinocchio_, d->pinocchio);

  d->multibody.joint->a = d->xout;
  d->multibody.joint->tau = u.tail(nu_ - n_thrusts_);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  pinocchio::computeJointKinematicHessians(*pinocchio_, d->pinocchio, q);

  // Computing the dynamics derivatives
  VectorXs tau = VectorXs::Zero(state_->get_nv());
  tau.tail(nu_ - n_thrusts_) = u.tail(nu_ - n_thrusts_);
  pinocchio::computeABADerivatives(*pinocchio_, d->pinocchio, q, v, tau,
                                   d->Fx.leftCols(nv), d->Fx.rightCols(nv),
                                   d->pinocchio.Minv);

  // derivatives w.r.t joint torques
  d->Fu.rightCols(nu_ - n_thrusts_).noalias() = d->pinocchio.Minv.rightCols(
      nu_ -
      n_thrusts_);  // TODO: We now assume dtau_du is identity for joint part

  // i-th rotor
  for (int i = 0; i < n_thrusts_; i++) {
    pinocchio::JointIndex rotor_parent_joint_index =
        pinocchio_->frames[rotors_[i].frame_id_].parent;

    // rotor frame Jacobian
    MatrixXs rotor_i_jacobian = MatrixXs::Zero(6, nv);
    pinocchio::computeFrameJacobian(*pinocchio_, d->pinocchio, q,
                                    rotors_[i].frame_id_, pinocchio::LOCAL,
                                    rotor_i_jacobian);

    // derivative w.r.t. thrusts
    d->Fu.col(i).noalias() = d->pinocchio.Minv * rotor_i_jacobian.transpose() *
                             rotors_[i].thrust_wrench_unit_.toVector();

    d->joint_hessian_tmp.setZero();
    pinocchio::getJointKinematicHessian(*pinocchio_, d->pinocchio,
                                        rotor_parent_joint_index,
                                        pinocchio::LOCAL, d->joint_hessian_tmp);

    // j-th joint
    for (int j = 0; j < nv; j++) {
      const Scalar* ptr = d->joint_hessian_tmp.data() + j * 6 * nv;
      Eigen::Map<const Eigen::Matrix<Scalar, 6, Eigen::Dynamic> >
          rotor_i_parent_joint_hessian_j(ptr, 6, nv);
      d->Fx.col(j).noalias() +=
          d->pinocchio.Minv * rotor_i_parent_joint_hessian_j.transpose() *
          rotors_[i].thrust_wrench_unit_parent_joint_.toVector() * u(i);
    }
  }

  d->multibody.joint->da_dx = d->Fx;
  d->multibody.joint->da_du = d->Fu;
  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::
    computeFExtByThrusts(const Eigen::Ref<const VectorXs>& u,
                         pinocchio::container::aligned_vector<Force>& fext) {
  // calculate the effect of thrusts on the system
  for (int i = 0; i < n_thrusts_; i++) {
    pinocchio::JointIndex rotor_parent_joint_index =
        pinocchio_->frames[rotors_[i].frame_id_].parent;

    fext.at(rotor_parent_joint_index) =
        rotors_[i].thrust_wrench_unit_parent_joint_ *
        u(i);  // TODO: We now assume only one rotor per joint
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelFreeThrustFwdDynamicsTpl<NewScalar>
DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelFreeThrustFwdDynamicsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef RotorTpl<NewScalar> RotorType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  std::vector<RotorType> rotors = vector_cast<NewScalar>(rotors_);
  if (constraints_) {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()), rotors,
        std::make_shared<ConstraintType>(
            constraints_->template cast<NewScalar>()));
    return ret;
  } else {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()), rotors);
    return ret;
  }
}

template <typename Scalar>
bool DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelFreeThrustFwdDynamicsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
    const std::size_t, const Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();

  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::rnea(
      *pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
      d->tmp_xstatic.tail(
          nv));  // compute tau due to gravity. result is in d->pinocchio.tau

  MatrixXs dtau_du = MatrixXs::Zero(nv, nu_);
  for (int i = 0; i < n_thrusts_; i++) {
    // thrust wrench units
    Force thrust_wrench_unit;
    thrust_wrench_unit.linear() = Vector3s(0, 0, 1);
    if (rotors_[i].direction_ == CLOCKWISE)
      thrust_wrench_unit.angular() = Vector3s(0, 0, rotors_[i].ctorque_);
    else
      thrust_wrench_unit.angular() = Vector3s(0, 0, -rotors_[i].ctorque_);

    // rotor frame Jacobian
    MatrixXs rotor_i_jacobian = MatrixXs::Zero(6, nv);
    pinocchio::computeFrameJacobian(*pinocchio_, d->pinocchio, q,
                                    rotors_[i].frame_id_, pinocchio::LOCAL,
                                    rotor_i_jacobian);

    dtau_du.col(i).noalias() =
        rotor_i_jacobian.transpose() * thrust_wrench_unit.toVector();
  }

  dtau_du.bottomRightCorner(nu_ - n_thrusts_, nu_ - n_thrusts_)
      .diagonal()
      .setOnes();  // joint torque part
  u.noalias() = pseudoInverse(dtau_du) * d->pinocchio.tau;
  d->pinocchio.tau.setZero();
}
}  // namespace crocoddyl
