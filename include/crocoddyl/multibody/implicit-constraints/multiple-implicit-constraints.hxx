///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/implicit-constraints/kinematic-loop.hpp"

namespace crocoddyl {

template <typename Scalar>
ImplicitConstraintModelMultipleTpl<Scalar>::ImplicitConstraintModelMultipleTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nu)
    : state_(state),
      nc_(0),
      nc_total_(0),
      nu_(nu),
      compute_all_constraints_(false) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state cannot be null");
  }
}

template <typename Scalar>
ImplicitConstraintModelMultipleTpl<Scalar>::ImplicitConstraintModelMultipleTpl(
    std::shared_ptr<StateMultibody> state)
    : state_(state),
      nc_(0),
      nc_total_(0),
      nu_(state != nullptr ? state->get_nv() : 0),
      compute_all_constraints_(false) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state cannot be null");
  }
}

template <typename Scalar>
ImplicitConstraintModelMultipleTpl<Scalar>::ImplicitConstraintModelMultipleTpl(
    const ImplicitConstraintModelMultipleTpl& other)
    : state_(other.state_),
      nc_(other.nc_),
      nc_total_(other.nc_total_),
      nu_(other.nu_),
      active_set_(other.active_set_),
      inactive_set_(other.inactive_set_),
      compute_all_constraints_(other.compute_all_constraints_) {
  typename ImplicitConstraintModelContainer::const_iterator it, end;
  for (it = other.constraints_.begin(), end = other.constraints_.end();
       it != end; ++it) {
    constraints_.insert(std::make_pair(
        it->first, std::make_shared<ImplicitConstraintItem>(*it->second)));
  }
}

template <typename Scalar>
ImplicitConstraintModelMultipleTpl<
    Scalar>::~ImplicitConstraintModelMultipleTpl() {}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::addConstraint(
    const std::string& name,
    std::shared_ptr<ImplicitConstraintModelAbstract> constraint,
    const bool active) {
  if (constraint == nullptr) {
    throw_pretty("Invalid argument: constraint cannot be null");
  }
  if (constraint->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << name
                 << " constraint item doesn't have the same control "
                    "dimension (" +
                        std::to_string(nu_) + ")");
  }
  if (constraint->get_state()->get_nx() != state_->get_nx() ||
      constraint->get_state()->get_ndx() != state_->get_ndx() ||
      constraint->get_state()->get_pinocchio()->nq !=
          state_->get_pinocchio()->nq ||
      constraint->get_state()->get_pinocchio()->nv !=
          state_->get_pinocchio()->nv ||
      constraint->get_state()->get_pinocchio()->njoints !=
          state_->get_pinocchio()->njoints) {
    throw_pretty("Invalid argument: "
                 << name << " constraint item has an incompatible state");
  }
  std::pair<typename ImplicitConstraintModelContainer::iterator, bool> ret =
      constraints_.insert(std::make_pair(
          name,
          std::make_shared<ImplicitConstraintItem>(name, constraint, active)));
  if (ret.second == false) {
    std::cerr << "Warning: we couldn't add the " << name
              << " constraint item, it already existed." << std::endl;
  } else if (active) {
    nc_ += constraint->get_nc();
    nc_total_ += constraint->get_nc();
    active_set_.insert(name);
  } else {
    nc_total_ += constraint->get_nc();
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::removeConstraint(
    const std::string& name) {
  typename ImplicitConstraintModelContainer::iterator it =
      constraints_.find(name);
  if (it != constraints_.end()) {
    const std::size_t nc_i = it->second->constraint->get_nc();
    if (it->second->active) {
      nc_ -= nc_i;
    }
    nc_total_ -= nc_i;
    constraints_.erase(it);
    active_set_.erase(name);
    inactive_set_.erase(name);
  } else {
    std::cerr << "Warning: we couldn't remove the " << name
              << " constraint item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::changeConstraintStatus(
    const std::string& name, const bool active) {
  typename ImplicitConstraintModelContainer::iterator it =
      constraints_.find(name);
  if (it != constraints_.end()) {
    if (active && !it->second->active) {
      nc_ += it->second->constraint->get_nc();
      active_set_.insert(name);
      inactive_set_.erase(name);
    } else if (!active && it->second->active) {
      nc_ -= it->second->constraint->get_nc();
      inactive_set_.insert(name);
      active_set_.erase(name);
    }
    it->second->active = active;
  } else {
    std::cerr << "Warning: we couldn't change the status of the " << name
              << " constraint item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::calc(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and "
                    "models");
  }

  data->Jc.setZero();
  data->a0.setZero();

  std::size_t nc = 0;
  const std::size_t nv = state_->get_nv();
  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  typename ImplicitConstraintDataContainer::const_iterator it_d, end_d;
  if (compute_all_constraints_) {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      const std::size_t nc_i = m_i->constraint->get_nc();
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between model and "
                    "data ("
                        << it_m->first << " != " << it_d->first << ")");
      if (m_i->active) {
        m_i->constraint->calc(d_i, x);
        data->a0.segment(nc, nc_i) = d_i->a0;
        data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
      }
      nc += nc_i;
    }
  } else {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between model and "
                    "data ("
                        << it_m->first << " != " << it_d->first << ")");
      if (m_i->active) {
        const std::size_t nc_i = m_i->constraint->get_nc();
        m_i->constraint->calc(d_i, x);
        data->a0.segment(nc, nc_i) = d_i->a0;
        data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
        nc += nc_i;
      }
    }
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and "
                    "models");
  }

  data->da0_dx.setZero();
  data->dv0_dq.setZero();

  std::size_t nc = 0;
  const std::size_t ndx = state_->get_ndx();
  const std::size_t nv = state_->get_nv();
  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  typename ImplicitConstraintDataContainer::const_iterator it_d, end_d;
  if (compute_all_constraints_) {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      const std::size_t nc_i = m_i->constraint->get_nc();
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between model and "
                    "data ("
                        << it_m->first << " != " << it_d->first << ")");
      if (m_i->active) {
        m_i->constraint->calcDiff(d_i, x);
        data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
        data->dv0_dq.block(nc, 0, nc_i, nv) = d_i->dv0_dq;
      }
      nc += nc_i;
    }
  } else {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between model and "
                    "data ("
                        << it_m->first << " != " << it_d->first << ")");
      if (m_i->active) {
        const std::size_t nc_i = m_i->constraint->get_nc();
        m_i->constraint->calcDiff(d_i, x);
        data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
        data->dv0_dq.block(nc, 0, nc_i, nv) = d_i->dv0_dq;
        nc += nc_i;
      }
    }
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateVelocity(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const VectorXs& vnext) const {
  if (static_cast<std::size_t>(vnext.size()) != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "vnext has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + ")");
  }
  data->vnext = vnext;
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateAcceleration(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const VectorXs& dv) const {
  if (static_cast<std::size_t>(dv.size()) != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "dv has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + ")");
  }
  data->dv = dv;
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateForce(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) !=
      (compute_all_constraints_ ? nc_total_ : nc_)) {
    throw_pretty(
        "Invalid argument: "
        << "force has wrong dimension (it should be " +
               std::to_string((compute_all_constraints_ ? nc_total_ : nc_)) +
               ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and "
                    "models");
  }

  for (typename pinocchio::container::aligned_vector<
           pinocchio::ForceTpl<Scalar> >::iterator it = data->fext.begin();
       it != data->fext.end(); ++it) {
    *it = pinocchio::ForceTpl<Scalar>::Zero();
  }

  std::size_t nc = 0;
  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  typename ImplicitConstraintDataContainer::const_iterator it_d, end_d;
  if (compute_all_constraints_) {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      const std::size_t nc_i = m_i->constraint->get_nc();
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between data and "
                    "model");
      if (m_i->active) {
        const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i =
            force.segment(nc, nc_i);
        m_i->constraint->updateForce(d_i, force_i);
        const KinematicLoopModelTpl<Scalar>* kinematic_loop =
            dynamic_cast<const KinematicLoopModelTpl<Scalar>*>(
                m_i->constraint.get());
        if (kinematic_loop != NULL) {
          const KinematicLoopDataTpl<Scalar>* d_loop =
              static_cast<const KinematicLoopDataTpl<Scalar>*>(d_i.get());
          data->fext[kinematic_loop->get_joint1_id()] += d_loop->joint1_f;
          data->fext[kinematic_loop->get_joint2_id()] += d_loop->joint2_f;
        } else {
          const pinocchio::JointIndex joint =
              state_->get_pinocchio()->frames[d_i->frame].parentJoint;
          data->fext[joint] += d_i->fext;
        }
      } else {
        m_i->constraint->setZeroForce(d_i);
      }
      nc += nc_i;
    }
  } else {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between data and "
                    "model");
      if (m_i->active) {
        const std::size_t nc_i = m_i->constraint->get_nc();
        const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i =
            force.segment(nc, nc_i);
        m_i->constraint->updateForce(d_i, force_i);
        const KinematicLoopModelTpl<Scalar>* kinematic_loop =
            dynamic_cast<const KinematicLoopModelTpl<Scalar>*>(
                m_i->constraint.get());
        if (kinematic_loop != NULL) {
          const KinematicLoopDataTpl<Scalar>* d_loop =
              static_cast<const KinematicLoopDataTpl<Scalar>*>(d_i.get());
          data->fext[kinematic_loop->get_joint1_id()] += d_loop->joint1_f;
          data->fext[kinematic_loop->get_joint2_id()] += d_loop->joint2_f;
        } else {
          const pinocchio::JointIndex joint =
              state_->get_pinocchio()->frames[d_i->frame].parentJoint;
          data->fext[joint] += d_i->fext;
        }
        nc += nc_i;
      } else {
        m_i->constraint->setZeroForce(d_i);
      }
    }
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateVelocityDiff(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const MatrixXs& dvnext_dx) const {
  if (static_cast<std::size_t>(dvnext_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(dvnext_dx.cols()) != state_->get_ndx()) {
    throw_pretty(
        "Invalid argument: " << "dvnext_dx has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + "," +
                                    std::to_string(state_->get_ndx()) + ")");
  }
  data->dvnext_dx = dvnext_dx;
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateAccelerationDiff(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const MatrixXs& ddv_dx) const {
  if (static_cast<std::size_t>(ddv_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(ddv_dx.cols()) != state_->get_ndx()) {
    throw_pretty(
        "Invalid argument: " << "ddv_dx has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + "," +
                                    std::to_string(state_->get_ndx()) + ")");
  }
  data->ddv_dx = ddv_dx;
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    const MatrixXs& df_dx, const MatrixXs& df_du) const {
  const std::size_t ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) !=
          (compute_all_constraints_ ? nc_total_ : nc_) ||
      static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw_pretty(
        "Invalid argument: "
        << "df_dx has wrong dimension (it should be " +
               std::to_string((compute_all_constraints_ ? nc_total_ : nc_)) +
               "," + std::to_string(ndx) + ")");
  }
  if (static_cast<std::size_t>(df_du.rows()) !=
          (compute_all_constraints_ ? nc_total_ : nc_) ||
      static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "df_du has wrong dimension (it should be " +
               std::to_string((compute_all_constraints_ ? nc_total_ : nc_)) +
               "," + std::to_string(nu_) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and "
                    "models");
  }

  std::size_t nc = 0;
  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  typename ImplicitConstraintDataContainer::const_iterator it_d, end_d;
  if (compute_all_constraints_) {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      const std::size_t nc_i = m_i->constraint->get_nc();
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between data and "
                    "model");
      if (m_i->active) {
        const Eigen::Block<const MatrixXs> df_dx_i =
            df_dx.block(nc, 0, nc_i, ndx);
        const Eigen::Block<const MatrixXs> df_du_i =
            df_du.block(nc, 0, nc_i, nu_);
        m_i->constraint->updateForceDiff(d_i, df_dx_i, df_du_i);
      } else {
        m_i->constraint->setZeroForceDiff(d_i);
      }
      nc += nc_i;
    }
  } else {
    for (it_m = constraints_.begin(), end_m = constraints_.end(),
        it_d = data->constraints.begin(), end_d = data->constraints.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
      const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the constraint name between data and "
                    "model");
      if (m_i->active) {
        const std::size_t nc_i = m_i->constraint->get_nc();
        const Eigen::Block<const MatrixXs> df_dx_i =
            df_dx.block(nc, 0, nc_i, ndx);
        const Eigen::Block<const MatrixXs> df_du_i =
            df_du.block(nc, 0, nc_i, nu_);
        m_i->constraint->updateForceDiff(d_i, df_dx_i, df_du_i);
        nc += nc_i;
      } else {
        m_i->constraint->setZeroForceDiff(d_i);
      }
    }
  }
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::updateRneaDiff(
    const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
    pinocchio::DataTpl<Scalar>& pinocchio) const {
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and "
                    "models");
  }

  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  typename ImplicitConstraintDataContainer::const_iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(),
      it_d = data->constraints.begin(), end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImplicitConstraintItem>& m_i = it_m->second;
    const std::shared_ptr<ImplicitConstraintDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first,
                  "it doesn't match the constraint name between data and "
                  "model");
    if (m_i->active) {
      pinocchio.dtau_dq += d_i->dtau_dq;
    }
  }
}

template <typename Scalar>
std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> >
ImplicitConstraintModelMultipleTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  if (data == nullptr) {
    throw_pretty("Invalid argument: Pinocchio data cannot be null");
  }
  return std::allocate_shared<ImplicitConstraintDataMultiple>(
      Eigen::aligned_allocator<ImplicitConstraintDataMultiple>(), this, data);
}

template <typename Scalar>
template <typename NewScalar>
ImplicitConstraintModelMultipleTpl<NewScalar>
ImplicitConstraintModelMultipleTpl<Scalar>::cast() const {
  typedef ImplicitConstraintModelMultipleTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ImplicitConstraintItemTpl<NewScalar> ConstraintType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), nu_);
  ret.setComputeAllConstraints(compute_all_constraints_);
  typename ImplicitConstraintModelContainer::const_iterator it_m, end_m;
  for (it_m = constraints_.begin(), end_m = constraints_.end(); it_m != end_m;
       ++it_m) {
    const std::string name = it_m->first;
    const ConstraintType& m_i = it_m->second->template cast<NewScalar>();
    ret.addConstraint(name, m_i.get_constraint(), m_i.get_active());
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ImplicitConstraintModelMultipleTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ImplicitConstraintModelMultipleTpl<
    Scalar>::ImplicitConstraintModelContainer&
ImplicitConstraintModelMultipleTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
std::size_t ImplicitConstraintModelMultipleTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImplicitConstraintModelMultipleTpl<Scalar>::get_nc_total() const {
  return nc_total_;
}

template <typename Scalar>
std::size_t ImplicitConstraintModelMultipleTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::set<std::string>&
ImplicitConstraintModelMultipleTpl<Scalar>::get_active_set() const {
  return active_set_;
}

template <typename Scalar>
const std::set<std::string>&
ImplicitConstraintModelMultipleTpl<Scalar>::get_inactive_set() const {
  return inactive_set_;
}

template <typename Scalar>
bool ImplicitConstraintModelMultipleTpl<Scalar>::getConstraintStatus(
    const std::string& name) const {
  typename ImplicitConstraintModelContainer::const_iterator it =
      constraints_.find(name);
  if (it != constraints_.end()) {
    return it->second->active;
  } else {
    std::cerr << "Warning: we couldn't get the status of the " << name
              << " constraint item, it doesn't exist." << std::endl;
    return false;
  }
}

template <typename Scalar>
bool ImplicitConstraintModelMultipleTpl<Scalar>::getComputeAllConstraints()
    const {
  return compute_all_constraints_;
}

template <typename Scalar>
void ImplicitConstraintModelMultipleTpl<Scalar>::setComputeAllConstraints(
    const bool status) {
  compute_all_constraints_ = status;
}

template <class Scalar>
std::ostream& operator<<(
    std::ostream& os, const ImplicitConstraintModelMultipleTpl<Scalar>& model) {
  const auto& active = model.get_active_set();
  const auto& inactive = model.get_inactive_set();
  os << "ImplicitConstraintModelMultiple:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::set<std::string>::const_iterator it = active.begin();
       it != active.end(); ++it) {
    const std::shared_ptr<typename ImplicitConstraintModelMultipleTpl<
        Scalar>::ImplicitConstraintItem>& constraint_item =
        model.get_constraints().find(*it)->second;
    os << "    " << *it << ": " << *constraint_item << std::endl;
  }
  os << "  Inactive:" << std::endl;
  for (std::set<std::string>::const_iterator it = inactive.begin();
       it != inactive.end(); ++it) {
    const std::shared_ptr<typename ImplicitConstraintModelMultipleTpl<
        Scalar>::ImplicitConstraintItem>& constraint_item =
        model.get_constraints().find(*it)->second;
    os << "    " << *it << ": " << *constraint_item << std::endl;
  }
  return os;
}

}  // namespace crocoddyl
