///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelContactInvDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactInvDynamicsTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelContactInvDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelAbstract> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs);

  DifferentialActionModelContactInvDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelAbstract> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs,
                                               boost::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelContactInvDynamicsTpl();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract> &data);
  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract> &data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs> &x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  const boost::shared_ptr<ActuationModelAbstract> &get_actuation() const;
  const boost::shared_ptr<ContactModelMultiple> &get_contacts() const;
  const boost::shared_ptr<CostModelSum> &get_costs() const;
  const boost::shared_ptr<ConstraintModelManager> &get_constraints() const;
  pinocchio::ModelTpl<Scalar> &get_pinocchio() const;
  virtual void print(std::ostream &os) const;

 protected:
  using Base::ng_;     //!< Number of inequality constraints
  using Base::nh_;     //!< Number of equality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  void init(const boost::shared_ptr<StateMultibody> &state);
  boost::shared_ptr<ActuationModelAbstract> actuation_;
  boost::shared_ptr<ContactModelMultiple> contacts_;
  boost::shared_ptr<CostModelSum> costs_;
  boost::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar> &pinocchio_;

 public:
  class ResidualModelRnea : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;

    ResidualModelRnea(boost::shared_ptr<StateMultibody> state, const std::size_t nc, const std::size_t nu)
        : Base(state, state->get_nv(), state->get_nv() + nu + nc, true, true, true), nc_(nc), na_(nu) {}
    virtual ~ResidualModelRnea() {}

    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
                      const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      data->r = d->pinocchio->tau - d->actuation->tau;
    }

    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
                          const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      const std::size_t nv = state_->get_nv();
      data->Rx.leftCols(nv) = d->pinocchio->dtau_dq;
      data->Rx.rightCols(nv) = d->pinocchio->dtau_dv;
      data->Rx -= d->actuation->dtau_dx;
      data->Ru.leftCols(nv) = d->pinocchio->M;
      data->Ru.middleCols(nv, na_) = -d->actuation->dtau_du;
      data->Ru.rightCols(nc_) = -d->contact->Jc.transpose();
    }

    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data) {
      return boost::allocate_shared<typename Data::ResidualDataRnea>(
          Eigen::aligned_allocator<typename Data::ResidualDataRnea>(), this, data);
    }

    virtual void print(std::ostream &os) const {
      os << "ResidualModelRnea {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
         << "nc=" << nc_;
    }

   protected:
    using Base::nu_;
    using Base::state_;

   private:
    std::size_t nc_;  //!< Number of the contacts
    std::size_t na_;  //!< Number of actuated joints
  };

 public:
  class ResidualModelContact : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    ResidualModelContact(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const std::size_t nr,
                         const std::size_t nc, const std::size_t nu)
        : Base(state, nr, state->get_nv() + nu + nc, true, true, true), id_(id) {}
    virtual ~ResidualModelContact() {}

    void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
              const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      d->r = d->contact->a0;
    }

    void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
                  const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      d->Rx = d->contact->da0_dx;
      d->Ru.leftCols(state_->get_nv()) = d->contact->Jc;
    }

    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data) {
      return boost::allocate_shared<typename Data::ResidualDataContact>(
          Eigen::aligned_allocator<typename Data::ResidualDataContact>(), this, data, id_);
    }

   protected:
    using Base::nr_;
    using Base::nu_;
    using Base::state_;
    using Base::unone_;

   private:
    pinocchio::FrameIndex id_;  //!< Reference frame id
  };
};
template <typename _Scalar>
struct DifferentialActionDataContactInvDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactInvDynamicsTpl(Model<Scalar> *const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData(), model->get_contacts()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        constraints(model->get_constraints()->createData(&multibody)),
        contacts(model->get_contacts()->createData(&pinocchio)),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_Jstatic(model->get_state()->get_nv(), model->get_nu() + model->get_contacts()->get_nc_total()),
        tmp_Jcstatic(model->get_state()->get_nv(), model->get_contacts()->get_nc_total()) {
    costs->shareMemory(this);
    constraints->shareMemory(this);

    tmp_xstatic.setZero();
    tmp_Jstatic.setZero();
    tmp_Jcstatic.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyInContactTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  boost::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts;
  VectorXs tmp_xstatic;
  MatrixXs tmp_Jstatic;
  MatrixXs tmp_Jcstatic;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;

  struct ResidualDataRnea : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef DataCollectorActMultibodyInContactTpl<Scalar> DataCollectorActMultibodyInContact;

    template <template <typename Scalar> class Model>
    ResidualDataRnea(Model<Scalar> *const model, DataCollectorAbstract *const data) : Base(model, data) {
      // Check that proper shared data has been passed
      DataCollectorActMultibodyInContact *d = dynamic_cast<DataCollectorActMultibodyInContact *>(shared);
      if (d == NULL) {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorActMultibodyInContact");
      }
      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation.get();
      contact = d->contacts.get();
    }

    pinocchio::DataTpl<Scalar> *pinocchio;        //!< Pinocchio data
    ActuationDataAbstractTpl<Scalar> *actuation;  //!< Actuation data
    ContactDataMultipleTpl<Scalar> *contact;      //!< Contact data
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

  struct ResidualDataContact : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef DataCollectorMultibodyInContactTpl<Scalar> DataCollectorMultibodyInContact;
    typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;

    template <template <typename Scalar> class Model>
    ResidualDataContact(Model<Scalar> *const model, DataCollectorAbstract *const data, const std::size_t id)
        : Base(model, data) {
      DataCollectorMultibodyInContact *d = dynamic_cast<DataCollectorMultibodyInContact *>(shared);
      if (d == NULL) {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibodyInContact");
      }
      typename ContactModelMultiple::ContactDataContainer::iterator it, end;
      for (it = d->contacts->contacts.begin(), end = d->contacts->contacts.end(); it != end; ++it) {
        if (id == it->second->frame) {  // TODO(cmastalli): use model->get_id() and avoid to pass id in constructor
          contact = it->second.get();
          break;
        }
      }
    }

    ForceDataAbstractTpl<Scalar> *contact;  //!< Contact force data
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };
};
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/contact-invdyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
