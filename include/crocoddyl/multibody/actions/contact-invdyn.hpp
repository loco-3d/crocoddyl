///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_

#include <stdexcept>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelContactInvDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactInvDynamicsTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
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

  /**
   * @brief Print relevant information of the contact forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

 protected:
  using Base::ng_;     //!< Number of inequality constraints
  using Base::nh_;     //!< Number of equality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
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
        : Base(state, state->get_nv(), state->get_nv() + nu, true, true, true),
          nc_(nc),
          _nv_a(nu),
          _nv_f(state->get_nv() - nu) {}
    virtual ~ResidualModelRnea() {}

    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
                      const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
        const std::size_t nv = state_->get_nv();
        const std::size_t nu = _nv_a;
        const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = u.segment(nv, nv+nu);
        d->r.head(_nv_f) = d->pinocchio.tau.head(_nv_f)
        data->r.tail(nu) = d->pinocchio->tau.tail(nu) - tau;
    }

    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &,
                          const Eigen::Ref<const VectorXs> &) {
      const boost::shared_ptr<typename Data::ResidualDataRnea> &d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      const std::size_t nv = state_->get_nv();
      const std::size_t nu = _nv_a;
      data->Rx.leftCols(nv) = d->pinocchio->dtau_dq;
      data->Rx.rightCols(nv) = d->pinocchio->dtau_dv;
      data->Rx -= d->actuation->dtau_dx;
      data->Ru.leftCols(nv) = d->pinocchio->M;
      data->Ru.rightCols(nv + na_) = -d->contacts->Jc.transpose();
    }

    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data) {
      return boost::allocate_shared<typename Data::ResidualDataRnea>(
          Eigen::aligned_allocator<typename Data::ResidualDataRnea>(), this, data);
    }

    /**
     * @brief Print relevant information of the RNEA residual model
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const {
      os << "ResidualModelRnea {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
         << ", na=" << na_ << "}";
    }

   protected:
    std::size_t nc_;
    std::size_t _nv_f;
    std::size_t _nv_a;
    using Base::nu_;
    using Base::state_;
  };

  template <typename _Scalar>
  class ResidualModelContactTpl : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataContactForceTpl<Scalar> Data;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef pinocchio::ForceTpl<Scalar> Force;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the contact acceleration residual model
     *
     * @param[in] state  Multibody state
     * @param[in] id     Reference frame id
     * @param[in] nr     Dimenstion of residual vector
     * @param[in] nc     Dimension of the contact force (nc <= 6)
     * @param[in] nu     Dimension of control vector
     */
    ResidualModelContactTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                            const std::size_t nr, const std::size_t nc, const std::size_t nu);

    /**
     * @brief Initialize the contact acceleration residual model
     *
     * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state  Multibody state
     * @param[in] id     Reference frame id
     * @param[in] nr     Dimenstion of residual vector
     * @param[in] nc     Dimension of the contact force (nc <= 6)
     */
    ResidualModelContactTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                            const std::size_t nr, const std::size_t nc);
    virtual ~ResidualModelContactTpl();

    /**
     * @brief Compute the contact force residual
     *
     * The residual is computed based on  the acceleration vector is computed by
     * `DifferentialActionModelContactInvDynamicsTpl` which is stored in
     * `DataCollectorContactTpl` .
     *
     * @param[in] data  Contact force data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the contact force residual
     *
     * The residual is computed based on  the acceleration vector is computed by
     * `DifferentialActionModelContactInvDynamicsTpl` which is stored in
     * `DataCollectorContactTpl` .
     *
     * @param[in] data  Contact force data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Create the contact acceleration residual data
     *
     * @param[in] data  shared data (it should be of type
     * DataCollectorContactTpl)
     * @return the residual data.
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

    //@cmastalli Do you think it is required? 
    /**
     * @brief Return the reference frame id
     */
    pinocchio::FrameIndex get_id() const;

    /**
     * @brief Print relevant information of the contact-force residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

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
        Kinv(model->get_state()->get_nv() + model->get_contacts()->get_nc_total(),
             model->get_state()->get_nv() + model->get_contacts()->get_nc_total()),
        df_dx(model->get_contacts()->get_nc_total(), model->get_state()->get_ndx()),
        df_du(model->get_contacts()->get_nc_total(), model->get_nu()),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_Jstatic(model->get_state()->get_nv(), model->get_nu() + model->get_contacts()->get_nc_total()) {
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Kinv.setZero();
    df_dx.setZero();
    df_du.setZero();
    tmp_xstatic.setZero();
    tmp_Jstatic.setZero();
    pinocchio.lambda_c.resize(model->get_contacts()->get_nc_total());
    pinocchio.lambda_c.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyInContactTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar>> costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar>> constraints;
  MatrixXs df_dx;
  MatrixXs df_du;
  VectorXs tmp_xstatic;
  MatrixXs tmp_Jstatic;

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

    template <template <typename Scalar> class Model>
    ResidualDataRnea(Model<Scalar> *const model, DataCollectorAbstract *const data) : Base(model, data) {
      // Check that proper shared data has been passed
      DataCollectorActMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorActMultibodyTpl<Scalar> *>(shared);
      if (d == NULL) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorActMultibody");
      }

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation.get();
    }

    pinocchio::DataTpl<Scalar> *pinocchio;        //!< Pinocchio data
    ActuationDataAbstractTpl<Scalar> *actuation;  //!< Actuation data
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

  template <typename _Scalar>
  struct ResidualDataContactTpl : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
    typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
    typedef pinocchio::ForceTpl<Scalar> Force;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar> class Model>
    ResidualDataContactTpl(Model<Scalar> *const model, DataCollectorAbstract *const data) : Base(model, data) {
      contact_type = ContactUndefined;

      // Check that proper shared data has been passed
      bool is_contact = true;
      DataCollectorContactTpl<Scalar> *d1 = dynamic_cast<DataCollectorContactTpl<Scalar> *>(shared);
      DataCollectorImpulseTpl<Scalar> *d2 = dynamic_cast<DataCollectorImpulseTpl<Scalar> *>(shared);
      if (d1 == NULL && d2 == NULL) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorContact or DataCollectorImpulse");
      }
      if (d2 != NULL) {
        is_contact = false;
      }

      const pinocchio::FrameIndex id = model->get_id();
      const boost::shared_ptr<StateMultibody> &state = boost::static_pointer_cast<StateMultibody>(model->get_state());
      std::string frame_name = state->get_pinocchio()->frames[id].name;
      bool found_contact = false;
      if (is_contact) {
        for (typename ContactModelMultiple::ContactDataContainer::iterator it = d1->contacts->contacts.begin();
             it != d1->contacts->contacts.end(); ++it) {
          if (it->second->frame == id) {
            ContactData3DTpl<Scalar> *d3d = dynamic_cast<ContactData3DTpl<Scalar> *>(it->second.get());
            if (d3d != NULL) {
              contact_type = Contact3D;
              found_contact = true;
              contact = it->second;
              break;
            }
            ContactData6DTpl<Scalar> *d6d = dynamic_cast<ContactData6DTpl<Scalar> *>(it->second.get());
            if (d6d != NULL) {
              contact_type = Contact6D;
              found_contact = true;
              contact = it->second;
              break;
            }
            throw_pretty("Domain error: there isn't defined at least a 3d contact for " + frame_name);
            break;
          }
        }
      } else {
        for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d2->impulses->impulses.begin();
             it != d2->impulses->impulses.end(); ++it) {
          if (it->second->frame == id) {
            ImpulseData3DTpl<Scalar> *d3d = dynamic_cast<ImpulseData3DTpl<Scalar> *>(it->second.get());
            if (d3d != NULL) {
              contact_type = Contact3D;
              found_contact = true;
              contact = it->second;
              break;
            }
            ImpulseData6DTpl<Scalar> *d6d = dynamic_cast<ImpulseData6DTpl<Scalar> *>(it->second.get());
            if (d6d != NULL) {
              contact_type = Contact6D;
              found_contact = true;
              contact = it->second;
              break;
            }
            throw_pretty("Domain error: there isn't defined at least a 3d impulse for " + frame_name);
            break;
          }
        }
      }
      if (!found_contact) {
        throw_pretty("Domain error: there isn't defined contact/impulse data for " + frame_name);
      }
    }

    boost::shared_ptr<ForceDataAbstractTpl<Scalar>> contact;  //!< Contact force data
    ContactType contact_type;                                 //!< Type of contact (3D / 6D)
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/contact-invdyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
