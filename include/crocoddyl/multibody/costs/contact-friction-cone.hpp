///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelContactFrictionConeTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef FrictionConeTpl<Scalar> FrictionCone;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;

  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation, const FrictionCone& cone,
                                  const FrameIndex& frame, const std::size_t& nu);
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation, const FrictionCone& cone,
                                  const FrameIndex& frame);
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state, const FrictionCone& cone,
                                  const FrameIndex& frame, const std::size_t& nu);
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state, const FrictionCone& cone,
                                  const FrameIndex& frame);
  ~CostModelContactFrictionConeTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrictionCone& get_friction_cone() const;
  const FrameIndex& get_frame() const;
  void set_friction_cone(const FrictionCone& cone);
  void set_frame(const FrameIndex& frame);

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;
  using Base::with_residuals_;

 protected:
  FrictionCone friction_cone_;
  FrameIndex frame_;
};

template <typename _Scalar>
struct CostDataContactFrictionConeTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataContactFrictionConeTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()),
        more_than_3_constraints(false) {
    Arr_Ru.setZero();

    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Avoids data casting at runtime
    std::string frame_name = model->get_state()->get_pinocchio().frames[model->get_frame()].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == model->get_frame()) {
        ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          found_contact = true;
          contact = it->second;
          break;
        }
        ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          more_than_3_constraints = true;
          found_contact = true;
          contact = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 3d contact for " + frame_name);
        break;
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  boost::shared_ptr<ContactDataAbstractTpl<Scalar> > contact;
  MatrixXs Arr_Ru;
  bool more_than_3_constraints;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-friction-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
