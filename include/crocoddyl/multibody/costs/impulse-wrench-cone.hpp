///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_WRENCH_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelImpulseWrenchConeTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataImpulseWrenchConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef WrenchConeTpl<Scalar> WrenchCone;
  typedef FrameWrenchConeTpl<Scalar> FrameWrenchCone;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::MatrixX6s MatrixX6s;

  CostModelImpulseWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation,
                                  const FrameWrenchCone& fref);
  CostModelImpulseWrenchConeTpl(boost::shared_ptr<StateMultibody> state, const FrameWrenchCone& fref);
  virtual ~CostModelImpulseWrenchConeTpl();

  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::state_;
  //   using Base::unone_;

 private:
  FrameWrenchCone fref_;
};

template <typename _Scalar>
struct CostDataImpulseWrenchConeTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef FrameWrenchConeTpl<Scalar> FrameWrenchCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataImpulseWrenchConeTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    }

    // Avoids data casting at runtime
    FrameWrenchCone fref = model->template get_reference<FrameWrenchCone>();
    std::string frame_name = model->get_state()->get_pinocchio()->frames[fref.id].name;
    bool found_impulse = false;
    for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d->impulses->impulses.begin();
         it != d->impulses->impulses.end(); ++it) {
      if (it->second->frame == fref.id) {
        ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          found_impulse = true;
          impulse = it->second;
          throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
        }
        ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          found_impulse = true;
          impulse = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 6d impulse for " + frame_name);
        break;
      }
    }
    if (!found_impulse) {
      throw_pretty("Domain error: there isn't defined impulse data for " + frame_name);
    }
  }

  boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > impulse;

  using Base::activation;
  using Base::cost;
  using Base::Lx;
  using Base::Lxx;
  using Base::r;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_WRENCH_CONE_HPP_
