///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelContactCoPPositionTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataContactCoPPositionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::Matrix46s Matrix46s;

  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& frame_id, 
                          const Vector2s& cop_region, const Vector3s& normal, const std::size_t& nu);
  virtual ~CostModelContactCoPPositionTpl();

  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const std::size_t& get_frame_id() const;
  const Vector2s& get_cop_region() const;
  const Vector3s& get_normal() const;

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private: 
  const std::size_t frame_id_; //!< contact frame id
  const Vector2s cop_region_; //!< desired 2d area (length, width) of the CoP defined w.r.t. the contact frame
  const Vector3s normal_; //!< vector normal to the contact surface 
};

template <typename _Scalar>
struct CostDataContactCoPPositionTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::Matrix46s Matrix46s;

  template <template <typename Scalar> class Model>
  CostDataContactCoPPositionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.setZero();
        
    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    //Get model parameters
    frame_id = model->get_frame_id();
    cop_region = model->get_cop_region();
    normal = model->get_normal();

    //Compute the inequality matrix
    A << 0, 0, cop_region[0] / 2, 0, -1, 0,
        0, 0, cop_region[0] / 2, 0, 1, 0,
        0, 0, cop_region[1] / 2, 1, 0, 0,
        0, 0, cop_region[1] / 2, -1, 0, 0;

    // Get the active 6d contact (avoids data casting at runtime)
    std::string frame_name = model->get_state()->get_pinocchio()->frames[frame_id].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == frame_id) {
        ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          throw_pretty("Domain error: a 6d contact model is required in " +
                         frame_name + "in order to compute the CoP");
          break;
        }
        ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          found_contact = true;
          contact = it->second;
          break;
        }
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  MatrixXs Arr_Ru;
  boost::shared_ptr<ContactDataAbstractTpl<Scalar> > contact; //!< contact force
  pinocchio::ForceTpl<Scalar> f; //!< transformed contact force
  std::size_t frame_id; //!< contact frame id
  Vector2s cop_region; //!< desired 2d area (length, width) of the CoP defined w.r.t. the contact frame
  Vector3s normal; //!< vector normal to the contact surface 
  Matrix46s A; //!< inequality matrix constraining the CoP position
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
#include "crocoddyl/multibody/costs/contact-cop-position.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_
