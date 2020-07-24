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
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Define a center of pressure cost function
 *
 * It builds a cost function that bounds the center of pressure (cop) for one contact surface to
 * lie inside a certain geometric area defined around the reference contact frame. The cost residual
 * vector is described as \mathbf{r} = \mathbf{A} \cdot \mathbf{f} \geq \mathbf{0}, where \mathbf{A}=
 * \begin{bmatrix} 0 & 0 & X/2 & 0 & -1 & 0 \\ 0 & 0 & X/2 & 0 & 1 & 0 \\ 0 & 0 & Y/2 & 1 & 0 & 0 \\
 * 0 & 0 & Y/2 & -1 & 0 & 0 \end{bmatrix} is the inequality matrix and \mathbf{f} is the reference spatial
 * contact force in the frame coordinate. The constraints for the cop to lie inside the convex hull of
 * the foot, see eq.(18-19) of https://hal.archives-ouvertes.fr/hal-02108449/document can be written as:
 * \begin{align}\begin{split}\tau^x &\leq Yf^z \\-\tau^x &\leq Yf^z \\\tau^y &\leq Yf^z \\-\tau^y &\leq Yf^z
 * \end{split}\end{align}$`
 * The cost is computed, from the residual vector \mathbf{r}, through an user defined activation model.
 * Additionally, the contact frame id, the desired support region for the cop and the inequality matrix
 * are handled within FrameCoPSupportTpl. The force vector \mathbf{f} and its derivatives are computed by
 * DifferentialActionModelContactFwdDynamicsTpl. These values are stored in a shared data (i.e.
 * DataCollectorContactTpl). Note that this cost function cannot be used with other action models.
 *
 * \sa DifferentialActionModelContactFwdDynamicsTpl, DataCollectorContactTpl, ActivationModelAbstractTpl
 */
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
  typedef ActivationModelQuadraticBarrierTpl<Scalar> ActivationModelQuadraticBarrier;
  typedef ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;

  /**
   * @brief Initialize the cop cost model
   *
   * @param[in] state        Multibody state
   * @param[in] activation   Activation model
   * @param[in] cop_support  ID of contact frame and support region of the cop
   * @param[in] nu           Dimension of control vector
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const FrameCoPSupport& cop_support, const std::size_t& nu);

  /**
   * @brief Initialize the cop cost model
   *
   * For this case the default nu is equal to `state->get_nv()`.
   *
   * @param[in] state        Multibody state
   * @param[in] activation   Activation model
   * @param[in] cop_support  ID of contact frame and support region of the cop
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const FrameCoPSupport& cop_support);

  /**
   * @brief Initialize the cop cost model
   *
   * For this case the default activation model is quadratic barrier, i.e.
   * `ActivationModelQuadraticBarrierTpl(ActivationBounds(0, inf))`.
   *
   * @param[in] state        Multibody state
   * @param[in] cop_support  ID of contact frame and support region of the cop
   * @param[in] nu           Dimension of control vector
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const FrameCoPSupport& cop_support,
                                 const std::size_t& nu);

  /**
   * @brief Initialize the cop cost model
   *
   * For this case the default activation model is quadratic barrier, i.e.
   * `ActivationModelQuadraticBarrierTpl(ActivationBounds(0, inf))` and is equal to `state->get_nv().
   *
   * @param[in] state        Multibody state
   * @param[in] cop_support  ID of contact frame and support region of the cop
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const FrameCoPSupport& cop_support);
  virtual ~CostModelContactCoPPositionTpl();

  /**
   * @brief Compute the cop cost
   *
   * The cop residual is computed based on the A matrix, the force vector is computed by
   * DifferentialActionModelContactFwdDynamicsTpl and the results are stored in DataCollectorContactTpl.
   *
   * @param[in] data  cop data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the cop cost
   *
   * The cop derivatives are based on the force derivatives computed by
   * DifferentialActionModelContactFwdDynamicsTpl and stored in DataCollectorContactTpl.
   *
   * @param[in] data  cop data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the cop cost data
   *
   * Each cost model has its own data that needs to be allocated.
   * This function returns the allocated data for a predefined cost.
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Return the cop
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the cop
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameCoPSupport cop_support_;  //!< frame name of the contact foot and support region of the cop
};

template <typename _Scalar>
struct CostDataContactCoPPositionTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;

  template <template <typename Scalar> class Model>
  CostDataContactCoPPositionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.setZero();

    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Get the active 6d contact (avoids data casting at runtime)
    FrameCoPSupport cop_support = model->template get_reference<FrameCoPSupport>();
    std::string frame_name = model->get_state()->get_pinocchio()->frames[cop_support.get_frame()].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == cop_support.get_frame()) {
        ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          throw_pretty("Domain error: a 6d contact model is required in " + frame_name +
                       "in order to compute the CoP");
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
  boost::shared_ptr<ContactDataAbstractTpl<Scalar> > contact;  //!< contact force
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
