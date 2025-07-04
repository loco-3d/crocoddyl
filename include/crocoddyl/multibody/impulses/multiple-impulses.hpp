///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
#define CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ImpulseItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ImpulseModelAbstractTpl<Scalar> ImpulseModelAbstract;

  ImpulseItemTpl() {}
  ImpulseItemTpl(const std::string& name,
                 std::shared_ptr<ImpulseModelAbstract> impulse,
                 const bool active = true)
      : name(name), impulse(impulse), active(active) {}

  template <typename NewScalar>
  ImpulseItemTpl<NewScalar> cast() const {
    typedef ImpulseItemTpl<NewScalar> ReturnType;
    ReturnType ret(name, impulse->template cast<NewScalar>(), active);
    return ret;
  }

  /**
   * @brief Print information on the impulse item
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const ImpulseItemTpl<Scalar>& model) {
    os << "{" << *model.impulse << "}";
    return os;
  }

  std::string name;
  std::shared_ptr<ImpulseModelAbstract> impulse;
  bool active;
};

/**
 * @brief Define a stack of impulse models
 *
 * The impulse models can be defined with active and inactive status. The idea
 * behind this design choice is to be able to create a mechanism that allocates
 * the entire data needed for the computations. Then, there are designed
 * routines that update the only active impulse.
 */
template <typename _Scalar>
class ImpulseModelMultipleTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef ImpulseDataMultipleTpl<Scalar> ImpulseDataMultiple;
  typedef ImpulseModelAbstractTpl<Scalar> ImpulseModelAbstract;

  typedef ImpulseItemTpl<Scalar> ImpulseItem;

  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, std::shared_ptr<ImpulseItem> >
      ImpulseModelContainer;
  typedef std::map<std::string, std::shared_ptr<ImpulseDataAbstract> >
      ImpulseDataContainer;
  typedef typename pinocchio::container::aligned_vector<
      pinocchio::ForceTpl<Scalar> >::iterator ForceIterator;

  /**
   * @brief Initialize the multi-impulse model
   *
   * @param[in] state  Multibody state
   */
  explicit ImpulseModelMultipleTpl(std::shared_ptr<StateMultibody> state);
  ~ImpulseModelMultipleTpl();

  /**
   * @brief Add impulse item
   *
   * Note that the memory is allocated for inactive impulses as well.
   *
   * @param[in] name     Impulse name
   * @param[in] impulse  Impulse model
   * @param[in] active   Impulse status (active by default)
   */
  void addImpulse(const std::string& name,
                  std::shared_ptr<ImpulseModelAbstract> impulse,
                  const bool active = true);

  /**
   * @brief Remove impulse item
   *
   * @param[in] name  Impulse name
   */
  void removeImpulse(const std::string& name);

  /**
   * @brief Change the impulse status
   *
   * @param[in] name     Impulse name
   * @param[in] active   Impulse status (True for active)
   */
  void changeImpulseStatus(const std::string& name, const bool active);

  /**
   * @brief Compute the total impulse Jacobian and impulse velocity
   *
   * @param[in] data  Multi-impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<ImpulseDataMultiple>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the impulse holonomic constraint
   *
   * @param[in] data  Multi-impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const std::shared_ptr<ImpulseDataMultiple>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Update the system velocity after impulse
   *
   * @param[in] data   Multi-impulse data
   * @param[in] vnext  System velocity after impulse
   * \f$\mathbf{v}'\in\mathbb{R}^{nv}\f$
   */
  void updateVelocity(const std::shared_ptr<ImpulseDataMultiple>& data,
                      const VectorXs& vnext) const;

  /**
   * @brief Update the spatial impulse defined in frame coordinate
   *
   * @param[in] data     Multi-impulse data
   * @param[in] impulse  Spatial impulse defined in frame coordinate
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c\in\mathbb{R}^{nc}\f$
   */
  void updateForce(const std::shared_ptr<ImpulseDataMultiple>& data,
                   const VectorXs& impulse);

  /**
   * @brief Update the Jacobian of the system velocity after impulse
   *
   * @param[in] data       Multi-impulse data
   * @param[in] dvnext_dx  Jacobian of the system velocity after impact in
   * generalized coordinates
   * \f$\frac{\partial\dot{\mathbf{v}'}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times{ndx}}\f$
   */
  void updateVelocityDiff(const std::shared_ptr<ImpulseDataMultiple>& data,
                          const MatrixXs& dvnext_dx) const;

  /**
   * @brief Update the Jacobian of the spatial impulse defined in frame
   * coordinate
   *
   * @param[in] data    Multi-contact data
   * @param[in] df_dx   Jacobian of the spatial impulse defined in frame
   * coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\Lambda}}_c}{\partial\mathbf{x}}\in\mathbb{R}^{nc\times{ndx}}\f$
   */
  void updateForceDiff(const std::shared_ptr<ImpulseDataMultiple>& data,
                       const MatrixXs& df_dx) const;

  /**
   * @brief Update the RNEA derivatives dtau_dq by adding the skew term
   * (necessary for impulses expressed in LOCAL_WORLD_ALIGNED / WORLD)
   * @brief as explained in this document :
   * https://www.overleaf.com/read/tzvrrxxtntwk
   *
   * @param[in] data       Multi-contact data
   * @param[in] pinocchio  Pinocchio data
   */
  void updateRneaDiff(const std::shared_ptr<ImpulseDataMultiple>& data,
                      pinocchio::DataTpl<Scalar>& pinocchio) const;

  /**
   * @brief Create the multi-impulse data
   *
   * @param[in] data  Pinocchio data
   * @return the multi-impulse data.
   */
  std::shared_ptr<ImpulseDataMultiple> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Cast the multi-impulse model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ImpulseModelMultipleTpl<NewScalar> A multi-impulse model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ImpulseModelMultipleTpl<NewScalar> cast() const;

  /**
   * @brief Return the multibody state
   */
  const std::shared_ptr<StateMultibody>& get_state() const;

  /**
   * @brief Return the impulse models
   */
  const ImpulseModelContainer& get_impulses() const;

  /**
   * @brief Return the dimension of active impulses
   */
  std::size_t get_nc() const;

  /**
   * @brief Return the dimension of all impulses
   */
  std::size_t get_nc_total() const;

  /**
   * @brief Return the names of the set of active impulses
   */
  const std::set<std::string>& get_active_set() const;

  /**
   * @brief Return the names of the set of inactive impulses
   */
  const std::set<std::string>& get_inactive_set() const;

  /**
   * @brief Return the status of a given impulse name
   */
  bool getImpulseStatus(const std::string& name) const;

  /**
   * @brief Print information on the impulse models
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ImpulseModelMultipleTpl<Scalar>& model);

 private:
  std::shared_ptr<StateMultibody> state_;
  ImpulseModelContainer impulses_;
  std::size_t nc_;
  std::size_t nc_total_;
  std::set<std::string> active_set_;
  std::set<std::string> inactive_set_;
};

/**
 * @brief Define the multi-impulse data
 *
 * \sa ImpulseModelMultipleTpl
 */
template <typename _Scalar>
struct ImpulseDataMultipleTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef ImpulseItemTpl<Scalar> ImpulseItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialized a multi-impulse data
   *
   * @param[in] model  Multi-impulse model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  ImpulseDataMultipleTpl(Model<Scalar>* const model,
                         pinocchio::DataTpl<Scalar>* const data)
      : Jc(model->get_nc_total(), model->get_state()->get_nv()),
        dv0_dq(model->get_nc_total(), model->get_state()->get_nv()),
        vnext(model->get_state()->get_nv()),
        dvnext_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints,
             pinocchio::ForceTpl<Scalar>::Zero()) {
    Jc.setZero();
    dv0_dq.setZero();
    vnext.setZero();
    dvnext_dx.setZero();
    for (typename ImpulseModelMultiple::ImpulseModelContainer::const_iterator
             it = model->get_impulses().begin();
         it != model->get_impulses().end(); ++it) {
      const std::shared_ptr<ImpulseItem>& item = it->second;
      impulses.insert(
          std::make_pair(item->name, item->impulse->createData(data)));
    }
  }

  MatrixXs Jc;  //!< Contact Jacobian in frame coordinate
                //!< \f$\mathbf{J}_c\in\mathbb{R}^{ni_{total}\times{nv}}\f$
                //!< (memory defined for active and inactive impulses)
  MatrixXs
      dv0_dq;  //!< Jacobian of the desired spatial contact acceleration in
               //!< frame coordinate
               //!< \f$\frac{\partial\underline{\mathbf{v}}_0}{\partial\mathbf{q}}\in\mathbb{R}^{ni_{total}\times{nv}}\f$
               //!< (memory defined for active and inactive impulse)
  VectorXs vnext;  //!< Constrained system velocity after impact in generalized
                   //!< coordinates \f$\dot{\mathbf{v}'}\in\mathbb{R}^{nv}\f$
  MatrixXs
      dvnext_dx;  //!< Jacobian of the system velocity after impact in
                  //!< generalized coordinates
                  //!< \f$\frac{\partial\dot{\mathbf{v}'}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
                  //!< ndx}\f$
  typename ImpulseModelMultiple::ImpulseDataContainer
      impulses;  //!< Stack of impulse data
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
      fext;  //!< External spatial forces in body coordinates
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulses/multiple-impulses.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ImpulseItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ImpulseModelMultipleTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ImpulseDataMultipleTpl)

#endif  // CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
