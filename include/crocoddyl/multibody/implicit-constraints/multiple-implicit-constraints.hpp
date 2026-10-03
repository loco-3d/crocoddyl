///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINTS_MULTIPLE_IMPLICIT_CONSTRAINTS_HPP_
#define CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINTS_MULTIPLE_IMPLICIT_CONSTRAINTS_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraint-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ImplicitConstraintItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ImplicitConstraintModelAbstractTpl<Scalar>
      ImplicitConstraintModelAbstract;

  /**
   * @brief Initialize the implicit-constraint item
   *
   * @param[in] name        Implicit-constraint name
   * @param[in] constraint  Implicit-constraint model
   * @param[in] active      Implicit-constraint status (active by default)
   */
  ImplicitConstraintItemTpl(
      const std::string& name,
      std::shared_ptr<ImplicitConstraintModelAbstract> constraint,
      const bool active = true)
      : name(name), constraint(constraint), active(active) {
    if (constraint == nullptr) {
      throw_pretty("Invalid argument: implicit-constraint model is null");
    }
  }
  ImplicitConstraintItemTpl(const ImplicitConstraintItemTpl&) = default;
  ImplicitConstraintItemTpl& operator=(const ImplicitConstraintItemTpl&) =
      delete;

  const std::string& get_name() const { return name; }

  const std::shared_ptr<ImplicitConstraintModelAbstract>& get_constraint()
      const {
    return constraint;
  }

  bool get_active() const { return active; }

  template <typename NewScalar>
  ImplicitConstraintItemTpl<NewScalar> cast() const {
    typedef ImplicitConstraintItemTpl<NewScalar> ReturnType;
    ReturnType ret(name, constraint->template cast<NewScalar>(), active);
    return ret;
  }

  /**
   * @brief Print information on the implicit-constraint item
   */
  friend std::ostream& operator<<(
      std::ostream& os, const ImplicitConstraintItemTpl<Scalar>& model) {
    os << "{" << *model.constraint << "}";
    return os;
  }

 private:
  std::string name;
  std::shared_ptr<ImplicitConstraintModelAbstract> constraint;
  bool active;

  friend class ImplicitConstraintModelMultipleTpl<Scalar>;
  friend struct ImplicitConstraintDataMultipleTpl<Scalar>;
};

/**
 * @brief Define a stack of implicit-constraint models
 *
 * The implicit-constraint models can be defined with active and inactive
 * status. The idea behind this design choice is to be able to create a
 * mechanism that allocates the entire data needed for the computations. Then,
 * there are designed routines that update the only active constraints.
 */
template <typename _Scalar>
class ImplicitConstraintModelMultipleTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImplicitConstraintDataAbstractTpl<Scalar>
      ImplicitConstraintDataAbstract;
  typedef ImplicitConstraintDataMultipleTpl<Scalar>
      ImplicitConstraintDataMultiple;
  typedef ImplicitConstraintModelAbstractTpl<Scalar>
      ImplicitConstraintModelAbstract;
  typedef ImplicitConstraintItemTpl<Scalar> ImplicitConstraintItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef std::map<std::string, std::shared_ptr<ImplicitConstraintItem> >
      ImplicitConstraintModelContainer;
  typedef std::map<std::string,
                   std::shared_ptr<ImplicitConstraintDataAbstract> >
      ImplicitConstraintDataContainer;
  /**
   * @brief Initialize the multiple implicit-constraint model
   *
   * @param[in] state  Multibody state
   * @param[in] nu     Dimension of control vector
   */
  ImplicitConstraintModelMultipleTpl(std::shared_ptr<StateMultibody> state,
                                     const std::size_t nu);

  /**
   * @brief Initialize the multiple implicit-constraint model
   *
   * @param[in] state  Multibody state
   */
  explicit ImplicitConstraintModelMultipleTpl(
      std::shared_ptr<StateMultibody> state);
  ImplicitConstraintModelMultipleTpl(
      const ImplicitConstraintModelMultipleTpl& other);
  ImplicitConstraintModelMultipleTpl& operator=(
      const ImplicitConstraintModelMultipleTpl&) = delete;
  ~ImplicitConstraintModelMultipleTpl();

  /**
   * @brief Add implicit-constraint item
   *
   * Note that the memory is allocated for inactive constraints as well.
   *
   * @param[in] name        Implicit-constraint name
   * @param[in] constraint  Implicit-constraint model
   * @param[in] active      Implicit-constraint status (active by default)
   */
  void addConstraint(
      const std::string& name,
      std::shared_ptr<ImplicitConstraintModelAbstract> constraint,
      const bool active = true);

  /**
   * @brief Remove implicit-constraint item
   *
   * @param[in] name  Implicit-constraint name
   */
  void removeConstraint(const std::string& name);

  /**
   * @brief Change the implicit-constraint status
   *
   * @param[in] name     Implicit-constraint name
   * @param[in] active   Implicit-constraint status (True for active)
   */
  void changeConstraintStatus(const std::string& name, const bool active);

  /**
   * @brief Compute the implicit-constraint Jacobian and implicit-constraint
   * acceleration
   *
   * @param[in] data  Multiple implicit-constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the implicit-constraint holonomic
   * constraint
   *
   * @param[in] data  Multiple implicit-constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Update the constrained system next velocity
   *
   * @param[in] data   Multiple implicit-constraint data
   * @param[in] vnext  Constrained system next velocity
   * \f$\mathbf{v}_{k+1}\in\mathbb{R}^{nv}\f$
   */
  void updateVelocity(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      const VectorXs& vnext) const;

  /**
   * @brief Update the constrained system acceleration
   *
   * @param[in] data  Multiple implicit-constraint data
   * @param[in] dv    Constrained system acceleration
   * \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
   */
  void updateAcceleration(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      const VectorXs& dv) const;

  /**
   * @brief Update the spatial force defined in frame coordinate
   *
   * @param[in] data   Multiple implicit-constraint data
   * @param[in] force  Spatial force defined in frame coordinate
   * \f${}^o\underline{\boldsymbol{\lambda}}_c\in\mathbb{R}^{nc}\f$
   */
  void updateForce(const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
                   const VectorXs& force);

  /**
   * @brief Update the Jacobian of the constrained system next velocity
   *
   * @param[in] data       Multiple implicit-constraint data
   * @param[in] dvnext_dx  Jacobian of the system next velocity in generalized
   * coordinates
   * \f$\frac{\partial\mathbf{v}_{k+1}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
   * ndx}\f$
   */
  void updateVelocityDiff(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      const MatrixXs& dvnext_dx) const;

  /**
   * @brief Update the Jacobian of the constrained system acceleration
   *
   * @param[in] data    Multiple implicit-constraint data
   * @param[in] ddv_dx  Jacobian of the system acceleration in generalized
   * coordinates
   * \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
   * ndx}\f$
   */
  void updateAccelerationDiff(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      const MatrixXs& ddv_dx) const;

  /**
   * @brief Update the Jacobian of the spatial force defined in frame
   * coordinate
   *
   * @param[in] data   Multiple implicit-constraint data
   * @param[in] df_dx  Jacobian of the spatial force defined in frame
   * coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{x}}\in\mathbb{R}^{nc\times{ndx}}\f$
   * @param[in] df_du  Jacobian of the spatial force defined in frame
   * coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{u}}\in\mathbb{R}^{nc\times{nu}}\f$
   */
  void updateForceDiff(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      const MatrixXs& df_dx, const MatrixXs& df_du) const;

  /**
   * @brief Update the RNEA derivatives dtau_dq by adding the skew term
   * (necessary for constraints expressed in LOCAL_WORLD_ALIGNED / WORLD)
   *
   * To learn more about the computation of the constraints derivatives in
   * different frames see https://hal.science/hal-03758989/document.
   *
   * @param[in] data       Multiple implicit-constraint data
   * @param[in] pinocchio  Pinocchio data
   */
  void updateRneaDiff(
      const std::shared_ptr<ImplicitConstraintDataMultiple>& data,
      pinocchio::DataTpl<Scalar>& pinocchio) const;

  /**
   * @brief Create the multiple implicit-constraint data
   *
   * @param[in] data  Pinocchio data
   * @return the multiple implicit-constraint data.
   */
  std::shared_ptr<ImplicitConstraintDataMultiple> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Cast the multiple implicit-constraint model to a different scalar
   * type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ImplicitConstraintModelMultipleTpl<NewScalar> A multiple
   * implicit-constraint model with the new scalar type.
   */
  template <typename NewScalar>
  ImplicitConstraintModelMultipleTpl<NewScalar> cast() const;

  /**
   * @brief Return the multibody state
   */
  const std::shared_ptr<StateMultibody>& get_state() const;

  /**
   * @brief Return the implicit-constraint models
   */
  const ImplicitConstraintModelContainer& get_constraints() const;

  /**
   * @brief Return the dimension of active implicit constraints
   */
  std::size_t get_nc() const;

  /**
   * @brief Return the dimension of all implicit constraints
   */
  std::size_t get_nc_total() const;

  /**
   * @brief Return the dimension of control vector
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the names of the set of active implicit constraints
   */
  const std::set<std::string>& get_active_set() const;

  /**
   * @brief Return the names of the set of inactive implicit constraints
   */
  const std::set<std::string>& get_inactive_set() const;

  /**
   * @brief Return the status of a given implicit-constraint name
   */
  bool getConstraintStatus(const std::string& name) const;

  /**
   * @brief Return the type of implicit-constraint computation
   *
   * True for all constraints, otherwise false for active constraints.
   */
  bool getComputeAllConstraints() const;

  /**
   * @brief Set the type of implicit-constraint computation: all or active
   * constraints
   *
   * @param status  Type of implicit-constraint computation (true for all
   * constraints and false for active constraints)
   */
  void setComputeAllConstraints(const bool status);

  /**
   * @brief Print information on the implicit-constraint models
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os,
      const ImplicitConstraintModelMultipleTpl<Scalar>& model);

 private:
  std::shared_ptr<StateMultibody> state_;
  ImplicitConstraintModelContainer constraints_;
  std::size_t nc_;
  std::size_t nc_total_;
  std::size_t nu_;
  std::set<std::string> active_set_;
  std::set<std::string> inactive_set_;
  bool compute_all_constraints_;
};

/**
 * @brief Define the multiple implicit-constraint data
 *
 * \sa ImplicitConstraintModelMultipleTpl
 */
template <typename _Scalar>
struct ImplicitConstraintDataMultipleTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef ImplicitConstraintItemTpl<Scalar> ImplicitConstraintItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty(
          "Invalid argument: multiple implicit-constraint model is null");
    }
    return model;
  }

 public:
  /**
   * @brief Initialized a multiple implicit-constraint data
   *
   * @param[in] model  Multiple implicit-constraint model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  ImplicitConstraintDataMultipleTpl(Model<Scalar>* const model,
                                    pinocchio::DataTpl<Scalar>* const data)
      : Jc(checkModel(model)->get_nc_total(),
           checkModel(model)->get_state()->get_nv()),
        a0(model->get_nc_total()),
        da0_dx(model->get_nc_total(), model->get_state()->get_ndx()),
        dv0_dq(model->get_nc_total(), model->get_state()->get_nv()),
        vnext(model->get_state()->get_nv()),
        dv(model->get_state()->get_nv()),
        dvnext_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        ddv_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints,
             pinocchio::ForceTpl<Scalar>::Zero()) {
    if (data == nullptr) {
      throw_pretty("Invalid argument: Pinocchio data is null");
    }
    Jc.setZero();
    a0.setZero();
    da0_dx.setZero();
    dv0_dq.setZero();
    vnext.setZero();
    dv.setZero();
    dvnext_dx.setZero();
    ddv_dx.setZero();
    for (typename ImplicitConstraintModelMultiple::
             ImplicitConstraintModelContainer::const_iterator it =
                 model->get_constraints().begin();
         it != model->get_constraints().end(); ++it) {
      const std::shared_ptr<ImplicitConstraintItem>& item = it->second;
      constraints.insert(
          std::make_pair(item->name, item->constraint->createData(data)));
    }
  }

  MatrixXs Jc;  //!< Implicit-constraint Jacobian in frame coordinate
                //!< \f$\mathbf{J}_c\in\mathbb{R}^{nc_{total}\times{nv}}\f$
                //!< (memory defined for active and inactive constraints)
  VectorXs a0;  //!< Desired spatial implicit-constraint acceleration in frame
                //!< coordinate
                //!< \f$\underline{\mathbf{a}}_0\in\mathbb{R}^{nc_{total}}\f$
                //!< (memory defined for active and inactive constraints)
  MatrixXs
      da0_dx;  //!< Jacobian of the desired spatial implicit-constraint
               //!< acceleration in frame coordinate
               //!< \f$\frac{\partial\underline{\mathbf{a}}_0}{\partial\mathbf{x}}\in\mathbb{R}^{nc_{total}\times{ndx}}\f$
               //!< (memory defined for active and inactive constraints)
  MatrixXs
      dv0_dq;  //!< Jacobian of the desired velocity-level drift in frame
               //!< coordinate
               //!< \f$\frac{\partial\underline{\mathbf{v}}_0}{\partial\mathbf{q}}\in\mathbb{R}^{nc_{total}\times{nv}}\f$
               //!< (memory defined for active and inactive constraints)
  VectorXs vnext;  //!< Constrained system next velocity in generalized
                   //!< coordinates
                   //!< \f$\mathbf{v}_{k+1}\in\mathbb{R}^{nv}\f$
  VectorXs dv;     //!< Constrained system acceleration in generalized
                   //!< coordinates
                   //!< \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
  MatrixXs
      dvnext_dx;  //!< Jacobian of the system next velocity in generalized
                  //!< coordinates
                  //!< \f$\frac{\partial\mathbf{v}_{k+1}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
                  //!< ndx}\f$
  MatrixXs
      ddv_dx;  //!< Jacobian of the system acceleration in generalized
               //!< coordinates
               //!< \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
               //!< ndx}\f$
  typename ImplicitConstraintModelMultiple::ImplicitConstraintDataContainer
      constraints;  //!< Stack of implicit-constraint data
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
      fext;  //!< External spatial forces in body coordinates
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ImplicitConstraintItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ImplicitConstraintModelMultipleTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ImplicitConstraintDataMultipleTpl)

#endif  // CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINTS_MULTIPLE_IMPLICIT_CONSTRAINTS_HPP_
