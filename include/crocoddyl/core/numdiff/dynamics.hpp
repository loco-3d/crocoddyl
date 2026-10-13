///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_DYNAMICS_HPP_
#define CROCODDYL_CORE_NUMDIFF_DYNAMICS_HPP_

#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/numdiff/restoration.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of a dynamics model.
 *
 * It computes \f$F_x,F_u,F_p\f$, dissipative-power derivatives and every
 * constraint Jacobian by forward finite differences. An optional
 * ParameterManager defines the active D075 parameter layout; update_p() sets
 * the nominal vector before evaluation. Native running and terminal overloads
 * are preserved, and terminal evaluation differentiates state blocks only.
 *
 * \sa `DynamicsModelAbstractTpl()`, `calcDiff_xu()`, `calcDiff_p()`
 */
template <typename _Scalar>
class DynamicsModelNumDiffTpl : public DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DynamicsModelBase, DynamicsModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsModelAbstractTpl<Scalar> Base;
  typedef DynamicsDataNumDiffTpl<Scalar> Data;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;

  /** @brief Construct from a dynamics model without an external manager. */
  explicit DynamicsModelNumDiffTpl(std::shared_ptr<Base> model);
  /** @brief Construct from a dynamics model and active parameter manager. */
  DynamicsModelNumDiffTpl(std::shared_ptr<Base> model,
                          std::shared_ptr<ParameterManager> params);
  virtual ~DynamicsModelNumDiffTpl() = default;

  /** @copydoc Base::calc() */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /** @copydoc Base::calc(const std::shared_ptr<DynamicsDataAbstract>&, const
   * Eigen::Ref<const VectorXs>&) */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /** @copydoc Base::calcDiff_xu() */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Numerically differentiate terminal state-dependent blocks. */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x) override;

  /** @copydoc Base::calcDiff_p() */
  virtual void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u) override;

  /** @copydoc Base::createData() */
  using Base::createData;
  virtual std::shared_ptr<DynamicsDataAbstract> createData() override;
  /** @brief Create data sharing an existing parameter-manager data object. */
  virtual std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /** @brief Set the active parameter manager on every internal data object. */
  void set_params(const std::shared_ptr<DynamicsDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update the nominal active parameter vector. */
  virtual void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /** @brief Cast the model and its parameter manager to another scalar. */
  template <typename NewScalar>
  DynamicsModelNumDiffTpl<NewScalar> cast() const;

  /** @brief Return the dynamics model being differentiated. */
  const std::shared_ptr<Base>& get_model() const;

  /** @brief Return the optional active parameter manager. */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /** @brief Return the finite-difference disturbance. */
  const Scalar get_disturbance() const;

  /** @brief Set the nonnegative finite-difference disturbance. */
  void set_disturbance(const Scalar disturbance);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::ng_;
  using Base::nh_;
  using Base::nu_;
  using Base::state_;

 private:
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& x);

  std::shared_ptr<Base> model_;
  std::shared_ptr<ParameterManager> params_;
  Scalar e_jac_;
};

/**
 * @brief Data and preallocated scratch for DynamicsModelNumDiffTpl.
 *
 * The object owns independent wrapped data for the nominal point and every
 * state, control and parameter perturbation. params_data is shared ownership
 * of the parameter payload supplied at construction.
 */
template <typename _Scalar>
struct DynamicsDataNumDiffTpl : public DynamicsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsDataAbstractTpl<Scalar> Base;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;

  /** @brief Allocate all wrapped data and same-size numerical scratch. */
  template <template <typename Scalar> class Model>
  explicit DynamicsDataNumDiffTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(internal::checkNumDiffModel(model)),
        params_data(params_data),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        up(model->get_model()->get_nu()),
        dp(model->get_np()),
        p(model->get_np()),
        pp(model->get_np()),
        xp(model->get_model()->get_state()->get_nx()),
        dvdot(model->get_model()->get_state()->get_ndx()) {
    dx.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    p.setZero();
    pp.setZero();
    xp.setZero();
    dvdot.setZero();
    if (params_data != nullptr &&
        (params_data->parameter_data != params_data.get() ||
         params_data->params == nullptr ||
         params_data->params->np != model->get_np())) {
      throw_pretty(
          "Invalid argument: parameter data has an incompatible "
          "dimension");
    }

    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    const std::size_t np = model->get_np();
    const auto create_wrapped_data = [&]() {
      return params_data != nullptr
                 ? model->get_model()->createData(params_data)
                 : model->get_model()->createData();
    };
    data_0 = create_wrapped_data();
    data_x.reserve(ndx);
    data_u.reserve(nu);
    data_p.reserve(np);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(create_wrapped_data());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(create_wrapped_data());
    }
    for (std::size_t i = 0; i < np; ++i) {
      data_p.push_back(create_wrapped_data());
    }
  }

  template <class Model>
  void resize(Model* const model) {
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nq = model->get_state()->get_nq();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const bool is_discrete =
        (model->get_dyn_type() == DynamicsType::DiscreteTime);
    const std::size_t nvdot = is_discrete ? nq + nv : nv;
    const std::size_t ndvdot = is_discrete ? ndx : nv;

    this->vdot.resize(nvdot);
    this->Fx.resize(ndvdot, ndx);
    this->Fu.resize(ndvdot, nu);
    this->Fp.resize(ndvdot, np);
    this->dissipative_P.resize(1);
    this->dP_dv.resize(1, nv);
    this->dP_dp.resize(1, np);
    this->h.resize(nh);
    this->Hx.resize(nh, ndx);
    this->Hu.resize(nh, nu);
    this->Hp.resize(nh, np);
    this->g.resize(ng);
    this->Gx.resize(ng, ndx);
    this->Gu.resize(ng, nu);
    this->Gp.resize(ng, np);
    this->tmp_ustatic.resize(nu);

    dx.resize(model->get_model()->get_state()->get_ndx());
    du.resize(model->get_model()->get_nu());
    up.resize(model->get_model()->get_nu());
    dp.resize(np);
    p.conservativeResize(np);
    pp.resize(np);
    xp.resize(model->get_model()->get_state()->get_nx());
    dvdot.resize(model->get_model()->get_state()->get_ndx());

    const auto create_wrapped_data = [&]() {
      return params_data != nullptr
                 ? model->get_model()->createData(params_data)
                 : model->get_model()->createData();
    };
    if (data_0 == nullptr) {
      data_0 = create_wrapped_data();
    }
    while (data_x.size() < ndx) {
      data_x.push_back(create_wrapped_data());
    }
    data_x.resize(ndx);
    while (data_u.size() < nu) {
      data_u.push_back(create_wrapped_data());
    }
    data_u.resize(nu);
    while (data_p.size() < np) {
      data_p.push_back(create_wrapped_data());
    }
    data_p.resize(np);

    this->setZero();
    dx.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    pp.setZero();
    xp.setZero();
    dvdot.setZero();
  }

  Scalar x_norm;  //!< Norm used to scale the state disturbance
  Scalar xh_jac;  //!< State disturbance
  Scalar uh_jac;  //!< Control disturbance
  Scalar ph_jac;  //!< Parameter disturbance
  std::shared_ptr<ParameterDataManager> params_data;  //!< Shared parameter data
  VectorXs dx;                                        //!< State disturbance
  VectorXs du;                                        //!< Control disturbance
  VectorXs up;     //!< Perturbed control vector
  VectorXs dp;     //!< Parameter disturbance
  VectorXs p;      //!< Nominal active parameter vector
  VectorXs pp;     //!< Perturbed parameter vector
  VectorXs xp;     //!< Perturbed state
  VectorXs dvdot;  //!< Tangent-space buffer for DiscreteTime vdot diff
  std::shared_ptr<Base> data_0;  //!< Nominal wrapped data
  std::vector<std::shared_ptr<Base> > data_x;
  std::vector<std::shared_ptr<Base> > data_u;
  std::vector<std::shared_ptr<Base> > data_p;

  using Base::Fp;
  using Base::g;
  using Base::Gp;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hp;
  using Base::Hu;
  using Base::Hx;
  using Base::vdot;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/dynamics.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::DynamicsModelNumDiffTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DynamicsDataNumDiffTpl)

#endif  // CROCODDYL_CORE_NUMDIFF_DYNAMICS_HPP_
