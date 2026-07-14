///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <string>

#include "crocoddyl/core/optctrl/problem-abstract.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename _Scalar>
class ProblemActionProbeTpl
    : public crocoddyl::ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ActionModelBase, ProblemActionProbeTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> Base;
  typedef crocoddyl::ActionDataAbstractTpl<Scalar> Data;
  typedef typename Base::VectorXs VectorXs;

  ProblemActionProbeTpl()
      : Base(std::make_shared<crocoddyl::StateVectorTpl<Scalar> >(4), 2),
        running_calc(0),
        terminal_calc(0),
        running_calc_diff(0),
        terminal_calc_diff(0) {}

  void calc(const std::shared_ptr<Data>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>&) override {
    data->xnext = x;
    data->cost = Scalar(1);
    ++running_calc;
  }

  void calc(const std::shared_ptr<Data>& data,
            const Eigen::Ref<const VectorXs>&) override {
    data->cost = Scalar(10);
    ++terminal_calc;
  }

  void calcDiff(const std::shared_ptr<Data>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    data->cost = Scalar(2);
    ++running_calc_diff;
  }

  void calcDiff(const std::shared_ptr<Data>& data,
                const Eigen::Ref<const VectorXs>&) override {
    data->cost = Scalar(20);
    ++terminal_calc_diff;
  }

  bool checkData(const std::shared_ptr<Data>& data) override {
    return data != nullptr;
  }

  template <typename NewScalar>
  ProblemActionProbeTpl<NewScalar> cast() const {
    return ProblemActionProbeTpl<NewScalar>();
  }

  std::size_t running_calc;
  std::size_t terminal_calc;
  std::size_t running_calc_diff;
  std::size_t terminal_calc_diff;
};

template <typename _Scalar>
class PhasedShootingProblemProbeTpl
    : public crocoddyl::ShootingProblemTpl<_Scalar> {
 public:
  typedef crocoddyl::ShootingProblemTpl<_Scalar> Base;
  using Base::Base;

  std::size_t get_n_phases() const override { return 1; }
};

template <typename Scalar>
void test_shooting_problem_polymorphic_interface() {
  typedef ProblemActionProbeTpl<Scalar> ActionModel;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ProblemAbstract::VectorXs VectorXs;

  const std::size_t T = 3;
  const std::shared_ptr<ActionModel> model = std::make_shared<ActionModel>();
  const std::vector<
      std::shared_ptr<typename ProblemAbstract::ActionModelAbstract> >
      models(T, model);
  const VectorXs x0 = VectorXs::LinSpaced(4, Scalar(-0.2), Scalar(0.4));
  ShootingProblem problem(x0, models, model);
  problem.set_nthreads(1);
  ProblemAbstract& base = problem;

  BOOST_CHECK_EQUAL(base.get_T(), T);
  BOOST_CHECK(base.get_x0().isApprox(x0));
  BOOST_CHECK_EQUAL(base.get_nx(), 4u);
  BOOST_CHECK_EQUAL(base.get_ndx(), 4u);
  BOOST_CHECK_EQUAL(base.get_nthreads(), problem.get_nthreads());
  BOOST_REQUIRE_EQUAL(base.get_runningModels().size(), T);
  BOOST_REQUIRE_EQUAL(base.get_runningDatas().size(), T);
  BOOST_CHECK(base.get_runningModels()[0] == model);
  BOOST_CHECK(base.get_terminalModel() == model);
  BOOST_CHECK(base.get_runningDatas()[0] == problem.get_runningDatas()[0]);
  BOOST_CHECK(base.get_terminalData() == problem.get_terminalData());

  std::vector<VectorXs> xs(T + 1, VectorXs::Zero(4));
  std::vector<VectorXs> us(T, VectorXs::Zero(2));
  BOOST_CHECK_EQUAL(base.calc(xs, us), Scalar(T + 10));
  BOOST_CHECK_EQUAL(model->running_calc, T);
  BOOST_CHECK_EQUAL(model->terminal_calc, 1u);

  BOOST_CHECK_EQUAL(base.calcDiff(xs, us), Scalar(2 * T + 20));
  BOOST_CHECK_EQUAL(model->running_calc_diff, T);
  BOOST_CHECK_EQUAL(model->terminal_calc_diff, 1u);

  base.rollout(us, xs);
  BOOST_CHECK(xs[0].isApprox(x0));
  for (std::size_t i = 1; i < xs.size(); ++i) {
    BOOST_CHECK(xs[i].isApprox(x0));
  }
  const std::vector<VectorXs> default_xs = base.ProblemAbstract::rollout_us(us);
  BOOST_REQUIRE_EQUAL(default_xs.size(), T + 1);
  for (std::size_t i = 0; i < default_xs.size(); ++i) {
    BOOST_CHECK(default_xs[i].isApprox(x0));
  }

  BOOST_CHECK(!base.is_updated());
  base.set_is_updated(true);
  BOOST_CHECK(base.is_updated());
  BOOST_CHECK(!base.is_updated());

  BOOST_CHECK_EQUAL(base.get_n_phases(), 0u);
  BOOST_CHECK(base.get_phase_idxs().empty());
  BOOST_CHECK(base.get_phase_edxs().empty());
  BOOST_CHECK(base.get_parameter_constraints_models().empty());
  BOOST_CHECK(base.get_parameter_constraints_datas().empty());
  BOOST_CHECK(!base.has_parameter_constraints());
  BOOST_CHECK_THROW(base.update_p(VectorXs::Zero(1)), crocoddyl::Exception);
}

template <typename Scalar>
void test_shooting_problem_existing_data_constructor() {
  typedef ProblemActionProbeTpl<Scalar> ActionModel;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ProblemAbstract::VectorXs VectorXs;

  const std::size_t T = 2;
  const std::shared_ptr<ActionModel> model = std::make_shared<ActionModel>();
  const std::vector<
      std::shared_ptr<typename ProblemAbstract::ActionModelAbstract> >
      models(T, model);
  std::vector<std::shared_ptr<typename ProblemAbstract::ActionDataAbstract> >
      datas(T);
  for (std::size_t i = 0; i < T; ++i) {
    datas[i] = model->createData();
  }
  const std::shared_ptr<typename ProblemAbstract::ActionDataAbstract>
      terminal_data = model->createData();
  ShootingProblem problem(VectorXs::Zero(4), models, model, datas,
                          terminal_data);
  ProblemAbstract& base = problem;

  BOOST_CHECK(base.get_runningDatas()[0] == datas[0]);
  BOOST_CHECK(base.get_runningDatas()[1] == datas[1]);
  BOOST_CHECK(base.get_terminalData() == terminal_data);
  BOOST_CHECK_EQUAL(base.get_T(), T);
}

template <typename Scalar>
void test_shooting_problem_structural_mutations() {
  typedef ProblemActionProbeTpl<Scalar> ActionModel;
  typedef crocoddyl::ShootingProblemTpl<Scalar> ShootingProblem;
  typedef PhasedShootingProblemProbeTpl<Scalar> PhasedShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef typename ShootingProblem::VectorXs VectorXs;

  const std::shared_ptr<ActionModel> model = std::make_shared<ActionModel>();
  const std::shared_ptr<ActionModel> replacement =
      std::make_shared<ActionModel>();
  const std::vector<std::shared_ptr<ActionModelAbstract> > models(2, model);
  PhasedShootingProblem phased(VectorXs::Zero(4), models, model);
  ShootingProblem& canonical = phased;
  const std::vector<std::shared_ptr<ActionModelAbstract> > original_models =
      canonical.get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> > original_datas =
      canonical.get_runningDatas();
  const std::shared_ptr<ActionModelAbstract> original_terminal_model =
      canonical.get_terminalModel();
  const std::shared_ptr<ActionDataAbstract> original_terminal_data =
      canonical.get_terminalData();
  const auto requires_reconstruction = [](const crocoddyl::Exception& e) {
    return std::string(e.what()).find("must be reconstructed") !=
           std::string::npos;
  };

  BOOST_CHECK_EXCEPTION(
      canonical.circularAppend(std::shared_ptr<ActionModelAbstract>(),
                               std::shared_ptr<ActionDataAbstract>()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK_EXCEPTION(
      canonical.circularAppend(std::shared_ptr<ActionModelAbstract>()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK_EXCEPTION(
      canonical.updateNode(3, std::shared_ptr<ActionModelAbstract>(),
                           std::shared_ptr<ActionDataAbstract>()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK_EXCEPTION(
      canonical.updateModel(3, std::shared_ptr<ActionModelAbstract>()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK_EXCEPTION(
      canonical.set_runningModels(
          std::vector<std::shared_ptr<ActionModelAbstract> >()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK_EXCEPTION(
      canonical.set_terminalModel(std::shared_ptr<ActionModelAbstract>()),
      crocoddyl::Exception, requires_reconstruction);
  BOOST_CHECK(canonical.get_runningModels() == original_models);
  BOOST_CHECK(canonical.get_runningDatas() == original_datas);
  BOOST_CHECK(canonical.get_terminalModel() == original_terminal_model);
  BOOST_CHECK(canonical.get_terminalData() == original_terminal_data);
  BOOST_CHECK_EQUAL(canonical.get_T(), 2u);

  ShootingProblem ordinary(VectorXs::Zero(4), models, model);
  const std::shared_ptr<ActionDataAbstract> replacement_data =
      replacement->createData();
  ordinary.circularAppend(replacement, replacement_data);
  BOOST_CHECK(ordinary.get_runningModels().back() == replacement);
  BOOST_CHECK(ordinary.get_runningDatas().back() == replacement_data);
  ordinary.circularAppend(model);
  BOOST_CHECK(ordinary.get_runningModels().back() == model);
  BOOST_CHECK(ordinary.get_runningDatas().back() != nullptr);
  ordinary.updateNode(0, replacement, replacement_data);
  BOOST_CHECK(ordinary.get_runningModels()[0] == replacement);
  BOOST_CHECK(ordinary.get_runningDatas()[0] == replacement_data);
  ordinary.updateModel(1, replacement);
  BOOST_CHECK(ordinary.get_runningModels()[1] == replacement);
  BOOST_CHECK(ordinary.get_runningDatas()[1] != nullptr);
  ordinary.set_runningModels(models);
  BOOST_CHECK(ordinary.get_runningModels() == models);
  BOOST_CHECK_EQUAL(ordinary.get_runningDatas().size(), models.size());
  ordinary.set_terminalModel(replacement);
  BOOST_CHECK(ordinary.get_terminalModel() == replacement);
  BOOST_CHECK(ordinary.get_terminalData() != nullptr);
}

template <typename Scalar>
void test_shooting_problem_no_allocation() {
  typedef ProblemActionProbeTpl<Scalar> ActionModel;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ProblemAbstract::VectorXs VectorXs;

  const std::size_t T = 3;
  const std::shared_ptr<ActionModel> model = std::make_shared<ActionModel>();
  const std::vector<
      std::shared_ptr<typename ProblemAbstract::ActionModelAbstract> >
      models(T, model);
  ShootingProblem problem(VectorXs::Zero(4), models, model);
  problem.set_nthreads(1);
  ProblemAbstract& base = problem;
  std::vector<VectorXs> xs(T + 1, VectorXs::Zero(4));
  const std::vector<VectorXs> us(T, VectorXs::Zero(2));

  base.calc(xs, us);
  base.calcDiff(xs, us);
  base.rollout(us, xs);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      base.calc(xs, us);
      base.calcDiff(xs, us);
      base.rollout(us, xs);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_problem_abstract");
  ts->add(
      BOOST_TEST_CASE(&test_shooting_problem_polymorphic_interface<double>));
  ts->add(BOOST_TEST_CASE(&test_shooting_problem_polymorphic_interface<float>));
  ts->add(BOOST_TEST_CASE(
      &test_shooting_problem_existing_data_constructor<double>));
  ts->add(
      BOOST_TEST_CASE(&test_shooting_problem_existing_data_constructor<float>));
  ts->add(BOOST_TEST_CASE(&test_shooting_problem_structural_mutations<double>));
  ts->add(BOOST_TEST_CASE(&test_shooting_problem_structural_mutations<float>));
  ts->add(BOOST_TEST_CASE(&test_shooting_problem_no_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_shooting_problem_no_allocation<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
