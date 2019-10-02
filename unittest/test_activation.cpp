///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <iterator>
#include <Eigen/Dense>
#include <pinocchio/fwd.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl_unittest_common.hpp"

// # Comment:
// '''
// c = sum( a(ri) )
// c' = sum( [a(ri)]' ) = sum( ri' a'(ri) ) = R' [ a'(ri) ]_i
// c'' = R' [a'(ri) ]_i' = R' [a''(ri) ] R

// ex
// a(x) =  x**2/x
// a'(x) = x
// a''(x) = 1

// sum(a(ri)) = sum(ri**2/2) = .5*r'r
// sum(ri' a'(ri)) = sum(ri' ri) = R' r
// sum(ri' a''(ri) ri') = R' r
// c'' = R'R
// '''

using namespace boost::unit_test;

struct ActionModelTypes {
  enum Type { ActionModelUnicycle, ActionModelLQR };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    v.push_back(ActionModelUnicycle);
    v.push_back(ActionModelLQR);
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<ActionModelTypes::Type> ActionModelTypes::all(ActionModelTypes::init_all());

class ActionModelFactory {
 public:
  ActionModelFactory(ActionModelTypes::Type type) {
    nx_ = 80;
    nu_ = 40;
    driftfree_ = true;
    num_diff_modifier_ = 1e4;
    action_model_ = NULL;
    action_type_ = type;
    switch (action_type_) {
      case ActionModelTypes::ActionModelUnicycle:
        action_model_ = new crocoddyl::ActionModelUnicycle();
        break;

      case ActionModelTypes::ActionModelLQR:
        action_model_ = new crocoddyl::ActionModelLQR(nx_, nu_, driftfree_);
        break;

      default:
        throw std::runtime_error("test_actions.cpp: This type of ActionModel requested has not been implemented yet.");
        break;
    }
  }

  ~ActionModelFactory() {
    switch (action_type_) {
      case ActionModelTypes::ActionModelUnicycle:
        crocoddyl_unit_test::delete_pointer((crocoddyl::ActionModelUnicycle*)action_model_);
        break;

      case ActionModelTypes::ActionModelLQR:
        crocoddyl_unit_test::delete_pointer((crocoddyl::ActionModelLQR*)action_model_);
        break;

      default:
        throw std::runtime_error("test_actions.cpp: This type of ActionModel requested has not been implemented yet.");
        break;
    }
    action_model_ = NULL;
  }

  crocoddyl::ActionModelAbstract* get_action_model() { return action_model_; }

  double num_diff_modifier_;

 private:
  int nx_;
  int nu_;
  bool driftfree_;
  ActionModelTypes::Type action_type_;
  crocoddyl::ActionModelAbstract* action_model_;
};

// # - ------------------------------
// # --- Dim 1 ----------------------
// h = np.sqrt(2 * EPS)

// def df(am, ad, x):
//     return (am.calc(ad, x + h) - am.calc(ad, x)) / h

// def ddf(am, ad, x):
//     return (am.calcDiff(ad, x + h)[0] - am.calcDiff(ad, x)[0]) / h

// am = ActivationModelQuad()
// ad = am.createData()
// x = np.random.rand(1)

// am.calc(ad, x)
// assertNumDiff(df(am, ad, x),
//               am.calcDiff(ad, x)[0], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x),
//               am.calcDiff(ad, x)[1], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)

// am = ActivationModelWeightedQuad(np.random.rand(1))
// ad = am.createData()
// assertNumDiff(df(am, ad, x),
//               am.calcDiff(ad, x)[0], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x),
//               am.calcDiff(ad, x)[1], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)

// am = ActivationModelSmoothAbs()
// ad = am.createData()
// assertNumDiff(df(am, ad, x),
//               am.calcDiff(ad, x)[0], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x),
//               am.calcDiff(ad, x)[1], 1e-6)  # threshold was 1e-6, is now 1e-6 (see assertNumDiff.__doc__)

// # - ------------------------------
// # --- Dim N ----------------------

// def df(am, ad, x):
//     dx = x * 0
//     J = np.zeros([len(x), len(x)])
//     for i, _ in enumerate(x):
//         dx[i] = h
//         J[:, i] = (am.calc(ad, x + dx) - am.calc(ad, x)) / h
//         dx[i] = 0
//     return J

// def ddf(am, ad, x):
//     dx = x * 0
//     J = np.zeros([len(x), len(x)])
//     for i, _ in enumerate(x):
//         dx[i] = h
//         J[:, i] = (am.calcDiff(ad, x + dx)[0] - am.calcDiff(ad, x)[0]) / h
//         dx[i] = 0
//     return J
//     return

// x = np.random.rand(3)

// am = ActivationModelQuad()
// ad = am.createData()
// J = df(am, ad, x)
// H = ddf(am, ad, x)
// assertNumDiff(np.diag(J.diagonal()), J, 5e-8)  # threshold was 1e-9, is now 5e-8 (see assertNumDiff.__doc__)
// assertNumDiff(np.diag(H.diagonal()), H, 5e-8)  # threshold was 1e-9, is now 5e-8 (see assertNumDiff.__doc__)
// assertNumDiff(df(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[1][:, 0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)

// am = ActivationModelWeightedQuad(np.random.rand(len(x)))
// ad = am.createData()
// assertNumDiff(df(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[1][:, 0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)

// am = ActivationModelSmoothAbs()
// ad = am.createData()
// assertNumDiff(df(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)
// assertNumDiff(ddf(am, ad, x).diagonal(),
//               am.calcDiff(ad, x)[1][:, 0],
//               np.sqrt(2 * EPS))  # threshold was 1e-6, is now 2.11e-8 (see assertNumDiff.__doc__)
