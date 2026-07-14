///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

template <typename Scalar>
struct LQRFixtureTpl {
  typedef crocoddyl::MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::VectorXs VectorXs;

  LQRFixtureTpl(const std::size_t nx, const std::size_t nu,
                const std::size_t np, const std::size_t ng,
                const std::size_t nh)
      : A(nx, nx),
        B(nx, nu),
        P(nx, np),
        Q(nx, nx),
        R(nu, nu),
        N(nx, nu),
        W(np, np),
        Y(nx, np),
        V(nu, np),
        G(ng, nx + nu + np),
        H(nh, nx + nu + np),
        f(nx),
        q(nx),
        r(nu),
        m(np),
        g(ng),
        h(nh) {
    const std::size_t nz = nx + nu + np;
    MatrixXs factor(nz, nz);
    for (std::size_t i = 0; i < nz; ++i) {
      for (std::size_t j = 0; j < nz; ++j) {
        factor(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            Scalar(0.01 * static_cast<double>(1 + i + 2 * j));
      }
    }
    const MatrixXs L =
        factor.transpose() * factor + Scalar(2) * MatrixXs::Identity(nz, nz);
    Q = L.topLeftCorner(nx, nx);
    R = L.block(nx, nx, nu, nu);
    N = L.block(0, nx, nx, nu);
    W = L.bottomRightCorner(np, np);
    Y = L.block(0, nx + nu, nx, np);
    V = L.block(nx, nx + nu, nu, np);

    fill_matrix(A, Scalar(0.11));
    fill_matrix(B, Scalar(-0.07));
    fill_matrix(P, Scalar(0.13));
    fill_matrix(G, Scalar(-0.09));
    fill_matrix(H, Scalar(0.17));
    fill_vector(f, Scalar(0.2));
    fill_vector(q, Scalar(-0.3));
    fill_vector(r, Scalar(0.4));
    fill_vector(m, Scalar(-0.5));
    fill_vector(g, Scalar(0.6));
    fill_vector(h, Scalar(-0.7));
  }

  static void fill_matrix(MatrixXs& matrix, const Scalar offset) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        matrix(i, j) =
            offset + Scalar(0.03) * Scalar(1 + i * matrix.cols() + j);
      }
    }
  }

  static void fill_vector(VectorXs& vector, const Scalar offset) {
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
      vector[i] = offset + Scalar(0.05) * Scalar(i + 1);
    }
  }

  MatrixXs A;
  MatrixXs B;
  MatrixXs P;
  MatrixXs Q;
  MatrixXs R;
  MatrixXs N;
  MatrixXs W;
  MatrixXs Y;
  MatrixXs V;
  MatrixXs G;
  MatrixXs H;
  VectorXs f;
  VectorXs q;
  VectorXs r;
  VectorXs m;
  VectorXs g;
  VectorXs h;
};

template <typename Scalar>
Scalar tolerance() {
  return Scalar(100) * std::sqrt(std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
void test_constructors_and_zero_parameter_compatibility() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t ng = 2;
  const std::size_t nh = 1;
  LQRFixtureTpl<Scalar> legacy(nx, nu, 0, ng, nh);

  Model matrices(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N);
  BOOST_CHECK_EQUAL(matrices.get_np(), 0u);
  BOOST_CHECK(matrices.get_f().isZero());
  BOOST_CHECK(matrices.get_q().isZero());
  BOOST_CHECK(matrices.get_r().isZero());

  Model affine(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N, legacy.f,
               legacy.q, legacy.r);
  BOOST_CHECK(affine.get_f().isApprox(legacy.f));
  BOOST_CHECK(affine.get_q().isApprox(legacy.q));
  BOOST_CHECK(affine.get_r().isApprox(legacy.r));

  Model constrained(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N, legacy.G,
                    legacy.H, legacy.f, legacy.q, legacy.r, legacy.g, legacy.h);
  BOOST_CHECK_EQUAL(constrained.get_ng(), ng);
  BOOST_CHECK_EQUAL(constrained.get_nh(), nh);

  Model legacy_size(nx, nu, false);
  Model parameter_size(nx, nu, 0, 0, 0, false);
  BOOST_CHECK(legacy_size.get_A().isApprox(parameter_size.get_A()));
  BOOST_CHECK(legacy_size.get_B().isApprox(parameter_size.get_B()));
  BOOST_CHECK(legacy_size.get_P().isApprox(parameter_size.get_P()));
  BOOST_CHECK(legacy_size.get_Q().isApprox(parameter_size.get_Q()));
  BOOST_CHECK(legacy_size.get_R().isApprox(parameter_size.get_R()));
  BOOST_CHECK(legacy_size.get_N().isApprox(parameter_size.get_N()));
  BOOST_CHECK(legacy_size.get_W().isApprox(parameter_size.get_W()));
  BOOST_CHECK(legacy_size.get_Y().isApprox(parameter_size.get_Y()));
  BOOST_CHECK(legacy_size.get_V().isApprox(parameter_size.get_V()));
  BOOST_CHECK(legacy_size.get_f().isApprox(parameter_size.get_f()));
  BOOST_CHECK(legacy_size.get_q().isApprox(parameter_size.get_q()));
  BOOST_CHECK(legacy_size.get_r().isApprox(parameter_size.get_r()));
  BOOST_CHECK_EQUAL(legacy_size.get_np(), 0u);

  const std::shared_ptr<typename Model::ActionDataAbstract> legacy_data =
      legacy_size.createData();
  const std::shared_ptr<typename Model::ActionDataAbstract> parameter_data =
      parameter_size.createData();
  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.4), Scalar(0.7));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.2), Scalar(0.8));
  legacy_size.calc(legacy_data, x, u);
  parameter_size.calc(parameter_data, x, u);
  legacy_size.calcDiff(legacy_data, x, u);
  parameter_size.calcDiff(parameter_data, x, u);
  BOOST_CHECK(legacy_data->xnext.isApprox(parameter_data->xnext));
  BOOST_CHECK_SMALL(
      static_cast<double>(legacy_data->cost - parameter_data->cost),
      static_cast<double>(tolerance<Scalar>()));
  BOOST_CHECK(legacy_data->Lx.isApprox(parameter_data->Lx));
  BOOST_CHECK(legacy_data->Lu.isApprox(parameter_data->Lu));

  Model copied(constrained);
  BOOST_CHECK(copied.get_A().isApprox(constrained.get_A()));
  BOOST_CHECK(copied.get_G().isApprox(constrained.get_G()));
  BOOST_CHECK(copied.get_h().isApprox(constrained.get_h()));

  Model random = Model::Random(nx, nu, ng, nh);
  BOOST_CHECK_EQUAL(random.get_np(), 0u);
  BOOST_CHECK_EQUAL(random.get_state()->get_nx(), nx);
  BOOST_CHECK_EQUAL(random.get_nu(), nu);
  BOOST_CHECK_EQUAL(random.get_ng(), ng);
  BOOST_CHECK_EQUAL(random.get_nh(), nh);
  BOOST_CHECK(random.checkData(random.createData()));
}

template <typename Scalar>
void test_parameterized_running_terminal_formulas_and_data() {
  typedef crocoddyl::ActionDataLQRTpl<Scalar> Data;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  const std::size_t ng = 2;
  const std::size_t nh = 1;
  const Scalar tol = tolerance<Scalar>();
  LQRFixtureTpl<Scalar> values(nx, nu, np, ng, nh);
  Model model(values.A, values.B, values.P, values.Q, values.R, values.N,
              values.W, values.Y, values.V, values.G, values.H, values.f,
              values.q, values.r, values.m, values.g, values.h);
  BOOST_CHECK_EQUAL(model.get_np(), np);

  const std::shared_ptr<typename Model::ActionDataAbstract> abstract_data =
      model.createData();
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(abstract_data);
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(model.checkData(data));
  BOOST_CHECK_EQUAL(data->p.size(), static_cast<Eigen::Index>(np));
  BOOST_CHECK(data->p.isZero());
  BOOST_CHECK(data->params == nullptr);
  BOOST_CHECK(data->Fp.isApprox(values.P));
  BOOST_CHECK(data->Lpp.isApprox(values.W));
  BOOST_CHECK(data->Lpx.isApprox(values.Y.transpose()));
  BOOST_CHECK(data->Lpu.isApprox(values.V.transpose()));
  BOOST_CHECK(data->Gp.isApprox(values.G.rightCols(np)));
  BOOST_CHECK(data->Hp.isApprox(values.H.rightCols(np)));

  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.4), Scalar(0.8));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.3), Scalar(0.9));
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(-0.6), Scalar(0.5));
  model.update_p(data, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK_THROW(model.update_p(data, VectorXs::Zero(np + 1)),
                    std::exception);

  model.calc(data, x, u);
  VectorXs expected_xnext = values.A * x + values.B * u + values.P * p;
  expected_xnext += values.f;
  Scalar expected_cost =
      Scalar(0.5) * x.dot(values.Q * x) + Scalar(0.5) * u.dot(values.R * u) +
      x.dot(values.N * u) + Scalar(0.5) * p.dot(values.W * p) +
      x.dot(values.Y * p) + u.dot(values.V * p) + values.q.dot(x) +
      values.r.dot(u) + values.m.dot(p);
  VectorXs expected_g = values.G.leftCols(nx) * x;
  expected_g.noalias() += values.G.middleCols(nx, nu) * u;
  expected_g.noalias() += values.G.rightCols(np) * p;
  expected_g += values.g;
  VectorXs expected_h = values.H.leftCols(nx) * x;
  expected_h.noalias() += values.H.middleCols(nx, nu) * u;
  expected_h.noalias() += values.H.rightCols(np) * p;
  expected_h += values.h;
  BOOST_CHECK(data->xnext.isApprox(expected_xnext, tol));
  BOOST_CHECK_SMALL(static_cast<double>(data->cost - expected_cost),
                    static_cast<double>(tol));
  BOOST_CHECK(data->g.isApprox(expected_g, tol));
  BOOST_CHECK(data->h.isApprox(expected_h, tol));

  model.calcDiff(data, x, u);
  BOOST_CHECK(data->Fx.isApprox(values.A, tol));
  BOOST_CHECK(data->Fu.isApprox(values.B, tol));
  BOOST_CHECK(data->Fp.isApprox(values.P, tol));
  BOOST_CHECK(data->Lx.isApprox(
      values.q + values.Q * x + values.N * u + values.Y * p, tol));
  BOOST_CHECK(data->Lu.isApprox(
      values.r + values.N.transpose() * x + values.R * u + values.V * p, tol));
  BOOST_CHECK(data->Lp.isApprox(values.m + values.Y.transpose() * x +
                                    values.V.transpose() * u + values.W * p,
                                tol));
  BOOST_CHECK(data->Lxx.isApprox(values.Q, tol));
  BOOST_CHECK(data->Lxu.isApprox(values.N, tol));
  BOOST_CHECK(data->Luu.isApprox(values.R, tol));
  BOOST_CHECK(data->Lpp.isApprox(values.W, tol));
  BOOST_CHECK(data->Lpx.isApprox(values.Y.transpose(), tol));
  BOOST_CHECK(data->Lpu.isApprox(values.V.transpose(), tol));
  BOOST_CHECK(data->Gx.isApprox(values.G.leftCols(nx), tol));
  BOOST_CHECK(data->Gu.isApprox(values.G.middleCols(nx, nu), tol));
  BOOST_CHECK(data->Gp.isApprox(values.G.rightCols(np), tol));
  BOOST_CHECK(data->Hx.isApprox(values.H.leftCols(nx), tol));
  BOOST_CHECK(data->Hu.isApprox(values.H.middleCols(nx, nu), tol));
  BOOST_CHECK(data->Hp.isApprox(values.H.rightCols(np), tol));

  data->Lu.setConstant(Scalar(21));
  data->Luu.setConstant(Scalar(22));
  data->Lxu.setConstant(Scalar(23));
  data->Lpu.setConstant(Scalar(24));
  model.calc(data, x);
  const Scalar terminal_cost =
      Scalar(0.5) * x.dot(values.Q * x) + Scalar(0.5) * p.dot(values.W * p) +
      x.dot(values.Y * p) + values.q.dot(x) + values.m.dot(p);
  expected_g = values.G.leftCols(nx) * x;
  expected_g.noalias() += values.G.rightCols(np) * p;
  expected_g += values.g;
  expected_h = values.H.leftCols(nx) * x;
  expected_h.noalias() += values.H.rightCols(np) * p;
  expected_h += values.h;
  BOOST_CHECK(data->xnext.isApprox(x, tol));
  BOOST_CHECK_SMALL(static_cast<double>(data->cost - terminal_cost),
                    static_cast<double>(tol));
  BOOST_CHECK(data->g.isApprox(expected_g, tol));
  BOOST_CHECK(data->h.isApprox(expected_h, tol));

  model.calcDiff(data, x);
  BOOST_CHECK(data->Lx.isApprox(values.q + values.Q * x + values.Y * p, tol));
  BOOST_CHECK(data->Lp.isApprox(
      values.m + values.Y.transpose() * x + values.W * p, tol));
  BOOST_CHECK(data->Lpp.isApprox(values.W, tol));
  BOOST_CHECK(data->Lpx.isApprox(values.Y.transpose(), tol));
  BOOST_CHECK(data->Gp.isApprox(values.G.rightCols(np), tol));
  BOOST_CHECK(data->Hp.isApprox(values.H.rightCols(np), tol));
  BOOST_CHECK(data->Lu.isConstant(Scalar(21)));
  BOOST_CHECK(data->Luu.isConstant(Scalar(22)));
  BOOST_CHECK(data->Lxu.isConstant(Scalar(23)));
  BOOST_CHECK(data->Lpu.isConstant(Scalar(24)));

  Data copied(*data);
  BOOST_CHECK(copied.p.isApprox(p));
  BOOST_CHECK(copied.Fp.isApprox(data->Fp));
  BOOST_CHECK(copied.Lp.isApprox(data->Lp));
  BOOST_CHECK(copied.Lpp.isApprox(data->Lpp));
  BOOST_CHECK(copied.Lpx.isApprox(data->Lpx));
  BOOST_CHECK(copied.Lpu.isApprox(data->Lpu));
  BOOST_CHECK(copied.Gp.isApprox(data->Gp));
  BOOST_CHECK(copied.Hp.isApprox(data->Hp));
  data->p.setZero();
  data->Fp.setZero();
  BOOST_CHECK(copied.p.isApprox(p));
  BOOST_CHECK(copied.Fp.isApprox(values.P));
}

template <typename Scalar>
void test_constant_derivatives_refresh_per_data() {
  typedef crocoddyl::ActionDataLQRTpl<Scalar> Data;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  const std::size_t ng = 2;
  const std::size_t nh = 1;
  const Scalar tol = tolerance<Scalar>();
  LQRFixtureTpl<Scalar> values(nx, nu, np, ng, nh);
  Model model(values.A, values.B, values.P, values.Q, values.R, values.N,
              values.W, values.Y, values.V, values.G, values.H, values.f,
              values.q, values.r, values.m, values.g, values.h);
  const std::shared_ptr<Data> first =
      std::dynamic_pointer_cast<Data>(model.createData());
  const std::shared_ptr<Data> second =
      std::dynamic_pointer_cast<Data>(model.createData());
  BOOST_REQUIRE(first != nullptr);
  BOOST_REQUIRE(second != nullptr);

  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.4), Scalar(0.8));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.3), Scalar(0.9));
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(-0.6), Scalar(0.5));
  model.update_p(first, p);
  model.update_p(second, p);
  model.calc(first, x, u);
  model.calcDiff(first, x, u);
  model.calc(second, x, u);
  model.calcDiff(second, x, u);

  LQRFixtureTpl<Scalar> updated(nx, nu, np, ng, nh);
  updated.A.array() += Scalar(0.31);
  updated.B.array() -= Scalar(0.27);
  updated.P.array() += Scalar(0.23);
  updated.Q *= Scalar(1.5);
  updated.R *= Scalar(1.5);
  updated.N *= Scalar(1.5);
  updated.W *= Scalar(1.5);
  updated.Y *= Scalar(1.5);
  updated.V *= Scalar(1.5);
  updated.G.array() += Scalar(0.19);
  updated.H.array() -= Scalar(0.17);
  model.set_LQR(updated.A, updated.B, updated.P, updated.Q, updated.R,
                updated.N, updated.W, updated.Y, updated.V, updated.G,
                updated.H, updated.f, updated.q, updated.r, updated.m,
                updated.g, updated.h);

  first->Fu.setConstant(Scalar(31));
  first->Lu.setConstant(Scalar(32));
  first->Luu.setConstant(Scalar(33));
  first->Lxu.setConstant(Scalar(34));
  first->Lpu.setConstant(Scalar(35));
  first->Gu.setConstant(Scalar(36));
  first->Hu.setConstant(Scalar(37));
  model.calcDiff(first, x);
  BOOST_CHECK(first->Lxx.isApprox(updated.Q, tol));
  BOOST_CHECK(first->Lpp.isApprox(updated.W, tol));
  BOOST_CHECK(first->Lpx.isApprox(updated.Y.transpose(), tol));
  BOOST_CHECK(first->Gx.isApprox(updated.G.leftCols(nx), tol));
  BOOST_CHECK(first->Gp.isApprox(updated.G.rightCols(np), tol));
  BOOST_CHECK(first->Hx.isApprox(updated.H.leftCols(nx), tol));
  BOOST_CHECK(first->Hp.isApprox(updated.H.rightCols(np), tol));
  BOOST_CHECK(first->Fu.isConstant(Scalar(31)));
  BOOST_CHECK(first->Lu.isConstant(Scalar(32)));
  BOOST_CHECK(first->Luu.isConstant(Scalar(33)));
  BOOST_CHECK(first->Lxu.isConstant(Scalar(34)));
  BOOST_CHECK(first->Lpu.isConstant(Scalar(35)));
  BOOST_CHECK(first->Gu.isConstant(Scalar(36)));
  BOOST_CHECK(first->Hu.isConstant(Scalar(37)));

  model.calcDiff(first, x, u);
  model.calcDiff(second, x, u);
  const std::shared_ptr<Data> data_objects[] = {first, second};
  for (const std::shared_ptr<Data>& data : data_objects) {
    BOOST_CHECK(data->Fx.isApprox(updated.A, tol));
    BOOST_CHECK(data->Fu.isApprox(updated.B, tol));
    BOOST_CHECK(data->Fp.isApprox(updated.P, tol));
    BOOST_CHECK(data->Lxx.isApprox(updated.Q, tol));
    BOOST_CHECK(data->Lxu.isApprox(updated.N, tol));
    BOOST_CHECK(data->Luu.isApprox(updated.R, tol));
    BOOST_CHECK(data->Lpp.isApprox(updated.W, tol));
    BOOST_CHECK(data->Lpx.isApprox(updated.Y.transpose(), tol));
    BOOST_CHECK(data->Lpu.isApprox(updated.V.transpose(), tol));
    BOOST_CHECK(data->Gx.isApprox(updated.G.leftCols(nx), tol));
    BOOST_CHECK(data->Gu.isApprox(updated.G.middleCols(nx, nu), tol));
    BOOST_CHECK(data->Gp.isApprox(updated.G.rightCols(np), tol));
    BOOST_CHECK(data->Hx.isApprox(updated.H.leftCols(nx), tol));
    BOOST_CHECK(data->Hu.isApprox(updated.H.middleCols(nx, nu), tol));
    BOOST_CHECK(data->Hp.isApprox(updated.H.rightCols(np), tol));
  }
}

template <typename Scalar>
void test_setters_set_lqr_and_dimension_errors() {
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef typename Model::MatrixXs MatrixXs;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  const std::size_t ng = 2;
  const std::size_t nh = 1;
  LQRFixtureTpl<Scalar> values(nx, nu, np, ng, nh);
  Model model(nx, nu, np, ng, nh, false);

  model.set_A(values.A);
  model.set_B(values.B);
  model.set_P(values.P);
  model.set_Q(values.Q);
  model.set_R(values.R);
  model.set_N(values.N);
  model.set_W(values.W);
  model.set_Y(values.Y);
  model.set_V(values.V);
  model.set_G(values.G);
  model.set_H(values.H);
  model.set_f(values.f);
  model.set_q(values.q);
  model.set_r(values.r);
  model.set_m(values.m);
  model.set_g(values.g);
  model.set_h(values.h);
  BOOST_CHECK(model.get_A().isApprox(values.A));
  BOOST_CHECK(model.get_B().isApprox(values.B));
  BOOST_CHECK(model.get_P().isApprox(values.P));
  BOOST_CHECK(model.get_Q().isApprox(values.Q));
  BOOST_CHECK(model.get_R().isApprox(values.R));
  BOOST_CHECK(model.get_N().isApprox(values.N));
  BOOST_CHECK(model.get_W().isApprox(values.W));
  BOOST_CHECK(model.get_Y().isApprox(values.Y));
  BOOST_CHECK(model.get_V().isApprox(values.V));
  BOOST_CHECK(model.get_G().isApprox(values.G));
  BOOST_CHECK(model.get_H().isApprox(values.H));
  BOOST_CHECK(model.get_f().isApprox(values.f));
  BOOST_CHECK(model.get_q().isApprox(values.q));
  BOOST_CHECK(model.get_r().isApprox(values.r));
  BOOST_CHECK(model.get_m().isApprox(values.m));
  BOOST_CHECK(model.get_g().isApprox(values.g));
  BOOST_CHECK(model.get_h().isApprox(values.h));

  model.set_LQR(values.A, values.B, values.P, values.Q, values.R, values.N,
                values.W, values.Y, values.V, values.G, values.H, values.f,
                values.q, values.r, values.m, values.g, values.h);
  BOOST_CHECK(model.get_P().isApprox(values.P));
  BOOST_CHECK(model.get_W().isApprox(values.W));

  BOOST_CHECK_THROW(model.set_A(MatrixXs::Zero(nx + 1, nx)), std::exception);
  BOOST_CHECK_THROW(model.set_B(MatrixXs::Zero(nx, nu + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_P(MatrixXs::Zero(nx, np + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_Q(MatrixXs::Zero(nx + 1, nx + 1)),
                    std::exception);
  BOOST_CHECK_THROW(model.set_R(MatrixXs::Zero(nu + 1, nu + 1)),
                    std::exception);
  BOOST_CHECK_THROW(model.set_N(MatrixXs::Zero(nx, nu + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_W(MatrixXs::Zero(np + 1, np + 1)),
                    std::exception);
  BOOST_CHECK_THROW(model.set_Y(MatrixXs::Zero(nx, np + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_V(MatrixXs::Zero(nu, np + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_G(MatrixXs::Zero(ng, nx + nu + np + 1)),
                    std::exception);
  BOOST_CHECK_THROW(model.set_H(MatrixXs::Zero(nh, nx + nu + np + 1)),
                    std::exception);
  BOOST_CHECK_THROW(model.set_f(VectorXs::Zero(nx + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_q(VectorXs::Zero(nx + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_r(VectorXs::Zero(nu + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_m(VectorXs::Zero(np + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_g(VectorXs::Zero(ng + 1)), std::exception);
  BOOST_CHECK_THROW(model.set_h(VectorXs::Zero(nh + 1)), std::exception);

  BOOST_CHECK_THROW(
      Model(values.A, values.B, MatrixXs::Zero(nx, np + 1), values.Q, values.R,
            values.N, values.W, values.Y, values.V, values.G, values.H,
            values.f, values.q, values.r, values.m, values.g, values.h),
      std::exception);
  BOOST_CHECK_THROW(
      model.set_LQR(values.A, values.B, values.P, values.Q, values.R, values.N,
                    values.W, values.Y, values.V,
                    MatrixXs::Zero(ng, nx + nu + np + 1), values.H, values.f,
                    values.q, values.r, values.m, values.g, values.h),
      std::exception);

  LQRFixtureTpl<Scalar> legacy(nx, nu, 0, ng, nh);
  Model legacy_model(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N, legacy.G,
                     legacy.H, legacy.f, legacy.q, legacy.r, legacy.g,
                     legacy.h);
  legacy_model.set_LQR(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N,
                       legacy.G, legacy.H, legacy.f, legacy.q, legacy.r,
                       legacy.g, legacy.h);
  BOOST_CHECK_THROW(
      legacy_model.set_LQR(legacy.A, legacy.B, legacy.Q, legacy.R, legacy.N,
                           MatrixXs::Zero(ng, nx + nu + 1), legacy.H, legacy.f,
                           legacy.q, legacy.r, legacy.g, legacy.h),
      std::exception);
}

template <typename Scalar>
void test_parameter_model_manager_and_scalar_casts() {
  typedef crocoddyl::ActionDataLQRTpl<Scalar> Data;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::StateVectorTpl<Scalar> State;
  typedef typename Model::VectorXs VectorXs;
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  const std::shared_ptr<State> state = std::make_shared<State>(nx);
  const std::shared_ptr<Params> params = std::make_shared<Params>(state, np);
  Params params_from_dimension(nx, np);
  BOOST_CHECK_EQUAL(params_from_dimension.get_state()->get_nx(), nx);
  BOOST_CHECK_EQUAL(params_from_dimension.get_np(), np);
  const VectorXs lb = VectorXs::Constant(np, Scalar(-2));
  const VectorXs ub = VectorXs::Constant(np, Scalar(3));
  params->set_lb(lb);
  params->set_ub(ub);
  const std::shared_ptr<typename Params::ParamsDataAbstract> params_data =
      params->createData();
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(-0.25), Scalar(0.75));
  params->update(params_data, p);
  BOOST_CHECK(params_data->p.isApprox(p));
  BOOST_CHECK_THROW(params->update(params_data, VectorXs::Zero(np + 1)),
                    std::exception);

  Model sensitivity_model(nx, nu, np);
  const std::shared_ptr<typename Model::ActionDataAbstract> sensitivity_data =
      sensitivity_model.createData();
  params_data->dx_dp.setOnes();
  params->computeParamSensitivity(sensitivity_data, params_data,
                                  VectorXs::Zero(nx), VectorXs::Zero(nu));
  BOOST_CHECK(params_data->dx_dp.isZero());

  const crocoddyl::ParamsModelBase& params_base = *params;
  const std::shared_ptr<crocoddyl::ParamsAbstractTpl<OtherScalar> >
      casted_params = params_base.template cast<OtherScalar>();
  const std::shared_ptr<crocoddyl::LQRParamsTpl<OtherScalar> >
      casted_lqr_params =
          std::dynamic_pointer_cast<crocoddyl::LQRParamsTpl<OtherScalar> >(
              casted_params);
  BOOST_REQUIRE(casted_lqr_params != nullptr);
  BOOST_CHECK(
      casted_lqr_params->get_lb().isApprox(lb.template cast<OtherScalar>()));
  BOOST_CHECK(
      casted_lqr_params->get_ub().isApprox(ub.template cast<OtherScalar>()));

  LQRFixtureTpl<Scalar> values(nx, nu, np, 2, 1);
  Model model(values.A, values.B, values.P, values.Q, values.R, values.N,
              values.W, values.Y, values.V, values.G, values.H, values.f,
              values.q, values.r, values.m, values.g, values.h);
  const crocoddyl::ActionModelLQRTpl<OtherScalar> casted_model =
      model.template cast<OtherScalar>();
  BOOST_CHECK(casted_model.get_A().isApprox(
      values.A.template cast<OtherScalar>(), tolerance<OtherScalar>()));
  BOOST_CHECK(casted_model.get_P().isApprox(
      values.P.template cast<OtherScalar>(), tolerance<OtherScalar>()));
  BOOST_CHECK(casted_model.get_W().isApprox(
      values.W.template cast<OtherScalar>(), tolerance<OtherScalar>()));
  BOOST_CHECK(casted_model.get_G().isApprox(
      values.G.template cast<OtherScalar>(), tolerance<OtherScalar>()));
  BOOST_CHECK(casted_model.get_m().isApprox(
      values.m.template cast<OtherScalar>(), tolerance<OtherScalar>()));

  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(state);
  manager->addParam("lqr", params);
  manager->addParam("inactive", std::make_shared<Params>(state, 1), false);
  BOOST_CHECK_EQUAL(manager->get_np(), np);
  BOOST_CHECK(manager->getParamStatus("lqr"));
  BOOST_CHECK(!manager->getParamStatus("inactive"));
  const std::shared_ptr<ParameterDataManager> manager_data =
      manager->createData();
  const std::shared_ptr<typename Model::ActionDataAbstract> abstract_data =
      model.createData(manager_data);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(abstract_data);
  BOOST_REQUIRE(data != nullptr);
  BOOST_CHECK(data->params == manager_data);
  model.set_params(data, manager);
  model.update_p(data, p);
  BOOST_CHECK(data->p.isApprox(p));
  BOOST_CHECK(manager_data->params->p.isApprox(p));
  BOOST_CHECK(manager_data->action_params.at("lqr")->p.isApprox(p));
  BOOST_CHECK(manager_data->action_params.at("inactive")->p.isZero());
  manager_data->params->dx_dp.setOnes();
  manager->calcDiff_action(manager_data, data, VectorXs::Zero(nx),
                           VectorXs::Zero(nu));
  BOOST_CHECK(manager_data->params->dx_dp.isZero());

  const std::shared_ptr<typename Model::ActionDataAbstract> null_shared_data =
      model.createData(std::shared_ptr<ParameterDataManager>());
  BOOST_CHECK(std::dynamic_pointer_cast<Data>(null_shared_data) != nullptr);
  BOOST_CHECK_THROW(model.set_params(data, std::shared_ptr<ParameterManager>()),
                    std::exception);
  const std::shared_ptr<ParameterManager> wrong_manager =
      std::make_shared<ParameterManager>(state);
  wrong_manager->addParam("wrong", std::make_shared<Params>(state, np + 1));
  BOOST_CHECK_THROW(model.set_params(data, wrong_manager), std::exception);
  manager->changeParamStatus("inactive", true);
  BOOST_CHECK_EQUAL(manager->get_np(), np + 1);
  BOOST_CHECK_THROW(model.set_params(data, manager), std::exception);

  std::ostringstream stream;
  stream << model << " " << *params;
  BOOST_CHECK(!stream.str().empty());
}

template <typename Scalar>
void test_shared_manager_updates_all_attached_data() {
  typedef crocoddyl::ActionDataLQRTpl<Scalar> Data;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef crocoddyl::ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  LQRFixtureTpl<Scalar> running_values(nx, nu, np, 1, 1);
  LQRFixtureTpl<Scalar> terminal_values(nx, 0, np, 1, 1);
  Model running(running_values.A, running_values.B, running_values.P,
                running_values.Q, running_values.R, running_values.N,
                running_values.W, running_values.Y, running_values.V,
                running_values.G, running_values.H, running_values.f,
                running_values.q, running_values.r, running_values.m,
                running_values.g, running_values.h);
  Model terminal(terminal_values.A, terminal_values.B, terminal_values.P,
                 terminal_values.Q, terminal_values.R, terminal_values.N,
                 terminal_values.W, terminal_values.Y, terminal_values.V,
                 terminal_values.G, terminal_values.H, terminal_values.f,
                 terminal_values.q, terminal_values.r, terminal_values.m,
                 terminal_values.g, terminal_values.h);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(running.get_state());
  manager->addParam("lqr", std::make_shared<Params>(running.get_state(), np));
  const std::shared_ptr<ParameterDataManager> manager_data =
      manager->createData();
  const std::shared_ptr<Data> running_data =
      std::static_pointer_cast<Data>(running.createData(manager_data));
  const std::shared_ptr<Data> terminal_data =
      std::static_pointer_cast<Data>(terminal.createData(manager_data));
  running.set_params(running_data, manager);
  terminal.set_params(terminal_data, manager);

  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.4), Scalar(0.8));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.2), Scalar(0.7));
  const VectorXs p_first = VectorXs::LinSpaced(np, Scalar(-0.3), Scalar(0.5));
  const VectorXs p_second = VectorXs::LinSpaced(np, Scalar(0.6), Scalar(-0.2));

  manager->update(manager_data, p_first);
  running.calc(running_data, x, u);
  running.calcDiff(running_data, x, u);
  terminal.calc(terminal_data, x);
  terminal.calcDiff(terminal_data, x);
  BOOST_CHECK(running_data->p.isZero());
  BOOST_CHECK(terminal_data->p.isZero());
  BOOST_CHECK(running_data->xnext.isApprox(
      running_values.A * x + running_values.B * u + running_values.P * p_first +
          running_values.f,
      tolerance<Scalar>()));
  BOOST_CHECK(running_data->Lp.isApprox(
      running_values.m + running_values.Y.transpose() * x +
          running_values.V.transpose() * u + running_values.W * p_first,
      tolerance<Scalar>()));
  BOOST_CHECK(terminal_data->Lp.isApprox(terminal_values.m +
                                             terminal_values.Y.transpose() * x +
                                             terminal_values.W * p_first,
                                         tolerance<Scalar>()));

  manager->update(manager_data, p_second);
  running.calc(running_data, x, u);
  running.calcDiff(running_data, x, u);
  terminal.calc(terminal_data, x);
  terminal.calcDiff(terminal_data, x);
  BOOST_CHECK(running_data->xnext.isApprox(
      running_values.A * x + running_values.B * u +
          running_values.P * p_second + running_values.f,
      tolerance<Scalar>()));
  BOOST_CHECK(running_data->Lp.isApprox(
      running_values.m + running_values.Y.transpose() * x +
          running_values.V.transpose() * u + running_values.W * p_second,
      tolerance<Scalar>()));
  BOOST_CHECK(terminal_data->Lp.isApprox(terminal_values.m +
                                             terminal_values.Y.transpose() * x +
                                             terminal_values.W * p_second,
                                         tolerance<Scalar>()));
}

template <typename Scalar>
void test_hot_paths_without_eigen_allocation() {
  typedef crocoddyl::ActionDataLQRTpl<Scalar> Data;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> Model;
  typedef crocoddyl::LQRParamsTpl<Scalar> Params;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename Model::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  LQRFixtureTpl<Scalar> values(nx, nu, np, 2, 1);
  Model model(values.A, values.B, values.P, values.Q, values.R, values.N,
              values.W, values.Y, values.V, values.G, values.H, values.f,
              values.q, values.r, values.m, values.g, values.h);
  const std::shared_ptr<ParameterManager> manager =
      std::make_shared<ParameterManager>(model.get_state());
  manager->addParam("lqr", std::make_shared<Params>(model.get_state(), np));
  const std::shared_ptr<typename Model::ParameterDataManager> manager_data =
      manager->createData();
  const std::shared_ptr<typename Model::ActionDataAbstract> abstract_data =
      model.createData(manager_data);
  const std::shared_ptr<Data> data =
      std::dynamic_pointer_cast<Data>(abstract_data);
  BOOST_REQUIRE(data != nullptr);
  model.set_params(data, manager);
  const VectorXs x = VectorXs::LinSpaced(nx, Scalar(-0.5), Scalar(0.5));
  const VectorXs u = VectorXs::LinSpaced(nu, Scalar(0.2), Scalar(0.8));
  const VectorXs p = VectorXs::LinSpaced(np, Scalar(-0.3), Scalar(0.7));
  model.update_p(data, p);
  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  model.calc(data, x);
  model.calcDiff(data, x);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      model.update_p(data, p);
      model.calc(data, x, u);
      model.calcDiff(data, x, u);
      model.calc(data, x);
      model.calcDiff(data, x);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_CHECK(data->p.isApprox(p));
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_lqr");
  ts->add(BOOST_TEST_CASE(
      &test_constructors_and_zero_parameter_compatibility<double>));
  ts->add(BOOST_TEST_CASE(
      &test_constructors_and_zero_parameter_compatibility<float>));
  ts->add(BOOST_TEST_CASE(
      &test_parameterized_running_terminal_formulas_and_data<double>));
  ts->add(BOOST_TEST_CASE(
      &test_parameterized_running_terminal_formulas_and_data<float>));
  ts->add(BOOST_TEST_CASE(&test_constant_derivatives_refresh_per_data<double>));
  ts->add(BOOST_TEST_CASE(&test_constant_derivatives_refresh_per_data<float>));
  ts->add(BOOST_TEST_CASE(&test_setters_set_lqr_and_dimension_errors<double>));
  ts->add(BOOST_TEST_CASE(&test_setters_set_lqr_and_dimension_errors<float>));
  ts->add(
      BOOST_TEST_CASE(&test_parameter_model_manager_and_scalar_casts<double>));
  ts->add(
      BOOST_TEST_CASE(&test_parameter_model_manager_and_scalar_casts<float>));
  ts->add(
      BOOST_TEST_CASE(&test_shared_manager_updates_all_attached_data<double>));
  ts->add(
      BOOST_TEST_CASE(&test_shared_manager_updates_all_attached_data<float>));
  ts->add(BOOST_TEST_CASE(&test_hot_paths_without_eigen_allocation<double>));
  ts->add(BOOST_TEST_CASE(&test_hot_paths_without_eigen_allocation<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
