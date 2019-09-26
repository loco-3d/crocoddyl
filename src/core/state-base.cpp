///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

StateAbstract::StateAbstract(unsigned int const& nx, unsigned int const& ndx) : nx_(nx), ndx_(ndx) {
  nv_ = ndx / 2;
  nq_ = nx_ - nv_;
}

StateAbstract::~StateAbstract() {}

const unsigned int& StateAbstract::get_nx() const { return nx_; }

const unsigned int& StateAbstract::get_ndx() const { return ndx_; }

const unsigned int& StateAbstract::get_nq() const { return nq_; }

const unsigned int& StateAbstract::get_nv() const { return nv_; }

}  // namespace crocoddyl
