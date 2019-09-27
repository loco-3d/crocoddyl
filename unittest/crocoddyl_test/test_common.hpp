///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

namespace crocoddyl_unit_test{

template <class PtrType>
void delete_pointer(PtrType* ptr) {
  if (ptr != NULL) {
    delete ptr;
    ptr = NULL;
  }
}

} // namespace crocoddyl_unit_test

#endif