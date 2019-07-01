///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_UTILS_HPP_
#define PYTHON_CROCODDYL_UTILS_HPP_

#include <Eigen/Dense>
#include <vector>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

template<class T>
bp::list std_vector_to_python_list(const std::vector<T>& vec) {
  const long unsigned int& n = vec.size();
  bp::list list;
  for (unsigned int i = 0; i < n; ++i) {
    list.append(vec[i]);
  }
  return list;
}

template<class T>
std::vector<T> python_list_to_std_vector(const bp::list& list) {
  const long int& n = len(list);
  std::vector<T> vec;
  vec.resize(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = bp::extract<T>(list[i]);
  }
  return vec;
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_UTILS_HPP_