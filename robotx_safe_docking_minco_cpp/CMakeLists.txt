cmake_minimum_required(VERSION 3.8)
project(robotx_safe_docking_minco_cpp)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_robotx_safe_docking_minco_cpp
  src/minco_cpp_bindings.cpp
)
target_include_directories(_robotx_safe_docking_minco_cpp
  PRIVATE
    include
)
target_link_libraries(_robotx_safe_docking_minco_cpp
  PRIVATE
    Eigen3::Eigen
)
target_compile_features(_robotx_safe_docking_minco_cpp PRIVATE cxx_std_17)

ament_python_install_package(robotx_minco_cpp)

install(
  TARGETS _robotx_safe_docking_minco_cpp
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/robotx_minco_cpp"
)

ament_package()
