cmake_minimum_required(VERSION 3.8)
project(robotx_safe_docking_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robotx_safe_docking_msgs REQUIRED)

add_executable(minco_replanner_node
  src/minco_replanner_node.cpp
)
target_compile_features(minco_replanner_node PRIVATE cxx_std_17)
target_link_libraries(minco_replanner_node
  Eigen3::Eigen
  pybind11::embed
)
ament_target_dependencies(minco_replanner_node
  nav_msgs
  rclcpp
  robotx_safe_docking_msgs
)

install(
  TARGETS minco_replanner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
