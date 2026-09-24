cmake_minimum_required(VERSION 3.5)
project(mpc)

# Default to C++17 (required for structured bindings)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# OSQP-Eigen is only needed by the C++ node. Without it (the grading image has
# none, see the README) the C++ target is skipped and the Python node still
# builds, so a Python team compiles anywhere.
find_package(OsqpEigen QUIET)
if(OsqpEigen_FOUND)
  message(STATUS "OsqpEigen found: building the C++ mpc_node")
else()
  message(STATUS "OsqpEigen not found: skipping the C++ mpc_node (the Python mpc_node.py is still installed)")
endif()

# Include Cpp "include" directory
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

if(OsqpEigen_FOUND)
  # Create Cpp executable
  add_executable(mpc_node src/mpc_node.cpp)

  target_include_directories(mpc_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
  )

  ament_target_dependencies(mpc_node
    rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2 tf2_ros
  )

  target_link_libraries(mpc_node OsqpEigen::OsqpEigen)

  # Install Cpp executables
  install(TARGETS
    mpc_node
    DESTINATION lib/${PROJECT_NAME})
endif()

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mpc_node.py
  scripts/utils.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install the launch files: the autograder starts your nodes with them
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install the waypoint files, so the node finds them wherever the package is built
install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()
