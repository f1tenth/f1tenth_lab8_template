cmake_minimum_required(VERSION 3.5)
project(mppi)

# Find dependencies
foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  rclpy
  geometry_msgs
  nav_msgs
  ackermann_msgs
  visualization_msgs)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mppi_node.py
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
