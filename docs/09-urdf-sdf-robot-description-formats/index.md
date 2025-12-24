---
sidebar_position: 9
title: "URDF, SDF: Robot Description Formats"
---

# URDF, SDF: Robot Description Formats

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the differences between URDF and SDF formats
- Create robot models using URDF for ROS-based systems
- Design simulation environments using SDF for Gazebo
- Convert between URDF and SDF formats when needed
- Implement advanced features like transmissions and sensors
- Validate and debug robot description files

## Introduction

Robot description formats are fundamental to robotics development, providing standardized ways to represent robot models, environments, and simulation parameters. Universal Robot Description Format (URDF) and Simulation Description Format (SDF) are the two primary formats used in the robotics community. URDF is primarily used for robot modeling in ROS-based systems, while SDF is used for simulation environments in Gazebo. Understanding both formats is essential for effective robotics development, as they serve different but complementary purposes in the robot development pipeline.

The choice between URDF and SDF depends on the specific application: URDF for robot modeling and kinematic descriptions, SDF for simulation environments and complete world descriptions. Modern robotics workflows often involve converting between these formats to leverage the strengths of both ROS for robot control and Gazebo for simulation.

## Universal Robot Description Format (URDF)

### URDF Overview

URDF (Universal Robot Description Format) is an XML-based format designed to represent robot models in ROS. It describes the physical and kinematic properties of robots, including links, joints, inertial properties, visual and collision geometries, and sensors. URDF is primarily focused on robot modeling rather than complete simulation environments.

#### Key Components of URDF
- **Links**: Rigid bodies that make up the robot structure
- **Joints**: Connections between links with kinematic constraints
- **Visual**: How the robot appears visually
- **Collision**: How the robot interacts physically with the environment
- **Inertial**: Mass and inertia properties for dynamics

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.1" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.1" />
      </geometry>
    </collision>
  </link>

  <!-- Additional links and joints would follow -->
</robot>
```

### Links and Their Properties

#### Link Definition
A link represents a rigid body in the robot model:

```xml
<link name="link_name">
  <!-- Inertial properties for dynamics simulation -->
  <inertial>
    <mass value="1.0" />
    <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0" />
    <inertia ixx="0.01" ixy="0.0" ixz="0.0"
             iyy="0.01" iyz="0.0" izz="0.01" />
  </inertial>

  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <!-- Various geometry types -->
      <box size="0.1 0.1 0.1" />
      <!-- <cylinder radius="0.1" length="0.2" /> -->
      <!-- <sphere radius="0.1" /> -->
      <!-- <mesh filename="package://my_robot/meshes/link.stl" /> -->
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1" />
    </material>
  </visual>

  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.1 0.1 0.1" />
    </geometry>
  </collision>
</link>
```

### Joints and Their Types

Joints define the kinematic relationships between links:

#### Joint Definition
```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name" />
  <child link="child_link_name" />
  <origin xyz="0.1 0 0" rpy="0 0 0" />

  <!-- Joint axis for revolute and prismatic joints -->
  <axis xyz="0 0 1" />

  <!-- Joint limits -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1" />

  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.0" />
</joint>
```

#### Joint Types
- **Fixed**: No relative motion between links
- **Revolute**: Single-axis rotation with limits
- **Continuous**: Single-axis rotation without limits
- **Prismatic**: Single-axis translation with limits
- **Planar**: Motion in a plane
- **Floating**: 6-DOF motion

#### Example Joint Types
```xml
<!-- Revolute joint (rotational with limits) -->
<joint name="shoulder_joint" type="revolute">
  <parent link="torso" />
  <child link="upper_arm" />
  <origin xyz="0 0 0.3" rpy="0 0 0" />
  <axis xyz="0 1 0" />
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1" />
</joint>

<!-- Continuous joint (rotational without limits) -->
<joint name="wheel_joint" type="continuous">
  <parent link="chassis" />
  <child link="wheel" />
  <origin xyz="0.2 0 -0.1" rpy="0 0 0" />
  <axis xyz="0 1 0" />
</joint>

<!-- Fixed joint (no motion) -->
<joint name="sensor_joint" type="fixed">
  <parent link="chassis" />
  <child link="camera" />
  <origin xyz="0.1 0 0.1" rpy="0 0 0" />
</joint>
```

### Transmissions for Actuators

Transmissions define how actuators connect to joints:

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Materials and Colors

```xml
<material name="blue">
  <color rgba="0 0 1 1" />
</material>

<material name="red">
  <color rgba="1 0 0 1" />
</material>

<material name="white">
  <color rgba="1 1 1 1" />
</material>
```

## Simulation Description Format (SDF)

### SDF Overview

SDF (Simulation Description Format) is an XML-based format designed for complete simulation environments. Unlike URDF, which focuses on robot models, SDF can describe entire worlds, including robots, environments, physics properties, lighting, and simulation parameters. SDF is the native format for Gazebo simulation.

#### Key Components of SDF
- **Worlds**: Complete simulation environments
- **Models**: Individual robot or object models
- **Lights**: Lighting in the simulation
- **Physics**: Physics engine configuration
- **GUI**: Visualization settings

### Basic SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include models from Gazebo Model Database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define custom model -->
    <model name="my_robot">
      <!-- Model definition similar to URDF but with SDF syntax -->
      <pose>0 0 0 0 0 0</pose>

      <!-- Links -->
      <link name="base_link">
        <pose>0 0 0.1 0 0 0</pose>

        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.01</izz>
          </inertia>
        </inertial>

        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.3 0.1</size>
            </box>
          </geometry>
        </collision>

        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.3 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Joints -->
      <joint name="joint_name" type="revolute">
        <parent>base_link</parent>
        <child>child_link</child>
        <axis>
          <xyz>0 0 1</xyz>
          <limit>
            <lower>-1.57</lower>
            <upper>1.57</upper>
            <effort>100</effort>
            <velocity>1</velocity>
          </limit>
        </axis>
      </joint>
    </model>

    <!-- Physics configuration -->
    <physics name="default" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

### World Definition in SDF

```xml
<world name="my_world">
  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- Physics engine -->
  <physics name="default" type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Models -->
  <model name="robot1">
    <!-- Robot definition -->
  </model>

  <!-- Lights -->
  <light name="sun" type="directional">
    <pose>0 0 10 0 0 0</pose>
    <diffuse>0.8 0.8 0.8 1</diffuse>
    <specular>0.2 0.2 0.2 1</specular>
    <direction>-0.5 0.1 -0.9</direction>
  </light>

  <!-- Plugins -->
  <plugin name="my_plugin" filename="libMyPlugin.so">
    <!-- Plugin parameters -->
  </plugin>
</world>
```

### Sensors in SDF

SDF allows detailed sensor definitions:

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
</sensor>
```

## Converting Between URDF and SDF

### URDF to SDF Conversion

URDF files can be converted to SDF for use in Gazebo:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or using the legacy command
rosrun xacro xacro robot.urdf.xacro > robot.urdf
gz sdf -p robot.urdf > robot.sdf
```

### Using xacro for Complex URDF

Xacro (XML Macros) allows parameterization and reuse in URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}" />
      <child link="${prefix}_wheel" />
      <origin xyz="${xyz}" rpy="${rpy}" />
      <axis xyz="0 1 0" />
    </joint>

    <link name="${prefix}_wheel">
      <inertial>
        <mass value="0.5" />
        <origin xyz="0 0 0" />
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
      </inertial>

      <visual>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </visual>

      <collision>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0" rpy="0 0 0" />
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0" rpy="0 0 0" />
</robot>
```

## Advanced Features

### Gazebo-Specific Extensions in URDF

URDF files can include Gazebo-specific extensions:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.08 0.04" />
    </geometry>
    <material name="blue" />
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.08 0.04" />
    </geometry>
  </collision>
  <inertial>
    <mass value="0.01" />
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
  </inertial>
</link>

<!-- Gazebo-specific extensions -->
<gazebo reference="camera_link">
  <material>Gazebo/Blue</material>
</gazebo>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>800</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Robot State Publisher Integration

URDF works with robot_state_publisher to broadcast transforms:

```xml
<!-- In launch file -->
<node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher">
  <param name="robot_description" command="$(find xacro)/xacro $(find my_robot_description)/urdf/my_robot.urdf.xacro" />
</node>
```

## Validation and Debugging

### URDF Validation Tools

#### Check URDF Syntax
```bash
# Validate URDF syntax
check_urdf robot.urdf

# Display robot information
urdf_to_graphiz robot.urdf
```

#### Common URDF Issues
- Missing or incorrect joint limits
- Invalid inertia tensors
- Non-physical mass values
- Incorrect kinematic chains

### SDF Validation

```bash
# Validate SDF syntax
gz sdf -k world.sdf

# Convert and check SDF
gz sdf -p world.sdf
```

## Best Practices

### URDF Best Practices

#### Structure and Organization
- Use xacro for complex robots to avoid repetition
- Organize files in a logical directory structure
- Use consistent naming conventions
- Include comments for complex parts

#### Physical Accuracy
- Use realistic mass and inertia values
- Ensure collision geometry matches visual geometry
- Use appropriate joint limits based on hardware
- Validate with kinematic and dynamic analysis

### SDF Best Practices

#### Performance Optimization
- Use simple collision geometries when possible
- Optimize mesh resolution for visual elements
- Use appropriate physics parameters
- Limit sensor update rates to necessary levels

#### Realism vs. Performance
- Balance visual fidelity with simulation speed
- Use noise models that match real sensors
- Validate simulation results against real data
- Document simulation assumptions

## Real-World Applications

### Industrial Robotics
- Robot arm modeling and simulation
- Factory automation environments
- Safety zone definition
- Path planning validation

### Mobile Robotics
- Wheeled robot platforms
- Sensor configuration
- Navigation testing
- Multi-robot simulation

### Humanoid Robotics
- Complex kinematic chains
- Balance and locomotion simulation
- Human interaction scenarios
- Grasping and manipulation

## Tools and Ecosystem

### Visualization Tools
- **RViz**: ROS visualization for URDF models
- **Gazebo GUI**: Simulation environment visualization
- **URDF Viewer**: Standalone URDF visualization
- **MeshLab**: 3D model inspection

### Editing Tools
- **SolidWorks to URDF**: Direct export from CAD
- **Blender**: 3D modeling with URDF/SDF export
- **Text editors**: Manual editing with syntax highlighting
- **Robotics simulators**: Integrated model creation

## Chapter Quiz

import QuizComponent from '@site/src/components/QuizComponent/QuizComponent';

<QuizComponent
  quizData={{
    title: "URDF and SDF Quiz",
    questions: [
      {
        question: "What does URDF stand for?",
        options: [
          "Universal Robot Design Format",
          "Unified Robot Description Format",
          "Universal Robot Description Format",
          "Universal Robotics Data Format"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "Which format is primarily used for simulation environments?",
        options: [
          "URDF",
          "SDF",
          "XACRO",
          "XML"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "What is the main purpose of transmissions in URDF?",
        options: [
          "To define visual properties",
          "To specify how actuators connect to joints",
          "To set collision properties",
          "To define material properties"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "Which tool can be used to validate URDF files?",
        options: [
          "gz sdf -k",
          "check_urdf",
          "ros2 validate",
          "urdf_check"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "What is xacro used for in the context of URDF?",
        options: [
          "To convert URDF to SDF",
          "To visualize URDF models",
          "To provide XML macros for parameterization and reuse",
          "To validate URDF syntax"
        ],
        correctAnswerIndex: 2
      }
    ]
  }}
/>

## Exercises

1. Create a URDF model for a simple differential drive robot with two wheels and a caster. Include proper inertial, visual, and collision properties.

2. Convert your URDF model to SDF and create a Gazebo world file that includes your robot and a simple environment with obstacles.

3. Use xacro to create a parameterized URDF model that can be easily modified for different robot configurations.

## Summary

URDF and SDF are fundamental formats for robot modeling and simulation in the ROS ecosystem. URDF provides a standardized way to describe robot kinematics and physical properties for ROS-based systems, while SDF enables complete simulation environments for Gazebo. Understanding both formats and how to convert between them is essential for effective robotics development. The use of xacro for complex models and proper validation techniques ensures robust robot descriptions that work correctly in both planning and simulation contexts.

## Further Reading

- URDF/XML Format Specification
- SDF Format Documentation
- Xacro Tutorial and Best Practices
- Robot Modeling in ROS
- Gazebo Simulation Guide