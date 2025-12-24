---
sidebar_position: 8
title: "Gazebo Simulation Basics"
---

# Gazebo Simulation Basics

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the architecture and components of Gazebo simulation
- Create and configure simulation environments with models and physics
- Integrate Gazebo with ROS 2 for robotics simulation
- Implement sensors and actuators in Gazebo models
- Design and run simulation scenarios for robot testing
- Evaluate Gazebo for specific robotics applications

## Introduction

Gazebo is a powerful, open-source robotics simulation environment that provides physics simulation, sensor simulation, and visualization capabilities for robotics development. As part of the Open Robotics ecosystem, Gazebo has become the de facto standard for robotics simulation, enabling developers to test algorithms, validate designs, and generate training data without the need for physical hardware. The platform supports realistic physics simulation using ODE (Open Dynamics Engine), Bullet, or DART physics engines, making it suitable for a wide range of robotics applications from simple mobile robots to complex humanoid systems.

Gazebo's strength lies in its ability to provide a realistic simulation environment that closely matches real-world physics and sensor characteristics. This enables effective sim-to-real transfer of algorithms and provides a safe, cost-effective platform for robotics development and testing. The integration with ROS and ROS 2 through Gazebo ROS packages makes it particularly valuable for the robotics community.

## Gazebo Architecture and Components

### Core Architecture

Gazebo's architecture consists of several interconnected components that work together to provide a complete simulation environment:

#### Server Component (gzserver)
- Handles physics simulation and model updates
- Manages simulation time and execution
- Processes client requests and commands
- Maintains the simulation world state

#### Client Component (gzclient)
- Provides visualization and user interface
- Enables interactive world manipulation
- Displays real-time simulation data
- Offers debugging and analysis tools

#### Physics Engine Interface
- Supports multiple physics engines (ODE, Bullet, DART)
- Handles collision detection and response
- Manages rigid body dynamics
- Provides contact and constraint solving

### Simulation World Structure

#### World Files
World files in Gazebo define the complete simulation environment using SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include models from fuel.gazebosim.org -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define custom models -->
    <model name="my_robot">
      <!-- Model definition -->
    </model>

    <!-- Physics engine configuration -->
    <physics name="default" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

#### Model Structure
Models in Gazebo are defined using SDF and contain:
- **Links**: Rigid bodies with mass, geometry, and visual properties
- **Joints**: Connections between links with kinematic constraints
- **Sensors**: Simulated sensors attached to links
- **Plugins**: Custom code that extends model behavior

## Installing and Running Gazebo

### Installation Options

#### Ubuntu Installation
```bash
# Install Gazebo Fortress (recommended version)
sudo apt update
sudo apt install gazebo libgazebo-dev

# For ROS 2 integration
sudo apt install ros-humble-gazebo-ros-pkgs
```

#### Docker Installation
```bash
# Pull Gazebo Docker image
docker pull gazebo:fortress

# Run Gazebo with GUI support
docker run -it --rm \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/dev/dri:/dev/dri" \
  --device=/dev/dri \
  --name=gazebo \
  gazebo:fortress
```

### Running Gazebo

#### Basic Commands
```bash
# Start Gazebo server only (no GUI)
gz sim -s

# Start Gazebo with GUI
gz sim

# Load a specific world file
gz sim -r my_world.sdf

# Start paused
gz sim -r -s my_world.sdf
```

## Creating Models and Environments

### SDF (Simulation Description Format)

SDF is the XML-based format used to describe simulation elements:

#### Basic Robot Model Example
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <!-- Chassis link -->
    <link name="chassis">
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
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Wheel joints and links would follow -->
  </model>
</sdf>
```

### Model Components

#### Links
Links represent rigid bodies in the simulation:

```xml
<link name="link_name">
  <!-- Mass and inertia properties -->
  <inertial>
    <mass>1.0</mass>
    <inertia>
      <ixx>0.01</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.01</iyy>
      <iyz>0</iyz>
      <izz>0.01</izz>
    </inertia>
  </inertial>

  <!-- Collision geometry -->
  <collision name="collision">
    <geometry>
      <box>
        <size>1.0 1.0 1.0</size>
      </box>
    </geometry>
  </collision>

  <!-- Visual geometry -->
  <visual name="visual">
    <geometry>
      <mesh>
        <uri>model://my_robot/meshes/link.dae</uri>
      </mesh>
    </geometry>
  </visual>
</link>
```

#### Joints
Joints connect links and define their relative motion:

```xml
<joint name="joint_name" type="revolute">
  <parent>parent_link</parent>
  <child>child_link</child>
  <axis>
    <xyz>0 0 1</xyz>  <!-- Rotation axis -->
    <limit>
      <lower>-1.57</lower>  <!-- Lower limit (radians) -->
      <upper>1.57</upper>   <!-- Upper limit (radians) -->
      <effort>100</effort>  <!-- Maximum effort (N-m) -->
      <velocity>1</velocity> <!-- Maximum velocity (rad/s) -->
    </limit>
  </axis>
</joint>
```

## Physics Configuration

### Physics Engine Settings

Gazebo supports multiple physics engines, each with different characteristics:

#### ODE (Open Dynamics Engine)
- Default physics engine
- Good balance of performance and accuracy
- Suitable for most robotics applications

#### Bullet Physics
- Better performance for complex scenes
- More accurate contact simulation
- Good for real-time applications

#### DART (Dynamic Animation and Robotics Toolkit)
- Advanced contact handling
- Better for articulated systems
- More sophisticated constraint solving

### Physics Parameters

```xml
<physics name="default" type="ode">
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Sensors in Gazebo

### Camera Sensors

Camera sensors simulate RGB cameras with realistic distortion:

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

### LiDAR Sensors

LiDAR sensors simulate 2D or 3D laser range finders:

```xml
<sensor name="lidar" type="gpu_lidar">
  <pose>0.1 0 0.2 0 0 0</pose>
  <lidar>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </lidar>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
</sensor>
```

### IMU Sensors

IMU sensors simulate inertial measurement units:

```xml
<sensor name="imu" type="imu">
  <pose>0 0 0 0 0 0</pose>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <always_on>1</always_on>
  <update_rate>100</update_rate>
</sensor>
```

## Gazebo-ROS Integration

### Gazebo ROS Packages

The Gazebo ROS packages provide seamless integration between Gazebo and ROS 2:

#### Key Components
- **gazebo_ros**: Core ROS-Gazebo integration
- **gazebo_plugins**: Common robot plugins
- **gazebo_msgs**: ROS message definitions for Gazebo
- **gazebo_dev**: Development files for custom plugins

### Launching with ROS 2

#### Launch File Example
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo,
    ])
```

### Robot Spawn and Control

#### Spawning Robots
```bash
# Spawn a robot model in Gazebo
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.sdf -x 0 -y 0 -z 1
```

#### Joint Control
```python
# Example of controlling joints in ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/my_robot/joint_group_position_controller/commands',
            10
        )

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [0.5, -0.3, 0.2]  # Joint positions
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = JointController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

## Creating Custom Plugins

### Model Plugins

Model plugins extend the behavior of models in Gazebo:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class MyRobotPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&MyRobotPlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small linear velocity to the model
      this->model->SetLinearVel(math::Vector3(0.01, 0, 0));
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(MyRobotPlugin)
}
```

### Sensor Plugins

Sensor plugins can process sensor data or provide custom sensor functionality:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomSensorPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
    {
      // Get the camera sensor as a camera sensor
      this->camera = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);

      if (!this->camera)
      {
        gzerr << "CustomSensorPlugin not attached to a camera sensor\n";
        return;
      }

      // Connect to the sensor update event
      this->newFrameConnection = this->camera->ConnectUpdated(
          std::bind(&CustomSensorPlugin::OnNewFrame, this));

      // Make sure the parent sensor is active
      this->camera->SetActive(true);
    }

    private: void OnNewFrame()
    {
      // Get the image data
      const unsigned char *data = this->camera->ImageData();

      // Process the image data as needed
      // ...
    }

    private: sensors::CameraSensorPtr camera;
    private: common::ConnectionPtr newFrameConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomSensorPlugin)
}
```

## Advanced Simulation Features

### Physics Properties

#### Material Properties
```xml
<material>
  <script>
    <uri>file://media/materials/scripts/gazebo.material</uri>
    <name>Gazebo/Blue</name>
  </script>
</material>
```

#### Surface Properties
```xml
<collision name="collision">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1e+13</kp>
        <kd>1</kd>
        <max_vel>0.01</max_vel>
        <min_depth>0</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Environment Effects

#### Lighting
```xml
<light name="sun" type="directional">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.5 0.1 -0.9</direction>
</light>
```

#### Weather and Atmosphere
Gazebo supports basic atmospheric effects that can affect sensor simulation:

```xml
<world name="default">
  <atmosphere type="adiabatic">
    <temperature>288.15</temperature>
    <pressure>101325</pressure>
  </atmosphere>

  <gravity>0 0 -9.8</gravity>
</world>
```

## Performance Optimization

### Simulation Performance

#### Time Stepping
- Use appropriate step sizes (typically 0.001s)
- Balance accuracy with performance
- Adjust real-time factor as needed

#### Rendering Optimization
- Reduce visual complexity when not needed
- Use simpler collision geometries
- Limit sensor update rates
- Optimize model mesh complexity

### Resource Management

#### Memory Usage
- Use efficient collision meshes
- Limit the number of complex models
- Optimize texture sizes
- Use level-of-detail where appropriate

## Debugging and Analysis

### Built-in Tools

#### Gazebo GUI Features
- Model and link selection
- Property inspection
- Physics visualization
- Real-time statistics

#### Command Line Tools
```bash
# List all models in simulation
gz model -m

# Get model pose
gz model -m robot_name -i

# Set model pose
gz model -m robot_name -x 1.0 -y 2.0 -z 0.0
```

### Custom Analysis

#### Data Logging
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.subscription = self.create_subscription(
            Odometry,
            '/robot/odom',
            self.odom_callback,
            10
        )

        self.csv_file = open('robot_trajectory.csv', 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    def odom_callback(self, msg):
        row = [
            self.get_clock().now().nanoseconds,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        self.csv_writer.writerow(row)

def main(args=None):
    rclpy.init(args=args)
    logger = DataLogger()
    rclpy.spin(logger)
    logger.csv_file.close()
    logger.destroy_node()
    rclpy.shutdown()
```

## Best Practices

### Model Design

#### Performance Considerations
- Use simple collision geometries when possible
- Optimize mesh resolution for visual components
- Limit the number of sensors per model
- Use appropriate physics parameters

#### Realism vs. Performance
- Balance visual fidelity with simulation speed
- Use noise models that match real sensors
- Validate simulation results against real data
- Document simulation assumptions

### Simulation Scenarios

#### Testing Strategies
- Create diverse test environments
- Include edge cases and failure scenarios
- Validate sensor models against real hardware
- Test controller robustness to simulation errors

## Chapter Quiz

import QuizComponent from '@site/src/components/QuizComponent/QuizComponent';

<QuizComponent
  quizData={{
    title: "Gazebo Simulation Basics Quiz",
    questions: [
      {
        question: "What does SDF stand for in Gazebo?",
        options: [
          "Simulation Development Format",
          "Standard Description Format",
          "Simulation Description Format",
          "System Definition Format"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "Which physics engines are supported by Gazebo?",
        options: [
          "ODE only",
          "ODE, Bullet, and DART",
          "Bullet and PhysX",
          "Havok and ODE"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "What is the purpose of a Gazebo model plugin?",
        options: [
          "To create new visual effects",
          "To extend the behavior of models in simulation",
          "To connect to external databases",
          "To handle network communication"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "Which ROS package provides the core integration between Gazebo and ROS 2?",
        options: [
          "ros_gazebo",
          "gazebo_ros",
          "ros2_gazebo",
          "gazebo_core"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "What is the default physics engine used by Gazebo?",
        options: [
          "Bullet",
          "DART",
          "ODE",
          "PhysX"
        ],
        correctAnswerIndex: 2
      }
    ]
  }}
/>

## Exercises

1. Create a simple robot model in Gazebo with a chassis and two wheels. Add a camera sensor and test its output using ROS 2 tools.

2. Design a simulation environment with obstacles and implement a navigation task where a robot must reach a goal while avoiding obstacles.

3. Research and compare Gazebo with other robotics simulators (Isaac Sim, Webots, PyBullet). Discuss the trade-offs for different applications.

## Summary

Gazebo provides a comprehensive simulation environment for robotics development with realistic physics, sensor simulation, and visualization capabilities. The platform's integration with ROS 2 through the gazebo_ros packages enables seamless development of robot algorithms in simulation before deployment on real hardware. Understanding Gazebo's architecture, model creation, and simulation configuration is essential for effective robotics development and testing.

## Further Reading

- Gazebo Documentation and Tutorials
- ROS 2 with Gazebo Integration Guide
- Simulation-Based Robot Development Best Practices
- Physics Simulation in Robotics Applications