---
sidebar_position: 7
title: "Building ROS 2 Packages in Python"
---

# Building ROS 2 Packages in Python

## Learning Objectives

By the end of this chapter, students will be able to:
- Create and structure ROS 2 packages using Python
- Implement nodes, publishers, subscribers, services, and actions in Python
- Define custom messages, services, and actions for Python packages
- Build and install ROS 2 Python packages
- Debug and test Python-based ROS 2 nodes
- Follow best practices for Python-based ROS 2 development

## Introduction

Python has become one of the most popular languages for robotics development due to its ease of use, extensive library ecosystem, and rapid prototyping capabilities. ROS 2's Python client library (rclpy) provides a comprehensive interface for building robotics applications, making it accessible to both beginners and experienced developers. Python's interpreted nature and rich ecosystem of scientific computing libraries make it ideal for robotics research, prototyping, and many production applications.

The rclpy library provides Python bindings for ROS 2's underlying C++ infrastructure, enabling Python nodes to communicate seamlessly with C++ nodes and other ROS 2 components. This integration allows teams to leverage Python's strengths for algorithm development while maintaining compatibility with the broader ROS 2 ecosystem.

## Package Structure and Organization

### Standard ROS 2 Package Layout

A well-structured ROS 2 Python package follows a standard layout:

```
my_robot_package/
├── package.xml          # Package metadata and dependencies
├── setup.py            # Python package setup
├── setup.cfg           # Installation configuration
├── CMakeLists.txt      # CMake configuration (for message generation)
├── my_robot_package/   # Main Python package directory
│   ├── __init__.py
│   ├── my_node.py      # Individual ROS 2 nodes
│   └── utils.py        # Helper functions and utilities
├── launch/             # Launch files for starting nodes
│   └── my_launch.py
├── test/               # Unit and integration tests
│   └── test_my_node.py
└── resource/           # Package resource files
```

### Package Metadata

The `package.xml` file contains essential metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package in Python</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Creating Python Nodes

### Basic Node Structure

A Python ROS 2 node inherits from `rclpy.node.Node` and implements the required functionality:

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('my_robot_node')

        # Log messages to the ROS 2 logger
        self.get_logger().info('MyRobotNode initialized')

def main(args=None):
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the node
    node = MyRobotNode()

    try:
        # Keep the node running and processing callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up resources
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parameter Declaration and Usage

Nodes can declare and use parameters for configuration:

```python
import rclpy
from rclpy.node import Node

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(f'Initialized with robot_name: {self.robot_name}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and Subscribers

### Publisher Implementation

Creating publishers to send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')

        # Create a publisher for the 'chatter' topic
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer to publish messages periodically
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = TalkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation

Creating subscribers to receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')

        # Create a subscription to the 'chatter' topic
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)

        # Prevent unused variable warning
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) Configuration

Configuring QoS settings for publishers and subscribers:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

class QoSTalkerNode(Node):
    def __init__(self):
        super().__init__('qos_talker')

        # Define a custom QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Create publisher with custom QoS
        self.publisher = self.create_publisher(String, 'qos_chatter', qos_profile)

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'QoS Message: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = QoSTalkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Actions

### Service Implementation

Creating service servers and clients in Python:

```python
# Service server implementation
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')

        # Create a service server
        self.srv = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    server = AddTwoIntsServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Implementation

Creating action servers and clients:

```python
# Action server implementation
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # Check if the goal has been canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            # Update the feedback message
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Simulate work with a delay
            time.sleep(1)

        # Complete the goal successfully
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    node = FibonacciActionServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Package Setup and Installation

### setup.py Configuration

The `setup.py` file defines how the Python package is built and installed:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Example ROS 2 package in Python',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_robot_node = my_robot_package.my_node:main',
            'talker = my_robot_package.talker:main',
            'listener = my_robot_package.listener:main',
        ],
    },
)
```

### setup.cfg Configuration

The `setup.cfg` file provides additional installation configuration:

```ini
[develop]
script-dir=$base/lib/my_robot_package
[install]
install-scripts=$base/lib/my_robot_package
```

## Launch Files

### Python Launch Files

Creating launch files to start multiple nodes together:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='talker',
            name='talker_node',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ],
            remappings=[
                ('original_topic', 'remapped_topic')
            ]
        ),
        Node(
            package='my_robot_package',
            executable='listener',
            name='listener_node',
            parameters=[
                {'param1': 'value1'}
            ]
        )
    ])
```

## Custom Messages, Services, and Actions

### Defining Custom Messages

Create custom message definitions in the `msg/` directory:

```
# CustomMessage.msg
string name
int32 id
float64[] values
geometry_msgs/Pose pose
```

### Using Custom Messages

After defining custom messages, they can be imported and used:

```python
from my_robot_package.msg import CustomMessage
import rclpy
from rclpy.node import Node

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')

        # Create publisher for custom message
        self.publisher = self.create_publisher(CustomMessage, 'custom_topic', 10)

        # Create subscription to custom message
        self.subscription = self.create_subscription(
            CustomMessage,
            'custom_topic',
            self.custom_message_callback,
            10)

    def custom_message_callback(self, msg):
        self.get_logger().info(f'Received custom message: {msg.name}, ID: {msg.id}')
```

## Testing and Debugging

### Unit Testing

Creating unit tests for ROS 2 Python nodes:

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_package.my_node import MyRobotNode

class TestMyRobotNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = MyRobotNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'my_robot_node')

if __name__ == '__main__':
    unittest.main()
```

### Debugging Techniques

#### ROS 2 Command Line Tools
```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info /my_robot_node

# Echo messages from a topic
ros2 topic echo /chatter

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# List all topics
ros2 topic list
```

#### Python Debugging
```python
# Use the ROS 2 logger for debugging
self.get_logger().debug('Debug message')
self.get_logger().info('Info message')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal message')

# Use Python's built-in debugger
import pdb; pdb.set_trace()
```

## Best Practices

### Code Organization

#### Node Structure
- Keep nodes focused on a single responsibility
- Separate business logic from ROS 2 infrastructure
- Use helper classes for complex logic
- Follow Python naming conventions (PEP 8)

#### Error Handling
- Implement proper exception handling
- Use ROS 2 logging instead of print statements
- Handle shutdown gracefully
- Validate inputs and parameters

### Performance Considerations

#### Memory Management
- Be mindful of message allocation
- Use appropriate data structures
- Consider message size and frequency
- Implement proper cleanup in destructors

#### Threading
- Use ROS 2's built-in multi-threading capabilities
- Avoid manual threading when possible
- Be careful with shared data between callbacks
- Consider using MultiThreadedExecutor when needed

### Documentation and Testing

#### Documentation
- Document public methods and classes
- Include usage examples
- Document parameters and their expected ranges
- Use docstrings following PEP 257

#### Testing
- Write unit tests for core logic
- Test different parameter configurations
- Test error conditions and edge cases
- Use integration tests for node interactions

## Advanced Topics

### Lifecycle Nodes

Creating nodes with explicit lifecycle management:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleTestNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_test_node')
        self.get_logger().info('Lifecycle node created')

    def on_configure(self, state):
        self.get_logger().info('Configuring lifecycle node')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating lifecycle node')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating lifecycle node')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up lifecycle node')
        return TransitionCallbackReturn.SUCCESS
```

### Composition

Running multiple nodes in a single process:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from my_robot_package.my_node import MyRobotNode

def main(args=None):
    rclpy.init(args=args)

    # Create multiple nodes
    node1 = MyRobotNode()
    node2 = AnotherNode()  # Another ROS 2 node

    # Create an executor to manage multiple nodes
    executor = MultiThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node1.destroy_node()
        node2.destroy_node()
        rclpy.shutdown()
```

## Building and Running

### Building the Package

To build a ROS 2 Python package:

```bash
# Source the ROS 2 installation
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution

# Navigate to your workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select my_robot_package

# Source the workspace
source install/setup.bash
```

### Running Nodes

```bash
# Run a specific node
ros2 run my_robot_package my_robot_node

# Run with parameters
ros2 run my_robot_package my_robot_node --ros-args -p robot_name:=my_robot -p max_velocity:=2.0

# Run a launch file
ros2 launch my_robot_package my_launch.py
```

## Chapter Quiz

import QuizComponent from '@site/src/components/QuizComponent/QuizComponent';

<QuizComponent
  quizData={{
    title: "Building ROS 2 Packages in Python Quiz",
    questions: [
      {
        question: "What is the main Python client library for ROS 2?",
        options: [
          "rospy",
          "rclpy",
          "roslibpy",
          "pyros"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "Which file contains the metadata for a ROS 2 package?",
        options: [
          "setup.py",
          "CMakeLists.txt",
          "package.xml",
          "setup.cfg"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "What method is used to initialize the ROS 2 client library in Python?",
        options: [
          "rclpy.start()",
          "rclpy.init()",
          "rclpy.run()",
          "rclpy.begin()"
        ],
        correctAnswerIndex: 1
      },
      {
        question: "Which command is used to run a ROS 2 node?",
        options: [
          "ros2 run package_name node_name",
          "ros2 execute package_name node_name",
          "ros2 start package_name node_name",
          "ros2 launch package_name node_name"
        ],
        correctAnswerIndex: 0
      },
      {
        question: "What is the purpose of the 'spin' function in ROS 2?",
        options: [
          "To rotate the robot physically",
          "To keep the node running and process callbacks",
          "To create a spinning animation",
          "To restart the node"
        ],
        correctAnswerIndex: 1
      }
    ]
  }}
/>

## Exercises

1. Create a ROS 2 Python package with a publisher that sends temperature sensor data and a subscriber that processes this data and logs warnings when temperature exceeds a threshold.

2. Implement a service server that calculates the distance between two 3D points and a client that calls this service with different coordinates.

3. Build a complete ROS 2 Python package with proper setup files, documentation, and unit tests.

## Summary

Building ROS 2 packages in Python provides a powerful and accessible way to develop robotics applications. The rclpy library offers comprehensive functionality for creating nodes, publishers, subscribers, services, and actions. Proper package structure, configuration files, and best practices ensure maintainable and reusable code. Python's ease of use combined with ROS 2's distributed architecture enables rapid development and prototyping of complex robotic systems.

## Further Reading

- ROS 2 Python Client Library (rclpy) Documentation
- Python Package Index (PyPI) Guidelines for ROS 2
- Advanced Python Programming for Robotics
- ROS 2 Testing and Quality Assurance Practices