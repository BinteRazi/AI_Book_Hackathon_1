---
sidebar_position: 6
title: "ROS 2 Nodes, Topics, Services, Actions"
---

# ROS 2 Nodes, Topics, Services, Actions

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental communication patterns in ROS 2
- Implement nodes using both C++ and Python
- Design effective topic-based communication systems
- Create and use services for request-response interactions
- Implement actions for long-running tasks with feedback
- Choose appropriate communication patterns for specific use cases

## Introduction

ROS 2 provides three primary communication patterns that form the backbone of distributed robotics applications: topics for asynchronous publish-subscribe communication, services for synchronous request-response interactions, and actions for asynchronous long-running tasks with feedback. Understanding when and how to use each pattern is crucial for building effective robot systems. These communication patterns enable the creation of modular, reusable components that can be combined to create complex robotic behaviors.

The node-based architecture of ROS 2 allows developers to break down complex robot behaviors into smaller, manageable components that communicate through standardized interfaces. This modularity enables code reuse, easier debugging, and collaborative development among robotics teams.

## Nodes: The Foundation of ROS 2

### Node Architecture

Nodes are the fundamental building blocks of ROS 2 applications, representing individual processes that perform specific computations. Each node can contain publishers, subscribers, services, and actions, and communicates with other nodes through the ROS 2 middleware.

#### Node Characteristics
- **Process Isolation**: Each node runs as a separate process
- **Unique Naming**: Nodes must have unique names within their namespace
- **Communication Interface**: Nodes provide publishers, subscribers, services, and actions
- **Parameter Management**: Nodes can declare and use parameters

### Creating Nodes

#### Python Implementation
```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')

        # Declare parameters
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Initialize node components
        self.get_logger().info('MyRobotNode initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

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

#### C++ Implementation
```cpp
#include <rclcpp/rclcpp.hpp>

class MyRobotNode : public rclcpp::Node
{
public:
    MyRobotNode() : Node("my_robot_node")
    {
        RCLCPP_INFO(this->get_logger(), "MyRobotNode initialized");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyRobotNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Node Lifecycle

ROS 2 nodes can implement a lifecycle to manage their state transitions:

#### Lifecycle States
- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Node is shutting down

#### Benefits
- Controlled startup and shutdown
- Resource management
- Safe state transitions
- Error recovery capabilities

## Topics: Publish-Subscribe Communication

### Topic Fundamentals

Topics enable asynchronous, one-to-many communication between nodes using a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics they are subscribed to. This decoupling allows for flexible system design where publishers and subscribers don't need to know about each other.

#### Key Characteristics
- **Asynchronous**: Publishers and subscribers operate independently
- **Many-to-Many**: Multiple publishers and subscribers can use the same topic
- **Message-Based**: Communication occurs through standardized messages
- **Type-Safe**: Messages have defined types and structures

### Quality of Service (QoS)

QoS settings control the delivery guarantees and behavior of topic communication:

#### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost, but delivery is faster

#### Durability Policy
- **Transient Local**: Late-joining subscribers receive last message
- **Volatile**: Only new messages are delivered

#### History Policy
- **Keep Last**: Store only the most recent N messages
- **Keep All**: Store all messages (limited by resource constraints)

### Publisher Implementation

#### Python Publisher
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
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
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()
```

#### C++ Publisher
```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

### Subscriber Implementation

#### Python Subscriber
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()
```

## Services: Request-Response Communication

### Service Architecture

Services provide synchronous, one-to-one communication for request-response interactions. When a client sends a request to a service, it waits for the server to process the request and return a response. This pattern is suitable for operations that have a clear beginning and end, such as configuration changes, data queries, or simple computations.

#### Service Characteristics
- **Synchronous**: Client waits for response
- **One-to-One**: Direct communication between client and server
- **Request-Response**: Defined request and response message types
- **Blocking**: Client is blocked until response is received

### Service Definition

Services are defined using .srv files that specify the request and response message types:

```
# AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

The part before `---` defines the request message, and the part after defines the response message.

### Service Server Implementation

#### Python Service Server
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

#### C++ Service Server
```cpp
#include <example_interfaces/srv/add_two_ints.hpp>
#include <rclcpp/rclcpp.hpp>

class MinimalService : public rclcpp::Node
{
public:
    MinimalService() : Node("minimal_service")
    {
        service_ = create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            [this](const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
                   example_interfaces::srv::AddTwoInts::Response::SharedPtr response) {
                response->sum = request->a + request->b;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                           "Incoming request\na: %ld, b: %ld", request->a, request->b);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sending back response: [%ld]", response->sum);
            });
    }

private:
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

### Service Client Implementation

#### Python Service Client
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    future = minimal_client.send_request(1, 2)

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                minimal_client.get_logger().error(f'Service call failed: {e}')
            else:
                minimal_client.get_logger().info(
                    f'Result of add_two_ints: {response.sum}')
            break

    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Actions: Long-Running Tasks with Feedback

### Action Architecture

Actions provide asynchronous communication for long-running tasks that require feedback and the ability to cancel. Unlike services, actions don't block the client, allowing for more responsive applications. Actions include three message types: goal, feedback, and result.

#### Action Components
- **Goal**: Request for a long-running task
- **Feedback**: Periodic updates on task progress
- **Result**: Final outcome of the task

### Action Use Cases

Actions are appropriate for:
- Navigation to a goal location
- Object manipulation tasks
- Calibration procedures
- Any task requiring progress monitoring

### Action Implementation

#### Action Definition
```
# Fibonacci.action
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

#### Python Action Server
```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
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
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)  # Simulate work

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))

        return result
```

#### Python Action Client
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
```

## Communication Pattern Selection

### When to Use Each Pattern

#### Topics
- **Best for**: Sensor data publishing, status updates, continuous streams
- **Characteristics**: Asynchronous, many-to-many, decoupled timing
- **Examples**: Camera images, laser scans, joint states

#### Services
- **Best for**: Configuration changes, data queries, simple computations
- **Characteristics**: Synchronous, one-to-one, blocking
- **Examples**: Setting parameters, requesting data, simple calculations

#### Actions
- **Best for**: Long-running tasks with progress feedback
- **Characteristics**: Asynchronous, one-to-one, cancellable
- **Examples**: Navigation, manipulation, calibration

### Performance Considerations

#### Topic Performance
- Message size affects network bandwidth
- Publication frequency affects CPU usage
- QoS settings impact delivery guarantees
- Memory usage scales with history settings

#### Service Performance
- Synchronous nature affects application responsiveness
- Error handling is critical for robustness
- Request/response size impacts latency
- Timeout handling prevents indefinite blocking

#### Action Performance
- More complex implementation than services
- Feedback frequency affects network usage
- Cancellation mechanisms add complexity
- State management is required

## Advanced Topics

### Namespaces and Composition

#### Namespaces
Namespaces provide logical grouping and prevent naming conflicts:
- Hierarchical organization similar to filesystem paths
- Enable multiple instances of similar components
- Support for robot-specific and application-specific scopes

#### Composition
Multiple nodes can be combined into a single process:
- Reduced inter-process communication overhead
- Simplified deployment and management
- Shared memory communication
- Single point of failure consideration

### Introspection and Debugging

#### Command Line Tools
- `ros2 topic`: Monitor and interact with topics
- `ros2 service`: Call services and monitor status
- `ros2 action`: Monitor and interact with actions
- `ros2 node`: List and manage nodes

#### Visualization Tools
- **rqt**: Qt-based GUI tools for monitoring
- **rviz2**: 3D visualization for robotics data
- **PlotJuggler**: Real-time plotting of ROS 2 data
- **Custom dashboards**: Application-specific monitoring

## Best Practices

### Design Principles

#### Modularity
- Single responsibility per node
- Clear interfaces between components
- Reusable components across applications
- Proper error handling and recovery

#### Performance
- Appropriate message rates for application needs
- Efficient message serialization
- Proper QoS configuration
- Resource management

### Error Handling

#### Robust Communication
- Handle connection failures gracefully
- Implement timeout mechanisms
- Provide meaningful error messages
- Fallback behaviors when possible

## Chapter Quiz

import QuizComponent from '@site/src/components/QuizComponent/QuizComponent';

<QuizComponent
  quizData={{
    title: "ROS 2 Communication Patterns Quiz",
    questions: [
      {
        question: "What are the three primary communication patterns in ROS 2?",
        options: [
          "Publish, Subscribe, Request",
          "Nodes, Topics, Services",
          "Topics, Services, Actions",
          "Publisher, Subscriber, Server"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "Which communication pattern is appropriate for long-running tasks with feedback?",
        options: [
          "Topics",
          "Services",
          "Actions",
          "Parameters"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "What does QoS stand for in ROS 2 communication?",
        options: [
          "Quality of Service",
          "Quick Operating System",
          "Query and Subscribe",
          "Quantum Operating System"
        ],
        correctAnswerIndex: 0
      },
      {
        question: "Which QoS policy controls whether late-joining subscribers receive previous messages?",
        options: [
          "Reliability",
          "History",
          "Durability",
          "Deadline"
        ],
        correctAnswerIndex: 2
      },
      {
        question: "What are the three components of an action in ROS 2?",
        options: [
          "Request, Response, Feedback",
          "Goal, Feedback, Result",
          "Input, Process, Output",
          "Start, Run, Stop"
        ],
        correctAnswerIndex: 1
      }
    ]
  }}
/>

## Exercises

1. Create a ROS 2 package with a publisher that sends sensor data and a subscriber that processes this data. Test the system with different QoS settings.

2. Implement a service server that performs a calculation (e.g., distance between two points) and a client that calls this service with different parameters.

3. Design an action server for a robot navigation task that includes goal validation, feedback on progress, and result reporting.

## Summary

ROS 2 provides three primary communication patterns - topics, services, and actions - each designed for specific types of interactions in robotics applications. Topics enable asynchronous, decoupled communication suitable for sensor data and status updates. Services provide synchronous request-response communication for discrete operations. Actions support long-running tasks with feedback and cancellation capabilities. Understanding when and how to use each pattern is essential for building effective, maintainable robotics applications.

## Further Reading

- ROS 2 Documentation on Communication Patterns
- Quality of Service Settings Guide
- Advanced ROS 2 Programming Techniques
- Real-time Considerations in ROS 2