import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the textbook with all 22 chapters
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'intro',
        {
          type: 'category',
          label: 'Part I: Foundations',
          items: [
            'introduction-to-physical-ai/index',
            'history-and-future-of-humanoid-robots/index',
            'embodied-intelligence-theory/index',
            'sensor-systems-lidar-depth-imu/index',
          ],
        },
        {
          type: 'category',
          label: 'Part II: ROS 2 and Simulation',
          items: [
            'understanding-ros2-the-robot-nervous-system/index',
            'ros2-nodes-topics-services-actions/index',
            'building-ros2-packages-in-python/index',
            'gazebo-simulation-basics/index',
            'urdf-sdf-robot-description-formats/index',
            'unity-robot-visualization/index',
            'nvidia-isaac-sim-foundations/index',
            'isaac-ros-ai-perception-and-navigation/index',
          ],
        },
        {
          type: 'category',
          label: 'Part III: Advanced Topics',
          items: [
            'vision-language-action-vla-systems/index',
            'voice-to-action-using-openai-whisper/index',
            'bipedal-locomotion-and-balance-control/index',
            'humanoid-arms-hands-and-grasping/index',
            'llm-planning-cognitive-reasoning-for-robots/index',
            'sim-to-real-transfer-techniques/index',
          ],
        },
        {
          type: 'category',
          label: 'Part IV: Implementation and Deployment',
          items: [
            'the-autonomous-humanoid-capstone/index',
            'hardware-requirements-lab-setup-guide/index',
            'cloud-vs-onpremise-robotics-labs/index',
            'glossary-and-resources/index',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
