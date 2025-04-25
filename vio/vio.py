#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from queue import Queue
from threading import Thread

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import numpy as np
from scipy.spatial.transform import Rotation as R
from vio import quaternion_from_axis_angle
from vio import ConfigEuRoC, ImageProcessor, MSCKF, EuRoCDataset, DataPublisher

class VIONode(Node):
    def __init__(self, dataset_path):
        super().__init__('vio_node')

        # Load configuration and dataset
        self.config = ConfigEuRoC()
        self.img_queue = Queue()
        self.imu_queue = Queue()
        self.feature_queue = Queue()


        self.dataset = EuRoCDataset(dataset_path)
        self.dataset.set_starttime(offset=40.0)

        # Initialize modules
        self.image_processor = ImageProcessor(self.config)
        self.msckf = MSCKF(self.config)

        # Create Transform Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Start IMU queue publisher
        self.imu_publisher = DataPublisher(
            self.dataset.imu, self.imu_queue, float('inf'), ratio=0.4)

        self.img_publisher = DataPublisher(
            self.dataset.stereo, self.img_queue, float('inf'), ratio=0.4)

        # Start IMU processing thread
        self.img_thread = Thread(target=self.process_img, daemon=True)
        self.imu_thread = Thread(target=self.process_imu, daemon=True)
        self.feature_thread = Thread(target=self.process_feature, daemon=True)

        self.imu_thread.start()
        self.img_thread.start()
        self.feature_thread.start()

        # Start playback
        now = self.get_clock().now().nanoseconds * 1e-9
        self.imu_publisher.start(now)
        self.img_publisher.start(now)

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return

            feature_msg = self.image_processor.stareo_callback(img_msg)
            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while rclpy.ok():
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                break

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

            # Publish both transforms
            self.publish_transforms()

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)


    def publish_transforms(self):
        tf_world_imu = TransformStamped()
        tf_world_imu.header.stamp = self.get_clock().now().to_msg()
        tf_world_imu.header.frame_id = 'world'            # <-- world is the parent
        tf_world_imu.child_frame_id = 'imu_link'          # <-- imu_link is rotated
        tf_world_imu.transform.translation.x = 1.0
        tf_world_imu.transform.translation.y = 1.0
        tf_world_imu.transform.translation.z = 0.0
        tf_world_imu.transform.rotation.x = self.msckf.state_server.imu_state.orientation[0]
        tf_world_imu.transform.rotation.y = self.msckf.state_server.imu_state.orientation[1]
        tf_world_imu.transform.rotation.z = self.msckf.state_server.imu_state.orientation[2]
        tf_world_imu.transform.rotation.w = self.msckf.state_server.imu_state.orientation[3]

        # --- 2. imu_link → cam0_link (from MSCKF state) ---
        try:
            rot = R.from_matrix(self.msckf.state_server.imu_state.R_cam0_imu)
            q = rot.as_quat()  # [x, y, z, w]
        except Exception as e:
            self.get_logger().error(f"Invalid rotation matrix: {e}")
            return

        t = self.msckf.state_server.imu_state.t_cam0_imu

        tf_imu_cam = TransformStamped()
        tf_imu_cam.header.stamp = self.get_clock().now().to_msg()
        tf_imu_cam.header.frame_id = 'imu_link'
        tf_imu_cam.child_frame_id = 'cam0_link'
        tf_imu_cam.transform.translation.x = t[0]
        tf_imu_cam.transform.translation.y = t[1]
        tf_imu_cam.transform.translation.z = t[2]
        tf_imu_cam.transform.rotation.x = q[0]
        tf_imu_cam.transform.rotation.y = q[1]
        tf_imu_cam.transform.rotation.z = q[2]
        tf_imu_cam.transform.rotation.w = q[3]

        try:
            rot = R.from_matrix(self.msckf.R_cam0_cam1)
            q = rot.as_quat()  # [x, y, z, w]
        except Exception as e:
            self.get_logger().error(f"Invalid rotation matrix: {e}")
            return
        t = self.msckf.t_cam0_cam1

        tf_cam0_cam1 = TransformStamped()
        tf_cam0_cam1.header.stamp = self.get_clock().now().to_msg()
        tf_cam0_cam1.header.frame_id = 'cam0_link'
        tf_cam0_cam1.child_frame_id = 'cam1_link'
        tf_cam0_cam1.transform.translation.x = t[0]
        tf_cam0_cam1.transform.translation.y = t[1]
        tf_cam0_cam1.transform.translation.z = t[2]
        tf_cam0_cam1.transform.rotation.x = q[0]
        tf_cam0_cam1.transform.rotation.y = q[1]
        tf_cam0_cam1.transform.rotation.z = q[2]
        tf_cam0_cam1.transform.rotation.w = q[3]

        try:
            rot = R.from_matrix(self.msckf.R_cam1_imu)
            q = rot.as_quat()  # [x, y, z, w]
        except Exception as e:
            self.get_logger().error(f"Invalid rotation matrix: {e}")
            return
        t = self.msckf.t_cam1_imu

        tf_imu_cam1 = TransformStamped()
        tf_imu_cam1.header.stamp = self.get_clock().now().to_msg()
        tf_imu_cam1.header.frame_id = 'imu_link'
        tf_imu_cam1.child_frame_id = 'cam2_link'
        tf_imu_cam1.transform.translation.x = t[0]
        tf_imu_cam1.transform.translation.y = t[1]
        tf_imu_cam1.transform.translation.z = t[2]
        tf_imu_cam1.transform.rotation.x = q[0]
        tf_imu_cam1.transform.rotation.y = q[1]
        tf_imu_cam1.transform.rotation.z = q[2]
        tf_imu_cam1.transform.rotation.w = q[3]

        # Broadcast both transforms
        self.tf_broadcaster.sendTransform([tf_world_imu, tf_imu_cam, tf_cam0_cam1, tf_imu_cam1])
        #self.tf_broadcaster.sendTransform([tf_world_imu, tf_imu_cam])


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./MH_01_easy',
                        help='Path of EuRoC MAV dataset.')
    args = parser.parse_args()

    rclpy.init(args=None)
    node = VIONode(args.path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
