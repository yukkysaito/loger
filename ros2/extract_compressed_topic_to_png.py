#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compressed image topic from a ROS 2 bag into PNG files."
    )
    parser.add_argument(
        "bag_path",
        type=Path,
        help="Path to a rosbag directory or a single .mcap/.db3 bag file.",
    )
    parser.add_argument(
        "topic",
        help="Compressed image topic to extract, for example /camera/image_raw/compressed.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where PNG files will be written.",
    )
    parser.add_argument(
        "--storage-id",
        choices=("mcap", "sqlite3"),
        help="Optional rosbag2 storage id. If omitted, it is inferred from the input.",
    )
    return parser.parse_args()


def infer_storage_id(bag_path: Path) -> str:
    if bag_path.is_file():
        suffix = bag_path.suffix.lower()
        if suffix == ".mcap":
            return "mcap"
        if suffix == ".db3":
            return "sqlite3"
        raise ValueError(f"Unsupported bag file extension: {bag_path.suffix}")

    if not bag_path.is_dir():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    has_mcap = any(bag_path.glob("*.mcap"))
    has_db3 = any(bag_path.glob("*.db3"))

    if has_mcap and not has_db3:
        return "mcap"
    if has_db3 and not has_mcap:
        return "sqlite3"
    if has_mcap and has_db3:
        raise ValueError(
            f"Could not infer storage id from mixed bag files under: {bag_path}"
        )

    raise ValueError(f"Could not infer storage id from: {bag_path}")


def resolve_bag_path(bag_path: Path, storage_id: str | None) -> tuple[Path, str]:
    resolved = bag_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Bag path does not exist: {resolved}")

    return resolved, storage_id or infer_storage_id(resolved)


def topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {metadata.name: metadata.type for metadata in reader.get_all_topics_and_types()}


def validate_topic(topic: str, topics: dict[str, str]) -> None:
    if topic not in topics:
        available = ", ".join(
            sorted(name for name, msg_type in topics.items() if msg_type == "sensor_msgs/msg/CompressedImage")
        )
        raise ValueError(
            f"Topic not found: {topic}. Available compressed topics: {available or '(none)'}"
        )

    if topics[topic] != "sensor_msgs/msg/CompressedImage":
        raise ValueError(
            f"Topic {topic} has type {topics[topic]}, expected sensor_msgs/msg/CompressedImage."
        )


def decode_image(msg: CompressedImage) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode compressed image with format: {msg.format}")
    return image


def image_timestamp_ns(msg: CompressedImage, bag_timestamp_ns: int) -> int:
    stamp = msg.header.stamp
    if stamp.sec == 0 and stamp.nanosec == 0:
        return bag_timestamp_ns
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def main() -> int:
    args = parse_args()
    bag_path, storage_id = resolve_bag_path(args.bag_path, args.storage_id)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )

    topics = topic_type_map(reader)
    validate_topic(args.topic, topics)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    saved_count = 0
    while reader.has_next():
        topic, data, bag_timestamp_ns = reader.read_next()
        if topic != args.topic:
            continue

        msg = deserialize_message(data, CompressedImage)
        image = decode_image(msg)
        timestamp_ns = image_timestamp_ns(msg, bag_timestamp_ns)
        output_path = output_dir / f"{saved_count:06d}_{timestamp_ns}.png"

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write PNG file: {output_path}")

        saved_count += 1

    if saved_count == 0:
        raise RuntimeError(f"No messages were found for topic: {args.topic}")

    print(f"Saved {saved_count} PNG files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
