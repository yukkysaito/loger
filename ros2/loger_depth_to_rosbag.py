#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rosbag2_py
import torch
import yaml
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as RosImage

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class FrameRecord:
    index: int
    bag_timestamp_ns: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    png_path: Path


SUPPORTED_IMAGE_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an image topic from a ROS 2 bag, run LoGeR, and write resized "
            "depth images back into a new ROS 2 bag with preserved timestamps."
        )
    )
    parser.add_argument(
        "input_bag",
        type=Path,
        help="Path to an input rosbag directory or a single .mcap/.db3 file.",
    )
    parser.add_argument(
        "image_topic",
        help="Input image topic. Supports sensor_msgs/msg/CompressedImage and sensor_msgs/msg/Image.",
    )
    parser.add_argument(
        "-o",
        "--output-bag",
        type=Path,
        required=True,
        help="Path to the output rosbag directory to create.",
    )
    parser.add_argument(
        "--depth-topic",
        help="Output depth topic name. Defaults to a name derived from the input image topic.",
    )
    parser.add_argument(
        "--storage-id",
        choices=("mcap", "sqlite3"),
        help="Optional input bag storage id. If omitted, inferred from the input path.",
    )
    parser.add_argument(
        "--output-storage-id",
        choices=("mcap", "sqlite3"),
        help="Optional output bag storage id. Defaults to the input storage id.",
    )
    parser.add_argument(
        "--model-name",
        type=Path,
        default=REPO_ROOT / "ckpts/LoGeR/latest.pt",
        help="LoGeR checkpoint path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ckpts/LoGeR/original_config.yaml",
        help="LoGeR config YAML path.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional resized inference resolution. If omitted, LoGeR auto-selects one size for the sequence.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for LoGeR inference.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional override for LoGeR window size.",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=None,
        help="Optional override for LoGeR overlap size.",
    )
    parser.add_argument(
        "--reset-every",
        type=int,
        default=None,
        help="Optional override for LoGeR reset cadence.",
    )
    parser.add_argument(
        "--sim3",
        action="store_true",
        help="Enable Sim3 alignment during LoGeR inference.",
    )
    parser.add_argument(
        "--se3",
        action="store_true",
        help="Enable SE3 alignment during LoGeR inference.",
    )
    parser.add_argument(
        "--sim3-scale-mode",
        default="median",
        choices=("median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"),
        help="Scale estimation mode used when --sim3 is enabled.",
    )
    parser.add_argument(
        "--no-ttt",
        action="store_true",
        help="Disable TTT memory during LoGeR inference.",
    )
    parser.add_argument(
        "--no-swa",
        action="store_true",
        help="Disable sliding-window attention memory during LoGeR inference.",
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
        raise ValueError(f"Could not infer storage id from mixed bag files under: {bag_path}")

    raise ValueError(f"Could not infer storage id from: {bag_path}")


def resolve_bag_path(bag_path: Path, storage_id: Optional[str]) -> tuple[Path, str]:
    resolved = bag_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Bag path does not exist: {resolved}")
    return resolved, storage_id or infer_storage_id(resolved)


def default_depth_topic(image_topic: str) -> str:
    base = image_topic[:-11] if image_topic.endswith("/compressed") else image_topic
    return f"{base}/loger_depth"


def make_reader(bag_path: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )
    return reader


def topic_metadata_map(reader: rosbag2_py.SequentialReader) -> dict[str, object]:
    return {metadata.name: metadata for metadata in reader.get_all_topics_and_types()}


def validate_image_topic(image_topic: str, metadata_by_topic: dict[str, object]) -> object:
    if image_topic not in metadata_by_topic:
        available = ", ".join(sorted(metadata_by_topic))
        raise ValueError(f"Topic not found: {image_topic}. Available topics: {available}")

    metadata = metadata_by_topic[image_topic]
    if metadata.type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported topic type for {image_topic}: {metadata.type}. "
            f"Supported types: {sorted(SUPPORTED_IMAGE_TYPES)}"
        )
    return metadata


def decode_to_rgb_array(data: bytes, message_type: str, bridge: CvBridge) -> tuple[np.ndarray, object]:
    if message_type == "sensor_msgs/msg/CompressedImage":
        msg = deserialize_message(data, CompressedImage)
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to decode compressed image with format: {msg.format}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, msg.header

    msg = deserialize_message(data, RosImage)
    rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    return np.asarray(rgb), msg.header


def extract_frames_to_pngs(
    bag_path: Path,
    storage_id: str,
    image_topic: str,
    image_topic_type: str,
    temp_dir: Path,
) -> list[FrameRecord]:
    reader = make_reader(bag_path, storage_id)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[image_topic]))
    bridge = CvBridge()
    frame_records: list[FrameRecord] = []

    while reader.has_next():
        topic, data, bag_timestamp_ns = reader.read_next()
        if topic != image_topic:
            continue

        rgb, header = decode_to_rgb_array(data, image_topic_type, bridge)
        png_path = temp_dir / f"{len(frame_records):06d}.png"
        PILImage.fromarray(rgb).save(png_path)
        frame_records.append(
            FrameRecord(
                index=len(frame_records),
                bag_timestamp_ns=bag_timestamp_ns,
                stamp_sec=int(header.stamp.sec),
                stamp_nanosec=int(header.stamp.nanosec),
                frame_id=str(header.frame_id),
                png_path=png_path,
            )
        )

    if not frame_records:
        raise RuntimeError(f"No frames found on topic: {image_topic}")

    return frame_records


def build_forward_kwargs(args: argparse.Namespace) -> dict[str, object]:
    forward_kwargs: dict[str, object] = {}
    if args.config is None or not args.config.exists():
        return forward_kwargs

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    training_settings = config.get("training_settings", {})
    model_settings = config.get("model", {})
    se3_from_config = model_settings.get("se3", config.get("se3", False))
    se3_value = True if args.se3 else bool(se3_from_config)

    forward_kwargs.update(
        {
            "window_size": args.window_size if args.window_size is not None else training_settings.get("window_size", -1),
            "overlap_size": args.overlap_size if args.overlap_size is not None else training_settings.get("overlap_size", 0),
            "reset_every": args.reset_every if args.reset_every is not None else training_settings.get("reset_every", 0),
            "num_iterations": config.get("num_iterations", 1),
            "sim3": bool(config.get("sim3", False) or args.sim3),
            "sim3_scale_mode": args.sim3_scale_mode,
            "se3": se3_value,
            "turn_off_ttt": args.no_ttt,
            "turn_off_swa": args.no_swa,
        }
    )
    return forward_kwargs


def run_loger_depth_inference(frame_records: list[FrameRecord], args: argparse.Namespace) -> np.ndarray:
    try:
        from demo_viser import load_images_from_paths, load_pi3_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LoGeR dependencies are missing. Install the project requirements before running inference."
        ) from exc

    model_path = args.model_name.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config is not None else None
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model = load_pi3_model(str(model_path), str(config_path) if config_path is not None else None)
    if model is None:
        raise RuntimeError("Failed to initialize LoGeR model")

    device = args.device
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model = model.to(device).eval()

    image_paths = [str(frame.png_path) for frame in frame_records]
    target_resolution = tuple(args.resolution) if args.resolution is not None else None
    images_tensor = load_images_from_paths(
        image_paths,
        Target_W=target_resolution[0] if target_resolution is not None else None,
        Target_H=target_resolution[1] if target_resolution is not None else None,
    ).to(device)

    if images_tensor.numel() == 0:
        raise RuntimeError("LoGeR input image tensor is empty")

    forward_kwargs = build_forward_kwargs(args)
    if use_cuda:
        major, _minor = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda, dtype=dtype):
        predictions = model(images_tensor[None], **forward_kwargs)

    local_points = predictions.get("local_points")
    if local_points is None:
        raise RuntimeError("LoGeR output does not contain local_points, cannot derive depth images")

    depths = local_points[0, ..., 2].detach().cpu().float().numpy()
    if depths.shape[0] != len(frame_records):
        raise RuntimeError(
            f"Depth count mismatch: got {depths.shape[0]} depth maps for {len(frame_records)} frames"
        )
    return depths


def create_output_writer(output_bag: Path, storage_id: str) -> rosbag2_py.SequentialWriter:
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )
    return writer


def copy_bag_and_insert_depth(
    input_bag: Path,
    input_storage_id: str,
    output_bag: Path,
    output_storage_id: str,
    image_topic: str,
    depth_topic: str,
    image_metadata: object,
    frame_records: list[FrameRecord],
    depths: np.ndarray,
) -> None:
    if output_bag.exists():
        raise FileExistsError(f"Output bag path already exists: {output_bag}")
    output_bag.parent.mkdir(parents=True, exist_ok=True)

    reader = make_reader(input_bag, input_storage_id)
    metadata_by_topic = topic_metadata_map(reader)
    if depth_topic in metadata_by_topic:
        raise ValueError(f"Depth topic already exists in the input bag: {depth_topic}")
    writer = create_output_writer(output_bag, output_storage_id)
    bridge = CvBridge()

    for metadata in metadata_by_topic.values():
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=metadata.name,
                type=metadata.type,
                serialization_format=metadata.serialization_format,
                offered_qos_profiles=getattr(metadata, "offered_qos_profiles", ""),
            )
        )

    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=depth_topic,
            type="sensor_msgs/msg/Image",
            serialization_format=getattr(image_metadata, "serialization_format", "cdr"),
            offered_qos_profiles=getattr(image_metadata, "offered_qos_profiles", ""),
        )
    )

    frame_index = 0
    while reader.has_next():
        topic, data, bag_timestamp_ns = reader.read_next()
        writer.write(topic, data, bag_timestamp_ns)

        if topic != image_topic:
            continue

        if frame_index >= len(frame_records):
            raise RuntimeError("Encountered more image messages during copy than during extraction")

        frame = frame_records[frame_index]
        if bag_timestamp_ns != frame.bag_timestamp_ns:
            raise RuntimeError(
                "Image topic ordering changed between extraction and copy pass: "
                f"expected bag timestamp {frame.bag_timestamp_ns}, got {bag_timestamp_ns}"
            )

        depth_image = np.ascontiguousarray(depths[frame_index].astype(np.float32))
        depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp.sec = frame.stamp_sec
        depth_msg.header.stamp.nanosec = frame.stamp_nanosec
        depth_msg.header.frame_id = frame.frame_id
        writer.write(depth_topic, serialize_message(depth_msg), bag_timestamp_ns)
        frame_index += 1

    if frame_index != len(frame_records):
        raise RuntimeError(f"Only wrote {frame_index} depth images for {len(frame_records)} extracted frames")


def main() -> int:
    args = parse_args()
    input_bag, input_storage_id = resolve_bag_path(args.input_bag, args.storage_id)
    output_bag = args.output_bag.expanduser().resolve()
    depth_topic = args.depth_topic or default_depth_topic(args.image_topic)
    output_storage_id = args.output_storage_id or input_storage_id

    reader = make_reader(input_bag, input_storage_id)
    metadata_by_topic = topic_metadata_map(reader)
    image_metadata = validate_image_topic(args.image_topic, metadata_by_topic)

    with tempfile.TemporaryDirectory(prefix="loger_rosbag_frames_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        print(f"Extracting frames from {args.image_topic}...")
        frame_records = extract_frames_to_pngs(
            bag_path=input_bag,
            storage_id=input_storage_id,
            image_topic=args.image_topic,
            image_topic_type=image_metadata.type,
            temp_dir=temp_dir,
        )
        print(f"Extracted {len(frame_records)} frames to {temp_dir}")

        print("Running LoGeR depth inference...")
        depths = run_loger_depth_inference(frame_records, args)
        print(f"LoGeR produced {depths.shape[0]} depth maps at resolution {depths.shape[2]}x{depths.shape[1]}")

        print(f"Writing output bag to {output_bag}...")
        copy_bag_and_insert_depth(
            input_bag=input_bag,
            input_storage_id=input_storage_id,
            output_bag=output_bag,
            output_storage_id=output_storage_id,
            image_topic=args.image_topic,
            depth_topic=depth_topic,
            image_metadata=image_metadata,
            frame_records=frame_records,
            depths=depths,
        )

    print(f"Done. Depth topic written to {depth_topic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
