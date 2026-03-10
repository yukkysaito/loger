# ROS 2 Utilities

## `extract_compressed_topic_to_png.py`

This script extracts a `sensor_msgs/msg/CompressedImage` topic from a ROS 2 bag (`mcap` or `db3`) and saves the images as PNG files.

### Requirements

- `python3`
- `rosbag2_py`
- `rclpy`
- `sensor_msgs`
- `opencv-python`
- `numpy`

Run it after sourcing your ROS 2 environment.

### Usage

```bash
python3 ros2/extract_compressed_topic_to_png.py \
  <bag_path> \
  <topic_name> \
  -o <output_dir>
```

### Arguments

- `bag_path`
  - Path to a ROS 2 bag directory, or a single `.mcap` / `.db3` file
- `topic_name`
  - Compressed image topic to export
- `-o`, `--output-dir`
  - Output directory for PNG files
  - Created automatically if it does not exist
- `--storage-id`
  - Optional
  - Either `mcap` or `sqlite3`
  - Usually not needed because the script infers it from the input path

### Output File Name Format

PNG files are saved with the following format:

```text
000000_<timestamp_ns>.png
```

- The first 6 digits are a zero-padded sequential index
- `<timestamp_ns>` is taken from `CompressedImage.header.stamp`
- If `header.stamp` is zero, the bag message timestamp is used instead

### Examples

#### Using a bag directory

```bash
python3 ros2/extract_compressed_topic_to_png.py \
  ./data/sample_bag \
  /sensing/camera/camera4/image_raw/compressed \
  -o ./ros2/output/camera4_png_test
```

#### Using a single mcap file

```bash
python3 ros2/extract_compressed_topic_to_png.py \
  ./data/sample_bag/sample_0.mcap \
  /sensing/camera/camera4/image_raw/compressed \
  -o ./ros2/output/camera4_png_test
```

### Verified Result

Verified on 2026-03-11 with a local `mcap` bag containing this topic:

- Topic: `/sensing/camera/camera4/image_raw/compressed`
- Output directory: `./ros2/output/camera4_png_test`
- Saved images: `2392`
- First image size: `2880 x 1860`

## `loger_depth_to_rosbag.py`

This script reads an image topic from a ROS 2 bag, runs LoGeR on the extracted frames, and writes resized depth maps back into a new ROS 2 bag as `sensor_msgs/msg/Image` with `32FC1` encoding.

### What It Preserves

- Original bag message timestamp for each source image message
- Original `header.stamp`
- Original `header.frame_id`

### Output Format

- ROS message type: `sensor_msgs/msg/Image`
- Encoding: `32FC1`
- Resolution: LoGeR inference resolution after resizing

### Usage

```bash
python3 ros2/loger_depth_to_rosbag.py \
  <input_bag> \
  <image_topic> \
  -o <output_bag>
```

### Example

```bash
python3 ros2/loger_depth_to_rosbag.py \
  ./data/sample_bag \
  /sensing/camera/camera4/image_raw/compressed \
  -o ./ros2/output/sample_bag_with_depth
```

### Notes

- The input topic may be `sensor_msgs/msg/CompressedImage` or `sensor_msgs/msg/Image`
- The output bag is written as a new bag; the original bag is left unchanged
- The default depth topic is derived from the image topic, for example:
  - `/sensing/camera/camera4/image_raw/compressed`
  - `/sensing/camera/camera4/image_raw/loger_depth`
- LoGeR model weights are required to run inference
