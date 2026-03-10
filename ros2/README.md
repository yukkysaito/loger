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
