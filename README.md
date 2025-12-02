# Distance Calculation of Coastal Bird Species

A Python-based project that calculates the distance of coastal bird species from the camera using computer vision techniques and video footage.

## Overview

This project implements an algorithm to calculate the distance of objects (specifically coastal birds) in video footage, displaying the results in metres. It uses the pinhole camera model and object detection to estimate distances based on the perceived size of objects in the frame.

## Features

- **Distance Calculation Algorithm**: Uses pinhole camera model to calculate distances
- **Video Processing**: Process video files frame-by-frame with object detection
- **Real-time Display**: Shows distance measurements overlaid on video frames
- **Camera Calibration**: Tools to calibrate your camera's focal length
- **Species-Specific Configurations**: Pre-configured settings for common coastal birds
- **Background Subtraction**: Automatic object detection using motion detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mateenojcod/-Distance-Calculation-of-Coastal-Bird-Species.git
cd -Distance-Calculation-of-Coastal-Bird-Species
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python >= 4.8.0)
- NumPy (>= 1.24.0)

## Usage

### Basic Distance Calculation

```python
from distance_calculator import DistanceCalculator

# Initialize calculator
calculator = DistanceCalculator(
    known_width=0.3,      # Object width in metres
    focal_length=750       # Camera focal length in pixels
)

# Calculate distance
perceived_width = 200  # Width in pixels as seen in frame
distance = calculator.calculate_distance(perceived_width)
print(f"Distance: {distance:.2f}m")
```

### Camera Calibration

Before accurate distance measurement, you need to calibrate your camera:

```python
calculator = DistanceCalculator(known_width=0.3, focal_length=100)

# Calibrate using a known distance
# E.g., object is 10m away and appears 280px wide
focal_length = calculator.calibrate_focal_length(
    known_distance=10.0,
    perceived_width=280
)
print(f"Calibrated focal length: {focal_length:.2f}px")
```

### Video Processing

Process a video file and calculate distances to detected objects:

```python
from distance_calculator import DistanceCalculator, VideoDistanceProcessor

# Setup
calculator = DistanceCalculator(known_width=0.3, focal_length=750)
processor = VideoDistanceProcessor(calculator)

# Process video
distances = processor.process_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4',  # Optional
    display=True  # Show video during processing
)

# Analyze results
if distances:
    avg_distance = sum(d[1] for d in distances) / len(distances)
    print(f"Average distance: {avg_distance:.2f}m")
```

### Running Examples

The project includes comprehensive examples:

```bash
# Basic distance calculations
python distance_calculator.py

# All usage examples
python example.py
```

## How It Works

### Pinhole Camera Model

The algorithm uses the pinhole camera model formula:

```
Distance = (Known_Width × Focal_Length) / Perceived_Width
```

Where:
- **Known_Width**: Actual width of the object in metres
- **Focal_Length**: Camera focal length in pixels (calibrated)
- **Perceived_Width**: Width of object in the frame in pixels

### Object Detection

The system uses background subtraction (MOG2) to detect moving objects:
1. Applies background subtraction to identify foreground objects
2. Removes shadows and noise using morphological operations
3. Finds contours and identifies the largest object
4. Calculates the width of the bounding box
5. Computes distance using the formula above

## Species Configuration

Different bird species require different configurations. Here are some examples:

| Species    | Average Width | Example Config |
|------------|--------------|----------------|
| Seagull    | 0.4m         | `DistanceCalculator(0.4, 800)` |
| Pelican    | 0.6m         | `DistanceCalculator(0.6, 800)` |
| Tern       | 0.25m        | `DistanceCalculator(0.25, 800)` |
| Albatross  | 0.8m         | `DistanceCalculator(0.8, 800)` |
| Cormorant  | 0.35m        | `DistanceCalculator(0.35, 800)` |

## Project Structure

```
.
├── distance_calculator.py   # Main algorithm implementation
├── example.py              # Usage examples and demonstrations
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## Technical Details

### DistanceCalculator Class

Core class for distance calculations:
- `calculate_distance(perceived_width)`: Calculate distance to an object
- `calibrate_focal_length(known_distance, perceived_width)`: Calibrate camera

### VideoDistanceProcessor Class

Handles video processing:
- `detect_object_width(frame)`: Detect objects and measure width
- `process_video(video_path, output_path, display)`: Process entire video
- `process_frame(frame)`: Process single frame

## Calibration Tips

1. **Use a known reference**: Place an object of known size at a measured distance
2. **Multiple measurements**: Take several calibration measurements for accuracy
3. **Same conditions**: Calibrate under similar lighting and conditions as your actual footage
4. **Camera settings**: Keep camera zoom and focus settings consistent

## Future Enhancements

- Machine learning-based object detection (YOLO, SSD)
- Automatic species identification
- Multi-object tracking
- Distance tracking over time
- Export results to CSV/JSON
- GUI application

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

This project uses computer vision techniques for wildlife monitoring and research of coastal bird species.
