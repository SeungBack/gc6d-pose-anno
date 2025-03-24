# GraspClutter6D Pose Annotator

![GraspClutter6D Pose Annotator Interface](./example.gif)

## Overview

The GraspClutter6D Pose Annotator is a specialized tool designed for:
- Annotating 6D object poses in datasets that follow the BOP format
- Monitoring annotation quality using depth difference metrics
- Copying and pasting annotations across multiple images within the same scene

## Installation

```bash
# Create and activate conda environment
conda create -n gc6d-anno python=3.7
conda activate gc6d-anno

# Install dependencies
pip install -r requirements.txt

# Launch the application
python object_pose_annotator.py
```

## User Guide

### Opening a Scene
1. Click the **Open File** button in the top-right corner
2. Navigate to `GraspClutter6D_root/scenes/scene_you_want/rgb/image_you_want`
3. The point cloud, image, and annotation will automatically load

### Object Pose Manipulation
1. Add objects you want to annotate in the 'Annotation Objects' panel
2. Select the object you wish to annotate
3. Use the following keyboard shortcuts:

| Key Combination | Action |
|----------------|--------|
| `W` | Move object up |
| `A` | Move object left |
| `S` | Move object down |
| `D` | Move object right |
| `Q` | Move object outward (away from camera) |
| `E` | Move object inward (toward camera) |
| `Shift + W/A/S/D/Q/E` | Rotate object with respect to camera coordinate frame |
| `Ctrl + Left-click` | Move the object to the clicked position |

### Scene Point Cloud Navigation
- **Left-click + drag**: Rotate viewpoint
- **Right-click + drag**: Translate viewpoint

### Image Panel Navigation
| Key | Action |
|-----|--------|
| `I` | Move image up |
| `K` | Move image down |
| `J` | Move image left |
| `L` | Move image right |
| `U` | Zoom in |
| `O` | Zoom out |
| `P` | Reset to default view |

### Interface Customization
- You can adjust the following settings in the right-top panel:
  - **Responsiveness**: Modify sensitivity for pose orientation control
  - **Point Size**: Change the size of points in the point cloud visualization
  - **Transparency**: Adjust the transparency level of rendered objects


### Additional Functions
| Key | Action |
|-----|--------|
| `T` | Reset to initial camera viewpoint |
| `R` | Refine object poses using Iterative Closest Points (ICP) algorithm |

### Cross-Image Annotation
You can copy object poses across different images within the same scene:
1. Locate the **Copy Annotation** function in the bottom-right corner
2. Designate the source image ID (containing the annotation to be copied)
3. Apply to the target image (where the annotation will be pasted)
4. The tool uses camera poses and intrinsic parameters to properly align annotations

### Saving and Quality Assessment
- Annotations are saved to each scene directory in `scene_gt.json` using the BOP format
- After saving, segmentation masks and annotation quality metrics are automatically updated
- The `Annotation Quality` panel displays absolute depth differences in millimeters, allowing you to monitor the precision of your annotations