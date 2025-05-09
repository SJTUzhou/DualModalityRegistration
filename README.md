# Dual Modality Registration Module for 3D Slicer

A 3D Slicer extension for registering medical images of different modalities using CT as a reference. This module is particularly useful for aligning images from different modalities (such as PET, SPECT, or segmentation masks) with a reference CT image.

## Features

- Support for multiple registration types:
  - Rigid registration
  - Affine registration
  - Deformable registration
  - Resampling only
  - Using existing transformation files
- Handles various image modalities:
  - CT images
  - PET images
  - SPECT images
  - Segmentation masks
  - Dose maps
- Options to:
  - Keep original image spacing
  - Load registered output directly into Slicer
  - Save transformation files for future use
  - Use existing transformation files

## Installation

### Prerequisites
1. Download and install 3D-Slicer from [https://download.slicer.org/](https://download.slicer.org/)

### Step-by-Step Installation Guide

1. **Install Required Python Packages**
   - Open the `bin` folder under the installation path of 3D-Slicer on your computer
   - Right-click in the folder and select "Open in Terminal"
   - Enter the following command and press Enter:
     ```bash
     ./PythonSlicer.exe -m pip install niftyreg nibabel numpy scipy
     ```

2. **Install the Module**
   - Download the `DualModalityRegistration.zip` file
   - Unzip it to a folder of your choice
   - Open 3D-Slicer
   - Click "Edit" in the upper left corner
   - Select "Application Settings"
   - Add the module in the following order:
     1. Click "Add"
     2. Navigate to the unzipped `DualModalityRegistration` folder
     3. Select the subfolder named `DualModalityRegistration` (note: the module and plugin have the same name)
     4. Confirm the selection
   - Restart 3D Slicer when prompted

3. **Verify Installation**
   - After restarting, the module should appear under "Modules → Examples → Dual Modality Registration"
   - If you can see the module interface, the installation was successful

## Usage

### Basic Workflow

1. Open 3D Slicer and load your images
2. Go to Modules → Examples → Dual Modality Registration
3. Set up your registration:

#### Input Images
- **Source CT**: Select the CT image that corresponds to your source modality
- **Image X**: Select the image you want to register (PET, SPECT, segmentation mask, etc.)
- **Target CT**: Select the reference CT image to which you want to register

#### Registration Options
Choose one of the following registration types:
- **Rigid**: Translation and rotation only
- **Affine**: Translation, rotation, scaling, and shearing
- **Deformable**: Non-linear registration (includes affine as initial step)
- **Resample Only**: Resample the image without registration
- **Existing Transformation**: Use a previously saved transformation file

#### Additional Options
- **Is Segmentation Mask**: Check this if Image X is a segmentation mask
- **Keep Original Image Spacing**: Maintain the original voxel spacing of the source image
- **Load Registered Output**: Automatically load the registered images into Slicer
- **Output Directory**: Choose where to save the registered images and transformation files

### Output Files

The module generates the following files in your chosen output directory:
- Registered CT image: `{registrationType}_registered_{sourceName}.nii.gz`
- Registered Image X: `{registrationType}_registered_{sourceImageName}.nii.gz`
- Transformation file: `{registrationType}_transform_{sourceName}@{targetName}.txt` (or `.nii.gz` for deformable registration)

## Tips

1. For segmentation masks, always check the "Is Segmentation Mask" option to use nearest-neighbor interpolation
2. Use rigid registration first, then try affine or deformable if needed
3. Save transformation files for future use to avoid re-registering
4. For best results, ensure your CT images are properly windowed before registration

## Requirements

- 3D Slicer 5.6.2 or later
- Python packages:
  - nibabel
  - niftyreg
  - numpy
  - scipy

## Author

Yizhou Chen (University of Bern, Switzerland)

## License

This module is subject to the same license terms as 3D Slicer.

## Support

For issues and feature requests, please use the GitHub issue tracker or contact the author. 