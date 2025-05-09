import logging
import os
from typing import Annotated, Optional
import vtk
import nibabel as nib
import time
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import copy
import numpy as np
from slicer import vtkMRMLScalarVolumeNode
import qt
import niftyreg
import shutil
from scipy.ndimage import zoom


# TODO
# - Add target modality Y for resampling
# - Add progress bar for registration
# - try multi-processing to avoid freezing the GUI


class NiftyRegistraion():
    """Remark: Check your Nifty files. nii_file.get_fdata().dtype should be the same in nii_file.header.get_data_dtype()"""
    @staticmethod
    def affine_register_nifti(ref_nifti:str, mov_nifti:str, output_nifti:str, output_transform_file:str, 
                            rigid_transform_only:bool=True, interp_mode:int=3, pad_value:int=-1024):
        args = ["aladin", "-ref", ref_nifti, "-flo", mov_nifti, "-res", output_nifti, "-aff", output_transform_file, "-interp", f"{interp_mode}", "-pad", f"{pad_value}", "-voff"] # "-voff"
        if rigid_transform_only:
            args.append("-rigOnly")
        # Tutorial: https://github.com/KCL-BMEIS/niftyreg/wiki/executable
        niftyreg.main(args)
        return output_nifti, output_transform_file
        
    @staticmethod
    def reg_resample_nifti(ref_nifti:str, mov_nifti:str, transform_file:str, 
                        output_nifti:str, interp_mode:int=3, pad_value:int=-1024):    
        args = ["resample", "-ref", ref_nifti, "-flo", mov_nifti, "-trans", transform_file, "-res", output_nifti, "-inter", f"{interp_mode}", "-pad", f"{pad_value}", "-voff"]
        niftyreg.main(args)
        return output_nifti    

    @staticmethod   
    def deformable_register_nifti(ref_nifti:str, mov_nifti:str, init_transform_file:str, 
                                output_nifti:str, output_transform_file:str, interp_mode:int=3, pad_value:int=-1024):
        args = ["f3d", "-ref", ref_nifti, "-flo", mov_nifti, "-res", output_nifti, "-cpp", output_transform_file, 
                "-aff", init_transform_file, "-interp", f"{interp_mode}", "-pad", f"{pad_value}", "-voff"] # "-voff"
        niftyreg.main(args)
        return output_nifti, output_transform_file
    
    @staticmethod
    def resample_3D_image_array(array_3d, current_voxel_size, target_voxel_size, order=3):
        # Calculate the new shape based on the new voxel size
        current_shape = array_3d.shape
        target_shape = np.round(
            np.array(current_shape) * np.array(current_voxel_size) / np.array(target_voxel_size)).astype(int)
        # Calculate the zoom factors for resampling
        zoom_factors = np.array(target_shape) / np.array(current_shape)
        # Resample the image data
        resampled_data = zoom(array_3d, zoom_factors, order=order)  # Using order parameter for interpolation
        resampled_data = np.clip(resampled_data, a_min=np.min(array_3d), a_max=np.max(array_3d))
        resampled_data = resampled_data.astype(array_3d.dtype)
        return resampled_data, zoom_factors

#
# DualModalityRegistration
#

# D:\Program Files\Slicer 5.6.2\bin> .\PythonSlicer.exe -m pip install tensorflow-intel -i https://pypi.tuna.tsinghua.edu.cn/simple

class DualModalityRegistration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("DualModalityRegistration")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Yizhou Chen (University of Bern)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
Dual modality registration module that registers an unknown modality image based on its CT image and the target CT image.""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
Developed by Yizhou Chen, University of Bern, Switzerland.""")


#
# DualModalityRegistrationParameterNode
#


@parameterNodeWrapper
class DualModalityRegistrationParameterNode:
    """
    The parameters needed by module.

    sourceCTVolume - The source CT volume for registration
    sourceImageVolume - The source image X that needs to be registered
    targetCTVolume - The target/reference CT volume
    outputCTVolume - The registered CT volume
    outputImageVolume - The registered image X volume
    registrationType - Type of registration (rigid, affine, or deformable)
    """

    sourceCTVolume: vtkMRMLScalarVolumeNode
    sourceImageVolume: vtkMRMLScalarVolumeNode
    targetCTVolume: vtkMRMLScalarVolumeNode
    outputCTVolume: vtkMRMLScalarVolumeNode
    outputImageVolume: vtkMRMLScalarVolumeNode
    registrationType: str = "rigid"  # Default to rigid registration
    textFilePath: str = ""


#
# DualModalityRegistrationWidget
#


class DualModalityRegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.interp_mode = 3  # Initialize interp_mode with default value

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Create logic class first
        self.logic = DualModalityRegistrationLogic()

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DualModalityRegistration.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.inputCTSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.targetCTSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputImageSelector.setMRMLScene(slicer.mrmlScene)

        # Set default output directory to module's output directory
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        self.ui.outputDirLineEdit.setText(os.path.join(moduleDir, "output"))

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.outputDirBrowseButton.connect("clicked(bool)", self.onOutputDirBrowseButton)
        
        # Node selector and file path connections
        self.setupNodeSelectorConnections()
        
        # Set up mutually exclusive checkboxes
        self.setupMutuallyExclusiveCheckboxes()

        # Connect the segmentation mask checkbox to update interp_mode
        self.ui.checkBoxSegmentationMask.toggled.connect(self.onSegmentationMaskToggled)

        # Connect the existing transformation checkbox to enable/disable the browse button
        self.ui.checkBoxExistingTransformation.toggled.connect(self.onExistingTransformationToggled)

        # Connect the browse button to open a file dialog for selecting .txt files
        self.ui.browseButton.connect("clicked(bool)", self.onBrowseTransformationFile)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.sourceCTVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.sourceCTVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[DualModalityRegistrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Enable or disable the apply button based on whether the input is valid."""
        if not self._parameterNode:
            return
            
        # Check if each input has either a node or a file path
        sourceCTValid = bool(self._parameterNode.sourceCTVolume or self.ui.inputCTFilePathLineEdit.text)
        sourceImageValid = bool(self._parameterNode.sourceImageVolume or self.ui.inputImageFilePathLineEdit.text)
        targetCTValid = bool(self._parameterNode.targetCTVolume or self.ui.targetCTFilePathLineEdit.text)
        
        if sourceCTValid and sourceImageValid and targetCTValid:
            self.ui.applyButton.toolTip = _("Run registration")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volumes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Create output volumes
            
            # Compute output
            self.logic.process(
                self._parameterNode.sourceCTVolume,
                self._parameterNode.sourceImageVolume,
                self._parameterNode.targetCTVolume,
                self._parameterNode.outputCTVolume,
                self._parameterNode.outputImageVolume,
                self._parameterNode.registrationType,
                sourceCTPath=self.ui.inputCTFilePathLineEdit.text,
                sourceImagePath=self.ui.inputImageFilePathLineEdit.text,
                targetCTPath=self.ui.targetCTFilePathLineEdit.text,
                interp_mode=self.interp_mode,
                loadRegisteredOutput=self.ui.checkBoxLoadRegisteredOutput.isChecked(),
                existingTransformPath=self.ui.textFilePathLineEdit.text,
                outputDir=self.ui.outputDirLineEdit.text,
                keep_original_image_spacing=self.ui.checkBoxKeepImageSpacing.isChecked()
            )

    def setupMutuallyExclusiveCheckboxes(self):
        """Set up the registration type checkboxes to be mutually exclusive."""
        # Store checkboxes in a list for easier management
        self.registrationCheckboxes = [
            self.ui.checkBoxRigidReg,
            self.ui.checkBoxAffineReg,
            self.ui.checkBoxDeformableReg,
            self.ui.checkBoxResamplingOnly,
            self.ui.checkBoxExistingTransformation
        ]
        
        # Connect each checkbox to the handler using toggled signal
        for checkbox in self.registrationCheckboxes:
            checkbox.toggled.connect(lambda checked, cb=checkbox: self.onRegistrationCheckboxToggled(checked, cb))
        
        # Set initial state - Rigid registration checked by default
        self.ui.checkBoxRigidReg.setChecked(True)
        if self._parameterNode:
            self._parameterNode.registrationType = "rigid"

    def onRegistrationCheckboxToggled(self, checked, checkbox):
        """Handle checkbox toggling to ensure only one is checked at a time."""
        if not self._parameterNode:
            return
            
        # If this checkbox is being checked
        if checked:
            # Uncheck all other checkboxes
            for cb in self.registrationCheckboxes:
                if cb != checkbox:
                    cb.setChecked(False)
            
            # Update the registration type in the parameter node
            if checkbox == self.ui.checkBoxRigidReg:
                self._parameterNode.registrationType = "rigid"
            elif checkbox == self.ui.checkBoxAffineReg:
                self._parameterNode.registrationType = "affine"
            elif checkbox == self.ui.checkBoxDeformableReg:
                self._parameterNode.registrationType = "deformable"
            elif checkbox == self.ui.checkBoxResamplingOnly:
                self._parameterNode.registrationType = "resampling"
            elif checkbox == self.ui.checkBoxExistingTransformation:
                self._parameterNode.registrationType = "existing"
            
        # If this checkbox is being unchecked
        else:
            # Count how many checkboxes are still checked
            checkedCount = sum(1 for cb in self.registrationCheckboxes if cb.isChecked())
            # If this was the last checked checkbox, prevent it from being unchecked
            if checkedCount == 0:
                checkbox.setChecked(True)

    def onSelect(self) -> None:
        """This method is called when input/output is changed."""
        self._checkCanApply()

    def setupNodeSelectorConnections(self):
        """Set up connections for node selectors and file paths."""
        # Source CT connections
        self.ui.inputCTSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda: self.onNodeSelectorChanged("sourceCT"))
        self.ui.inputCTBrowseButton.connect("clicked(bool)", lambda: self.onBrowseButton("sourceCT"))
        
        # Source Image connections
        self.ui.inputImageSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda: self.onNodeSelectorChanged("sourceImage"))
        self.ui.inputImageBrowseButton.connect("clicked(bool)", lambda: self.onBrowseButton("sourceImage"))
        
        # Target CT connections
        self.ui.targetCTSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda: self.onNodeSelectorChanged("targetCT"))
        self.ui.targetCTBrowseButton.connect("clicked(bool)", lambda: self.onBrowseButton("targetCT"))

    def onNodeSelectorChanged(self, selectorType):
        """Handle node selector changes to maintain mutual exclusivity."""
        if not self._parameterNode:
            return
            
        # Clear corresponding file path when node is selected
        if selectorType == "sourceCT":
            node = self.ui.inputCTSelector.currentNode()
            if node:
                filePath = self.getNodeFilePath(node)
                self.ui.inputCTFilePathLineEdit.setText(filePath)
        elif selectorType == "sourceImage":
            node = self.ui.inputImageSelector.currentNode()
            if node:
                filePath = self.getNodeFilePath(node)
                self.ui.inputImageFilePathLineEdit.setText(filePath)
        elif selectorType == "targetCT":
            node = self.ui.targetCTSelector.currentNode()
            if node:
                filePath = self.getNodeFilePath(node)
                self.ui.targetCTFilePathLineEdit.setText(filePath)
                
        self._checkCanApply()

    def getNodeFilePath(self, node):
        """Get the file path for a node, generating a temporary NIfTI file name if needed."""
        storageNode = node.GetStorageNode()
        if storageNode and storageNode.GetFileName():
            return storageNode.GetFileName()
        else:
            # Generate a temporary NIfTI file name based on the DICOM folder name
            dicomDir = node.GetAttribute("DICOM.instanceUIDs")
            if dicomDir:
                folderName = os.path.basename(dicomDir)
                return os.path.join(self.logic.tempDir, f"{folderName}.nii.gz")
        return ""

    def onBrowseButton(self, selectorType):
        """Handle browse button clicks for file selection."""
        if not self._parameterNode:
            slicer.util.errorDisplay("Parameter node is not initialized.")
            return

        # Create and configure file dialog
        fileDialog = qt.QFileDialog()
        fileDialog.setWindowTitle(f"Select {selectorType} File")
        fileDialog.setNameFilter("NIfTI Files (*.nii.gz *.nii)")  # Restrict to .nii.gz and .nii files
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)

        if fileDialog.exec_():
            filePath = fileDialog.selectedFiles()[0]
            # accept only NIfTI files
            if not filePath.endswith(".nii.gz") and not filePath.endswith(".nii"):
                # show warning message
                slicer.util.warningDisplay("Please select a NIfTI file.")
            if filePath:
                # Clear corresponding node selector and update line edit
                if selectorType == "sourceCT":
                    self.ui.inputCTSelector.setCurrentNode(None)
                    self.ui.inputCTFilePathLineEdit.setText(filePath)
                elif selectorType == "sourceImage":
                    self.ui.inputImageSelector.setCurrentNode(None)
                    self.ui.inputImageFilePathLineEdit.setText(filePath)
                elif selectorType == "targetCT":
                    self.ui.targetCTSelector.setCurrentNode(None)
                    self.ui.targetCTFilePathLineEdit.setText(filePath)
                
                self._checkCanApply()

    def onSegmentationMaskToggled(self, checked):
        if checked:
            self.interp_mode = 0
        else:
            self.interp_mode = 3

    def onExistingTransformationToggled(self, checked):
        self.ui.browseButton.setEnabled(checked)

    def onBrowseTransformationFile(self):
        fileDialog = qt.QFileDialog()
        fileDialog.setWindowTitle("Select Transformation File")
        fileDialog.setNameFilter("Text Files (*.txt);;NIfTI Files (*.nii *.nii.gz)")
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)

        if fileDialog.exec_():
            filePath = fileDialog.selectedFiles()[0]
            if filePath:
                self.ui.textFilePathLineEdit.setText(filePath)

    def onOutputDirBrowseButton(self):
        """Handle browse button click for output directory selection."""
        dirDialog = qt.QFileDialog()
        dirDialog.setWindowTitle("Select Output Directory")
        dirDialog.setFileMode(qt.QFileDialog.Directory)
        dirDialog.setDirectory(self.ui.outputDirLineEdit.text())

        if dirDialog.exec_():
            dirPath = dirDialog.selectedFiles()[0]
            if dirPath:
                self.ui.outputDirLineEdit.setText(dirPath)


#
# DualModalityRegistrationLogic
#


class DualModalityRegistrationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.regis = NiftyRegistraion()
        self.tempDir = None

    def getParameterNode(self):
        return DualModalityRegistrationParameterNode(super().getParameterNode())
    
    
    def correct_nifti_dtype(self, nifti_path:str, dtype=None):
        _img_nii = nib.load(nifti_path)
        if dtype is None:
            dtype = _img_nii.get_fdata().dtype
        _img_nii.header.set_data_dtype(dtype)
        return nib.Nifti1Image(_img_nii.get_fdata().astype(dtype), _img_nii.affine, _img_nii.header)

    def process(self,
                sourceCT: vtkMRMLScalarVolumeNode,
                sourceImage: vtkMRMLScalarVolumeNode,
                targetCT: vtkMRMLScalarVolumeNode,
                outputCT: vtkMRMLScalarVolumeNode,
                outputImage: vtkMRMLScalarVolumeNode,
                registrationType: str = "rigid",
                sourceCTPath: str = "",
                sourceImagePath: str = "",
                targetCTPath: str = "",
                interp_mode: int = 3,
                loadRegisteredOutput: bool = True,
                existingTransformPath: str = "",
                outputDir: str = "",
                keep_original_image_spacing: bool = False) -> None:
        """
        Run the registration process.
        :param sourceCT: source CT volume for registration
        :param sourceImage: source image X that needs to be registered
        :param targetCT: target/reference CT volume
        :param outputCT: output registered CT volume
        :param outputImage: output registered image X volume
        :param registrationType: type of registration (rigid, affine, or deformable)
        :param sourceCTPath: path to source CT file
        :param sourceImagePath: path to source image file
        :param targetCTPath: path to target CT file
        :param interp_mode: interpolation mode for image resampling
        :param loadRegisteredOutput: whether to create new volumes with registered output
        :param existingTransformPath: path to existing transformation file
        :param outputDir: output directory for registered volumes
        """
        if not all([sourceCT or sourceCTPath, sourceImage or sourceImagePath, targetCT or targetCTPath]):
            raise ValueError("Input or output volumes are invalid")

        startTime = time.time()
        logging.info("Processing started")

        # Use provided output directory or module's output directory
        
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        if not outputDir:
            outputDir = os.path.join(moduleDir, "output")
        os.makedirs(outputDir, exist_ok=True)
        
        temp_dir = os.path.join(moduleDir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
            
        # Save volumes to files if paths are not provided
        if not os.path.exists(sourceCTPath) and bool(sourceCT):
            slicer.util.saveNode(sourceCT, sourceCTPath)
        if not os.path.exists(sourceImagePath) and bool(sourceImage):
            slicer.util.saveNode(sourceImage, sourceImagePath)
        if not os.path.exists(targetCTPath) and bool(targetCT):
            slicer.util.saveNode(targetCT, targetCTPath)

        # CT dtype should be int16 !!! for the niftyreg registration
        _img_nii = self.correct_nifti_dtype(sourceCTPath, np.int16)
        sourceCTPath = os.path.join(temp_dir, f"{os.path.basename(sourceCTPath)}")
        nib.save(_img_nii, sourceCTPath)
        print("Saved source CT as int16. Path: ", sourceCTPath)
        __image_data = nib.load(sourceCTPath).get_fdata()
        source_CT_min = np.min(__image_data)
        source_CT_max = np.max(__image_data)

        # unify the dtype of the source image
        _img_nii = self.correct_nifti_dtype(sourceImagePath, dtype=None)
        sourceImagePath = os.path.join(temp_dir, f"{os.path.basename(sourceImagePath)}")
        nib.save(_img_nii, sourceImagePath)
        __image_data = nib.load(sourceImagePath).get_fdata()
        source_image_min = np.min(__image_data)
        source_image_max = np.max(__image_data)
        
        # CT dtype should be int16 !!! for the niftyreg registration
        _img_nii = self.correct_nifti_dtype(targetCTPath, np.int16)
        targetCTPath = os.path.join(temp_dir, f"{os.path.basename(targetCTPath)}")
        nib.save(_img_nii, targetCTPath)
        print("Saved target CT as int16. Path: ", targetCTPath)
        
        # Define output paths
        source_name = os.path.basename(sourceCTPath).split(".")[0]
        target_name = os.path.basename(targetCTPath).split(".")[0]
        source_image_name = os.path.basename(sourceImagePath).split(".")[0]
        
        outputCTPath = os.path.join(outputDir, f"{registrationType}_registered_{source_name}.nii.gz")
        outputImagePath = os.path.join(outputDir, f"{registrationType}_registered_{source_image_name}.nii.gz")
        transformPath = os.path.join(outputDir, f"{registrationType}_transform_{source_name}@{target_name}.txt")

        Image_interp_mode = interp_mode
        Image_pad_value = 0
        CT_interp_mode = 3
        CT_pad_value = -1024
        
        # Perform registration based on type
        if registrationType in ["rigid", "affine"]:
            isRigid = (registrationType == "rigid")
            self.regis.affine_register_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceCTPath,
                output_nifti=outputCTPath,
                output_transform_file=transformPath,
                rigid_transform_only=isRigid,
                interp_mode=CT_interp_mode,
                pad_value=CT_pad_value
            )
            
            # Resample the source image using the computed transformation
            self.regis.reg_resample_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceImagePath,
                transform_file=transformPath,
                output_nifti=outputImagePath,
                interp_mode=Image_interp_mode,
                pad_value=Image_pad_value
            )
        
        elif registrationType == "deformable":
            self.regis.affine_register_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceCTPath,
                output_nifti=outputCTPath,
                output_transform_file=transformPath,
                rigid_transform_only=False,
                interp_mode=CT_interp_mode,
                pad_value=CT_pad_value
            )
            init_transformPath = copy.deepcopy(transformPath)
            transformPath = os.path.join(outputDir, f"{registrationType}_transform_{source_name}@{target_name}.nii.gz")
            self.regis.deformable_register_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceCTPath,
                init_transform_file=init_transformPath,
                output_nifti=outputCTPath,
                output_transform_file=transformPath,
                interp_mode=CT_interp_mode,
                pad_value=CT_pad_value
            )
            os.remove(init_transformPath)
            # Resample the source image using the computed transformation
            self.regis.reg_resample_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceImagePath,
                transform_file=transformPath,
                output_nifti=outputImagePath,
                interp_mode=Image_interp_mode,
                pad_value=Image_pad_value
            )
        
        elif registrationType == "resampling":
            self.regis.reg_resample_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceImagePath,
                transform_file=os.path.join(moduleDir, "identity_transform.txt"),
                output_nifti=outputImagePath,
                interp_mode=Image_interp_mode,
                pad_value=Image_pad_value
            )
        
        elif registrationType == "existing":
            self.regis.reg_resample_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceCTPath,
                transform_file=existingTransformPath,
                output_nifti=outputCTPath,
                interp_mode=CT_interp_mode,
                pad_value=CT_pad_value
            )
            self.regis.reg_resample_nifti(
                ref_nifti=targetCTPath,
                mov_nifti=sourceImagePath,
                transform_file=existingTransformPath,
                output_nifti=outputImagePath,
                interp_mode=Image_interp_mode,
                pad_value=Image_pad_value
            )
        
        else:
            raise ValueError(f"Unknown registration type: {registrationType}")

        _ct_nii = nib.load(outputCTPath)
        nib.save(nib.Nifti1Image(np.clip(_ct_nii.get_fdata(), source_CT_min, source_CT_max),
                                    _ct_nii.affine, _ct_nii.header), outputCTPath)
        
        # clip output image to non-negative values
        _img_nii = nib.load(outputImagePath)
        if keep_original_image_spacing:
            logging.info("Restoring original spacing to the registered image.")
            sourceImageSpacing = nib.load(sourceImagePath).header.get_zooms()[0:3]
            currentImageSpacing = _img_nii.header.get_zooms()[0:3]
            currentImageData = _img_nii.get_fdata()
            resampledImageData, zoom_factors = self.regis.resample_3D_image_array(currentImageData, currentImageSpacing, sourceImageSpacing, order=interp_mode)
            nib.save(nib.Nifti1Image(np.clip(resampledImageData, source_image_min, source_image_max),
                                     _img_nii.affine, _img_nii.header), outputImagePath)
        else:
            nib.save(nib.Nifti1Image(np.clip(_img_nii.get_fdata(), source_image_min, source_image_max),
                                    _img_nii.affine, _img_nii.header), outputImagePath)
        

        # If load registered output is checked, create new volumes with the same names
        if loadRegisteredOutput:
            outputCT = slicer.util.loadVolume(outputCTPath)
            outputImage = slicer.util.loadVolume(outputImagePath)


        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
        
        
        shutil.rmtree(temp_dir)  # Remove the temporary directory after processing


#
# DualModalityRegistrationTest
#


class DualModalityRegistrationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_DualModalityRegistration1()

    def test_DualModalityRegistration1(self):
        """Test the dual modality registration functionality."""
        self.delayDisplay("Starting the test")

        # Load test data
        
        # # Modalities: CT and PET
        # sourceCTPath = "E:/PBPK-Adapted/data/PSMA-PETCT-LowResolution/registered_CT_nifti/G1_01_CT.nii.gz"  # Path to source CT test data
        # sourceImagePath = "E:/PBPK-Adapted/data/PSMA-PETCT-LowResolution/resampled_PET_nifti/G1_01_PET.nii.gz"  # Path to source image test data
        # targetCTPath = "E:/PBPK-Adapted/data/SPECT-CT-LowResolution/registered_CT_nifti/G1_01_Cycle_2021-03-17_ANON07311704924420210317_CT.nii.gz"
        # interp_mode = 3
        
        # # Modalities: CT and DoseMap
        # sourceCTPath = "E:/PBPK-Adapted/data/SPECT-CT-LowResolution/registered_CT_nifti/G1_01_Cycle_2021-03-17_ANON07311704924420210317_CT.nii.gz"
        # sourceImagePath = "E:/PBPK-Adapted/data/DoseMap-LowResolution/registered_MTPDoseMap_nifti/G1_01_Cycle_2021-03-17_DoseMap.nii.gz"  
        # targetCTPath = "E:/PBPK-Adapted/data/PSMA-PETCT-LowResolution/registered_CT_nifti/G1_01_CT.nii.gz"
        # interp_mode = 3
        
        
        # # Modalities: CT and SPECT
        # sourceCTPath = "E:/PBPK-Adapted/data/SPECT-CT-LowResolution/registered_CT_nifti/G1_01_Cycle_2021-03-17_ANON07311704924420210317_CT.nii.gz"
        # sourceImagePath = "E:/PBPK-Adapted/data/SPECT-CT/sr_SPECT_nifti/G1_01_Cycle_2021-03-17_ANON07311704924420210317_SPECT.nii.gz"  
        # targetCTPath = "E:/PBPK-Adapted/data/PSMA-PETCT-LowResolution/registered_CT_nifti/G1_01_CT.nii.gz"
        # interp_mode = 3
        
        
        # Modalities: CT and non-negative Segmentation mask
        sourceCTPath = "E:/PBPK-Adapted/data/PSMA-PETCT-LowResolution/registered_CT_nifti/G1_01_CT.nii.gz"  
        sourceImagePath = "E:/PBPK-Adapted/data/PSMA-PETCT/registered_TumorSeg_nifti/G1_01_TumorSeg.nii.gz"
        targetCTPath = "E:/PBPK-Adapted/data/SPECT-CT-LowResolution/registered_CT_nifti/G1_01_Cycle_2021-03-17_ANON07311704924420210317_CT.nii.gz"
        interp_mode = 0
        
        
        # Test the module logic
        logic = DualModalityRegistrationLogic()

        # Set up output directory for test
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        testOutputDir = os.path.join(moduleDir, "test_output")
        if not os.path.exists(testOutputDir):
            os.makedirs(testOutputDir)

        # Test rigid registration
        self.delayDisplay("Testing rigid registration")
        logic.process(
            sourceCT=None,
            sourceImage=None,
            targetCT=None,
            outputCT=None,
            outputImage=None,
            registrationType="rigid",
            sourceCTPath=sourceCTPath,
            sourceImagePath=sourceImagePath,
            targetCTPath=targetCTPath,
            interp_mode=interp_mode,
            loadRegisteredOutput=True,
            existingTransformPath="",
            outputDir=testOutputDir,
            keep_original_image_spacing=True
        )
        
        # Test affine registration
        self.delayDisplay("Testing affine registration")
        logic.process(
            sourceCT=None,
            sourceImage=None,
            targetCT=None,
            outputCT=None,
            outputImage=None,
            registrationType="affine",
            sourceCTPath=sourceCTPath,
            sourceImagePath=sourceImagePath,
            targetCTPath=targetCTPath,
            interp_mode=interp_mode,
            loadRegisteredOutput=True,
            existingTransformPath="",
            outputDir=testOutputDir,
            keep_original_image_spacing=True
        )
        
        # Test deformable registration
        self.delayDisplay("Testing deformable registration")
        logic.process(
            sourceCT=None,
            sourceImage=None,
            targetCT=None,
            outputCT=None,
            outputImage=None,
            registrationType="deformable",
            sourceCTPath=sourceCTPath,
            sourceImagePath=sourceImagePath,
            targetCTPath=targetCTPath,
            interp_mode=interp_mode,
            loadRegisteredOutput=True,
            existingTransformPath="",
            outputDir=testOutputDir,
            keep_original_image_spacing=True
        )

        # TODO: Check other registration types

        self.delayDisplay("Test completed")
