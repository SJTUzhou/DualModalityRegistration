cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DualModalityRegistration)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DualModalityRegistration")
set(EXTENSION_CATEGORY "Example")
set(EXTENSION_CONTRIBUTORS "Yizhou Chen (University of Bern)")
set(EXTENSION_DESCRIPTION "PET/CT SPECT/CT Registration")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/DualModalityRegistration.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/DualModalityRegistration/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DualModalityRegistration)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
