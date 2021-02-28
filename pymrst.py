def setup():
  """
  Setup PyMRST
  """
  import subprocess

  def apt_install(name):
    subprocess.call(['apt', 'install', name])
  def pip_install(name):
    subprocess.call(['pip', 'install', name])
  def clone(url):
    subprocess.call(['git', 'clone', url])
  
  # Install octave and cloning repositories
  apt_install('octave')
  pip_install('oct2py')
  clone('https://bitbucket.org/mrst/mrst-core.git')
  clone('https://bitbucket.org/mrst/mrst-autodiff.git')
  clone('https://github.com/yohanesnuwara/reservoir_datasets')
#   clone('https://github.com/yohanesnuwara/pyMRST')
  
def reshape_3d(array_1d, dimension):
  """
  Reshape 1D array of MRST cell properties result to 3D cube

  NOTE: This function is used after "get_output_cell_properties"

  INPUT:

  array_1d: 1D array result from running function "get_output_cell_properties"
  dimension: Cell dimension. EXAMPLE: dimension=(30,20,10)

  OUTPUT:

  array_3d: 3D cube of data
  """
  import numpy as np
  lenX, lenY, lenZ = dimension
  return np.reshape(array_1d, (lenZ, lenY, lenX)).T

def getCellData(directory, filename, dimension):
  """
  Get MRST cells output from MAT file in a DIRECTORY, convert to 3D array

  NOTE: 
  
  Cell data in MRST is data in 3D, e.g. pressure, saturation, porosity, etc.

  INPUT:

  directory: directory location of MAT file
  filename: MAT filename (1D array)

  OUTPUT:

  cube: cell data in 3D array
  """
  import numpy as np  

  # Get pressure 1D array
  array_1d = np.loadtxt(directory+"/"+filename, skiprows=5)

  # Reshape 1D array to 3D 
  cube = reshape_3d(array_1d, dimension=dimension) 

  return cube


def plotCellData(cube, plane, position, cmap="plasma", vmin=None, vmax=None):
  """
  Plot MRST cells output from a 3D cell data, in a 2D plane map

  NOTE: 
  
  Cell data in MRST is data in 3D, e.g. pressure, saturation, porosity, etc.

  INPUT:

  cube: The 3D data after transformation of MRST 1D array result using "reshape_3d"
  plane: The XZ, YZ, XY plane. Specify as "xz", "yz", or "xy"
  position: Index of plane (Starts from 0 to N-1)
  e.g. If the cube has shape (30,20,10), want to slice at XZ-surface, on 15th Y-slice
       RUN: slice_cube(cube, "xz", 15)

  OUTPUT:

  Plot of cell data in 2D 
  """

  import matplotlib.pyplot as plt

  lenX, lenY, lenZ = cube.shape # Cube shape
  if plane=="xz":
    slice2d = cube[:,position,:] # Constant Y for XZ plane
    plt.imshow(slice2d.T, extent=(0,lenX,lenZ,0), aspect="auto", cmap=cmap, 
               vmin=vmin, vmax=vmax)
    plt.xlabel("X [grid]"); plt.ylabel("Z [grid]")
    plt.colorbar()
  if plane=="yz":
    slice2d = cube[position,:,:] # Constant X for YZ plane
    plt.imshow(slice2d.T, extent=(0,lenY,lenZ,0), aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax)
    plt.xlabel("Y [grid]"); plt.ylabel("Z [grid]")
    plt.colorbar()
  if plane=="xy":
    slice2d = cube[:,:,position] # Constant Z for XY plane
    plt.imshow(slice2d, extent=(0,lenX,0,lenY), aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax) # No transpose
    plt.xlabel("X [grid]"); plt.ylabel("Y [grid]")
    plt.colorbar()  
