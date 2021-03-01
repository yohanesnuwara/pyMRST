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

def write_input(nx, ny, nz, lx, ly, lz, poro, k, rock, fluid, well, 
                bc_front, bc_back, bc_left, bc_right):
  """
  Convert inputs given in Python to write a MATLAB program that executes
  reservoir geometry, rock property, fluid, boundary condition creation 
  """
  input = "addpath /content/pyMRST \n"
  input += "addpaths\n\n"

  # Timestep
  input += "# Timestep\n"
  input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
  input += "steps = {}; \n\n".format(steps)

  # Reservoir geometry
  input += "# Reservoir geometry\n"
  input += "[nx,ny,nz] = deal({}, {}, {});\n".format(nx, ny, nz)
  input += "[lx,ly,lz] = deal({}, {}, {});\n".format(lx, ly, lz)
  input += "G = cartGrid([nx ny nz],[lx ly lz]);\n"
  input += "G = computeGeometry(G);\n\n"

  # Porosity 
  input += "# Porosity \n"
  if poro["type"]=="heterogeneous":
    if poro["field"]=="gaussian":
      input += "p = gaussianField(G.cartDims, [{} {}], [5 3 1], {});\n\n".format(poro["min"], poro["max"], poro["std"])

  # Permeability 
  input += "# Permeability \n"
  if k["type"]=="heterogeneous":
    if k["field"]=="kozeny":
      input += "K = p.^3.*(1e-5)^2./(0.81*72*(1-p).^2); \n\n"

  # Make rock
  input += "# Make rock\n"
  input +=  "rock = makeRock(G, K(:), p(:)); \n\n"

  # Rock PV calculation
  input += "# Rock PV \n"
  if fluid["type"]=="oil": # Slightly compressible simulation
    input += "cr = {}; \n".format(rock["c"])
    input += "p_r = {}; \n".format(rock["p_r"])
    input += "pv_r = poreVolume(G, rock); \n"
    input += "pv = @(p) pv_r .* exp( cr * (p - p_r) ); \n\n" 

  # Boundary conditions
  input += "# Boundary conditions \n"
  input += "bc = [];\n"

  bc_loc = ["'FRONT'", "'BACK'", "'LEFT'", "'RIGHT'"]
  bc_type = [bc_front["type"], bc_back["type"], bc_left["type"], bc_right["type"]]
  bc_val = [bc_front["value"], bc_back["value"], bc_left["value"], bc_right["value"]]

  for i in range(len(bc_type)):
    if i==0:
      # First boundary condition, bc=[]
      if bc_type[i]=="fluxside":
        input += "bc = fluxside([], G, {}, {});\n".format(bc_loc[i], bc_val[i])
      if bc_type[i]=="pside":
        input += "bc = pside([], G, {}, {});\n".format(bc_loc[i], bc_val[i])
    if i>0:
      if bc_type[i]=="fluxside":
        input += "bc = fluxside(bc, G, {}, {});\n".format(bc_loc[i], bc_val[i])
      if bc_type[i]=="pside":
        input += "bc = pside(bc, G, {}, {});\n".format(bc_loc[i], bc_val[i])      
  input += "\n"

  # Fluid
  # If a string, so single-phase (numphase=1)
  if type(fluid["type"])==str:
    input += "# Fluid is {}\n".format(fluid["type"])
    # Single phase
    if fluid["type"]=="water":
      input += "fluid     = initSingleFluid('mu', {}, 'rho', {});\n\n".format(fluid["mu"], fluid["rho"])
    if fluid["type"]=="oil":
      input += "mu = {}; \n".format(fluid["mu"])
      input += "c = {}; \n".format(fluid["c"])
      input += "rho_r = {}; \n".format(fluid["rho_r"])
      input += "rhoS = {}; \n".format(fluid["rhoS"])
      input += "rho = @(p) rho_r .* exp( c * (p - p_r) );\n\n"

    if fluid["type"]=="gas":
      input += "mu = @(p) {}*(1+{}*(p-{})); \n".format(fluid["mu0"], fluid["c_mu"], fluid["p_r"])
      input += "@(p) {} .* exp( {} * (p - {}) );\n\n".format(fluid["rho_r"], fluid["c"], fluid["p_r"])

  else:
    if len(fluid["type"])==2:
      # Two-phase
      print("no")

  # Well
  input += "# Well\n"
  if type(fluid["type"])==str:
    # Single-phase. Well doesn't have phase
    for i in range(len(well["type"])):
      well_loc = well["cell_loc"][i]
      well_type = well["type"][i]
      well_value = well["value"][i]
      well_radius = well["radius"][i]
      well_skin = well["skin"][i]
      well_direction = well["direction"][i]

      # Well locations convert to list to avoid breaking into new line
      input += "well_loc = {};".format(list(well_loc))  
      input += "\n"

      if well_type=="bhp":
        input += "pwf = {}; \n".format(well_value)      

      if i==0:
        # First well
        well_number = "[]"
      if i>0:
        # The next wells
        well_number = "W"

      if well_direction=="y":
        # Well is horizontal to y
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {}, 'Dir', 'y');\n\n".format(well_number, well_type, well_value, well_radius)
      elif well_direction=="x":
        # Well is horizontal to x
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {}, 'Dir', 'x');\n\n".format(well_number, well_type, well_value, well_radius)      
      else:
        # Well is vertical
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {});\n\n".format(well_number, well_type, well_value, well_radius)

  else:
    if len(fluid["type"])==2:
      # Two-phase. Well have phase
      a=2  

  # print(input)

  # write file instead of %%writefile
  # input_file = open("/content/INPUT.m", "w")
  input_file = open("/content/pyMRST/INPUT.m", "w")
  input_file.write(input)
  input_file.close()  
  
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
    plt.imshow(slice2d.T, extent=(0,lenX,0,lenY), aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax) # No transpose
    plt.xlabel("X [grid]"); plt.ylabel("Y [grid]")
    plt.colorbar()  

def water_1phase(nx, ny, nz, lx, ly, lz, poro, k, fluid, well, 
                         bc_front, bc_back, bc_left, bc_right):
  """
  MRST Incompressible Water Simulation
  """
  import oct2py as op
  # Execute writing MATLAB input program from given inputs
  # After executed, new .m file is created: INPUT.m
  write_input(nx, ny, nz, lx, ly, lz, poro, k, rock, fluid, well, 
              bc_front, bc_back, bc_left, bc_right)
  
  # Execute simulation program "water_1phase.m"
  # After executed, new .mat files (that contains PRESSURE, PORO, PERM result)
  # is created inside new directory "result_water_1phase"
  # !octave -W /content/pyMRST/water_1phase.m
  octave = op.Oct2Py()
  octave.run("/content/pyMRST/water_1phase.m")    