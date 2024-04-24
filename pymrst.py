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
  pip_install('bayesian-optimization')
  clone('https://bitbucket.org/mrst/mrst-core.git')
  clone('https://bitbucket.org/mrst/mrst-autodiff.git')
  clone('https://github.com/yohanesnuwara/reservoir_datasets')
#   clone('https://github.com/yohanesnuwara/pyMRST')

def write_input(nx, ny, nz, lx, ly, lz, poro, k, rock, fluid, well, 
                bc_front, bc_back, bc_left, bc_right, 
                numSteps=None, totTime=None, steps=None, 
                save_fluid_data=False):
  """
  Convert inputs given in Python to write a MATLAB program that executes
  reservoir geometry, rock property, fluid, boundary condition creation 
  """
  input = "addpath /content/pyMRST \n"
  input += "addpaths\n\n"

  # # Timestep
  # input += "# Timestep\n"
  # input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
  # input += "steps = {}; \n\n".format(steps)

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
  if poro["type"]=="homogeneous":
    input += "p = {};\n\n".format(poro["value"])

  # Permeability 
  input += "# Permeability \n"
  if k["type"]=="heterogeneous":
    if k["field"]=="kozeny":
      input += "K = p.^3.*(1e-5)^2./(0.81*72*(1-p).^2); \n\n"
  if k["type"]=="homogeneous":
    input += "K = {};\n\n".format(k["value"])

  # Make rock
  input += "# Make rock\n"
  input +=  "rock = makeRock(G, K(:), p(:)); \n\n"

  # Rock PV calculation
  input += "# Rock PV \n"
  if fluid["type"]=="oil" or fluid["type"]=="gas": # Compressible simulations
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
      ## Timestep
      input += "# Timestep\n"
      input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
      input += "steps = {}; \n\n".format(steps)      

    if fluid["type"]=="gas":
      input += "mu0 = {}; \n".format(fluid["mu0"])
      input += "c = {}; \n".format(fluid["c"])      
      input += "rho_r = {}; \n".format(fluid["rho_r"])
      input += "rhoS = {}; \n".format(fluid["rhoS"])      
      input += "c_mu = {}; \n".format(fluid["c_mu"])            
      input += "mu = @(p) mu0*(1+c_mu*(p-p_r)); \n"
      input += "rho = @(p) rho_r .* exp( c * (p - p_r) );\n\n"
      ## Timestep
      input += "# Timestep\n"
      input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
      input += "steps = {}; \n\n".format(steps)  

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
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {}, 'Dir', 'y');\n\n".format(well_number, well_type, well_value, well_radius)
      elif well_direction=="x":
        # Well is horizontal to x
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {}, 'Dir', 'x');\n\n".format(well_number, well_type, well_value, well_radius)      
      else:
        # Well is vertical
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {});\n\n".format(well_number, well_type, well_value, well_radius)

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

# def write_input(nx, ny, nz, lx, ly, lz, poro, k, rock, fluid, well, 
#                 bc_front, bc_back, bc_left, bc_right):
#   """
#   Convert inputs given in Python to write a MATLAB program that executes
#   reservoir geometry, rock property, fluid, boundary condition creation 
#   """
#   input = "addpath /content/pyMRST \n"
#   input += "addpaths\n\n"

#   # Timestep
#   input += "# Timestep\n"
#   input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
#   input += "steps = {}; \n\n".format(steps)

#   # Reservoir geometry
#   input += "# Reservoir geometry\n"
#   input += "[nx,ny,nz] = deal({}, {}, {});\n".format(nx, ny, nz)
#   input += "[lx,ly,lz] = deal({}, {}, {});\n".format(lx, ly, lz)
#   input += "G = cartGrid([nx ny nz],[lx ly lz]);\n"
#   input += "G = computeGeometry(G);\n\n"

#   # Porosity 
#   input += "# Porosity \n"
#   if poro["type"]=="heterogeneous":
#     if poro["field"]=="gaussian":
#       input += "p = gaussianField(G.cartDims, [{} {}], [5 3 1], {});\n\n".format(poro["min"], poro["max"], poro["std"])

#   # Permeability 
#   input += "# Permeability \n"
#   if k["type"]=="heterogeneous":
#     if k["field"]=="kozeny":
#       input += "K = p.^3.*(1e-5)^2./(0.81*72*(1-p).^2); \n\n"

#   # Make rock
#   input += "# Make rock\n"
#   input +=  "rock = makeRock(G, K(:), p(:)); \n\n"

#   # Rock PV calculation
#   input += "# Rock PV \n"
#   if fluid["type"]=="oil": # Slightly compressible simulation
#     input += "cr = {}; \n".format(rock["c"])
#     input += "p_r = {}; \n".format(rock["p_r"])
#     input += "pv_r = poreVolume(G, rock); \n"
#     input += "pv = @(p) pv_r .* exp( cr * (p - p_r) ); \n\n" 

#   # Boundary conditions
#   input += "# Boundary conditions \n"
#   input += "bc = [];\n"

#   bc_loc = ["'FRONT'", "'BACK'", "'LEFT'", "'RIGHT'"]
#   bc_type = [bc_front["type"], bc_back["type"], bc_left["type"], bc_right["type"]]
#   bc_val = [bc_front["value"], bc_back["value"], bc_left["value"], bc_right["value"]]

#   for i in range(len(bc_type)):
#     if i==0:
#       # First boundary condition, bc=[]
#       if bc_type[i]=="fluxside":
#         input += "bc = fluxside([], G, {}, {});\n".format(bc_loc[i], bc_val[i])
#       if bc_type[i]=="pside":
#         input += "bc = pside([], G, {}, {});\n".format(bc_loc[i], bc_val[i])
#     if i>0:
#       if bc_type[i]=="fluxside":
#         input += "bc = fluxside(bc, G, {}, {});\n".format(bc_loc[i], bc_val[i])
#       if bc_type[i]=="pside":
#         input += "bc = pside(bc, G, {}, {});\n".format(bc_loc[i], bc_val[i])      
#   input += "\n"

#   # Fluid
#   # If a string, so single-phase (numphase=1)
#   if type(fluid["type"])==str:
#     input += "# Fluid is {}\n".format(fluid["type"])
#     # Single phase
#     if fluid["type"]=="water":
#       input += "fluid     = initSingleFluid('mu', {}, 'rho', {});\n\n".format(fluid["mu"], fluid["rho"])
#     if fluid["type"]=="oil":
#       input += "mu = {}; \n".format(fluid["mu"])
#       input += "c = {}; \n".format(fluid["c"])
#       input += "rho_r = {}; \n".format(fluid["rho_r"])
#       input += "rhoS = {}; \n".format(fluid["rhoS"])
#       input += "rho = @(p) rho_r .* exp( c * (p - p_r) );\n\n"

#     if fluid["type"]=="gas":
#       input += "mu = @(p) {}*(1+{}*(p-{})); \n".format(fluid["mu0"], fluid["c_mu"], fluid["p_r"])
#       input += "@(p) {} .* exp( {} * (p - {}) );\n\n".format(fluid["rho_r"], fluid["c"], fluid["p_r"])

#   else:
#     if len(fluid["type"])==2:
#       # Two-phase
#       print("no")

#   # Well
#   input += "# Well\n"
#   if type(fluid["type"])==str:
#     # Single-phase. Well doesn't have phase
#     for i in range(len(well["type"])):
#       well_loc = well["cell_loc"][i]
#       well_type = well["type"][i]
#       well_value = well["value"][i]
#       well_radius = well["radius"][i]
#       well_skin = well["skin"][i]
#       well_direction = well["direction"][i]

#       # Well locations convert to list to avoid breaking into new line
#       input += "well_loc = {};".format(list(well_loc))  
#       input += "\n"

#       if well_type=="bhp":
#         input += "pwf = {}; \n".format(well_value)      

#       if i==0:
#         # First well
#         well_number = "[]"
#       if i>0:
#         # The next wells
#         well_number = "W"

#       if well_direction=="y":
#         # Well is horizontal to y
#         input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {}, 'Dir', 'y');\n\n".format(well_number, well_type, well_value, well_radius)
#       elif well_direction=="x":
#         # Well is horizontal to x
#         input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {}, 'Dir', 'x');\n\n".format(well_number, well_type, well_value, well_radius)      
#       else:
#         # Well is vertical
#         input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'InnerProduct', 'ip_tpf', 'Val', {}, 'Radius', {});\n\n".format(well_number, well_type, well_value, well_radius)

#   else:
#     if len(fluid["type"])==2:
#       # Two-phase. Well have phase
#       a=2  

#   # print(input)

#   # write file instead of %%writefile
#   # input_file = open("/content/INPUT.m", "w")
#   input_file = open("/content/pyMRST/INPUT.m", "w")
#   input_file.write(input)
#   input_file.close()  

def model_input(model, fluid, well, 
                bc_front, bc_back, bc_left, bc_right, 
                numSteps=None, totTime=None, steps=None, save_data=False):
  """
  Convert inputs given in Python to write a MATLAB program that executes
  reservoir geometry, rock property, fluid, boundary condition creation 
  """
  # saving data if true
  if save_data:
    directory = "/content/pyMRST/input_data"
    os.mkdir(directory)
                  
  input = "addpath /content/pyMRST \n"
  input += "addpaths\n\n"

  # Check if a model is used. If not, use the specified inputs.
  model_name = model["name"]
  if model_name=="spe10":
    # SPE10 is used
    input += "# SPE10 model \n"
    input += "[startlayer,nlayer] = deal({}, {}); \n".format(model["startlayer"], model["nlayer"])
    input += "dims = [60 220 nlayer]; \n" 
    input += "domain = dims.*[20 10 2]*ft; \n"
    input += "G = computeGeometry(cartGrid(dims,domain)); \n\n"

    input += "# Make rock \n"
    input += "rock = getSPE10rock((1:dims(1)), (1:dims(2)), startlayer:startlayer+(nlayer-1)); \n"            
    input += "rock.poro = max(rock.poro,.0005); \n\n"

  # # Rock PV calculation
  # input += "# Rock PV \n"


  # if fluid["type"]=="oil" or fluid["type"]=="gas": # Compressible simulations
  #   input += "cr = {}; \n".format(rock["c"])
  #   input += "p_r = {}; \n".format(rock["p_r"])
  #   input += "pv_r = poreVolume(G, rock); \n"
  #   input += "pv = @(p) pv_r .* exp( cr * (p - p_r) ); \n\n" 

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
      ## Timestep
      input += "# Timestep\n"
      input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
      input += "steps = {}; \n\n".format(steps)      

    if fluid["type"]=="gas":
      input += "mu0 = {}; \n".format(fluid["mu0"])
      input += "c = {}; \n".format(fluid["c"])      
      input += "rho_r = {}; \n".format(fluid["rho_r"])
      input += "rhoS = {}; \n".format(fluid["rhoS"])      
      input += "c_mu = {}; \n".format(fluid["c_mu"])            
      input += "mu = @(p) mu0*(1+c_mu*(p-p_r)); \n"
      input += "rho = @(p) rho_r .* exp( c * (p - p_r) );\n\n"
      ## Timestep
      input += "# Timestep\n"
      input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
      input += "steps = {}; \n\n".format(steps)  

  else:
    if len(fluid["type"])==2:
      # Two-phase
      input += "# Two-phase oil-water\n"
      input += "fluid = initSimpleFluid('mu', [{}, {}], 'rho', [{}, {}], 'n', [{}, {}]); \n\n".\
      format(fluid["mu"][0], fluid["mu"][1], fluid["rho"][0], fluid["rho"][1],
             fluid["n"][0], fluid["n"][1])

      if save_data:
        # Save fluid data as txt
        a=1
      
      input += "# Timestep \n"
      input += "[numSteps, totTime] = deal({}, {}*day); \n".format(numSteps, totTime)
      input += "steps = {}; \n\n".format(steps) 

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
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {}, 'Dir', 'y');\n\n".format(well_number, well_type, well_value, well_radius)
      elif well_direction=="x":
        # Well is horizontal to x
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {}, 'Dir', 'x');\n\n".format(well_number, well_type, well_value, well_radius)      
      else:
        # Well is vertical
        input += "W = addWell({}, G, rock, well_loc, 'Type', '{}', 'Val', {}, 'Radius', {});\n\n".format(well_number, well_type, well_value, well_radius)

  else:
    if len(fluid["type"])==2:
      # Two-phase. Well have 2 phase
      for i in range(len(well["type"])):    
        well_type = well["type"][i]
        well_value = well["value"][i]
        well_phase = well["phase"][i]
        well_radius = well["radius"][i]
        well_skin = well["skin"][i]
        well_direction = well["direction"][i]

        if well_direction==None:
          # For this 2-phase only vertical well is still implemented.
          well_locx = well["cellx_loc"][i]
          well_locy = well["celly_loc"][i]    
          well_locz = well["cellz_loc"][i]          
          # # Well locations convert to list to avoid breaking into new line
          # input += "well_loc = {};".format(list(well_loc))  
          # input += "\n"

          if i==0:
            # First well
            well_number = "[]"
          if i>0:
            # The next wells
            well_number = "W"

          input += "W = verticalWell({}, G, rock, {}, {}, {}, 'Type', '{}', 'Val', {}, 'Radius', {}, 'InnerProduct', 'ip_tpf', 'Comp_i', {});\n".\
          format(well_number, well_locx, well_locy, well_locz, well_type, well_value, well_radius, well_phase)

        else:
          print("Horizontal well for 2-phase is NOT YET implemented")


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

def getWellSol(directory, filename):
  """
  Plot well solutions output from MAT file, in a 2D plane map

  NOTE: 
  
  Well solutions in MRST is data such as flowing pressure, production rate, etc.  
  """
  import numpy as np  

  # Get pressure 1D array
  array = np.loadtxt(directory+"/"+filename, skiprows=5)

  return array

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
    plt.imshow(slice2d.T, extent=(0,lenX,lenY,0), aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax) # No transpose
    plt.xlabel("X [grid]"); plt.ylabel("Y [grid]")
    plt.colorbar()  

def water_1phase():
  """
  MRST Incompressible Water Simulation
  """
  import oct2py as op
  # Execute simulation program "water_1phase.m"
  # After executed, new .mat files (that contains PRESSURE, PORO, PERM result)
  # is created inside new directory "result_water_1phase"
  octave = op.Oct2Py()
  octave.run("/content/pyMRST/water_1phase.m")    

def oil_1phase():
  """
  MRST Slightly Compressible Oil Simulation (Constant Viscosity over Pressure)
  """
  import oct2py as op
  # Execute simulation program "oil_1phase.m"
  # After executed, new .mat files (that contains PRESSURE, PORO, PERM result)
  # is created inside new directory "result_oil_1phase"
  octave = op.Oct2Py()
  octave.run("/content/pyMRST/oil_1phase.m")  

def gas_1phase():
  """
  MRST Compressible Gas Simulation (Variable Viscosity over Pressure)
  """
  import oct2py as op
  # Execute simulation program "gas_1phase.m"
  # After executed, new .mat files (that contains PRESSURE, PORO, PERM result)
  # is created inside new directory "result_gas_1phase"
  octave = op.Oct2Py()
  octave.run("/content/pyMRST/gas_1phase.m")
  
def oilwater_2phase():
  """
  MRST Two-phase Oil-water Simulation
  """
  import oct2py as op
  # Execute simulation program "oilwater_2phase.m"
  # After executed, new .mat files is created inside new directory 
  # "result_oilwater_2phase"
  octave = op.Oct2Py()
  octave.run("/content/pyMRST/oilwater_2phase.m")

def oilwater_2phase2():
  """
  MRST Two-phase Oil-water Simulation
  """
  import oct2py as op
  # Execute simulation program "oilwater_2phase.m"
  # After executed, new .mat files is created inside new directory 
  # "result_oilwater_2phase"
  octave = op.Oct2Py()
  octave.eval("/content/pyMRST/oilwater_2phase.m", verbose=False)

def run_simulation():
  """
  MRST Two-phase Oil-water Simulation
  """
  import subprocess
  # Execute simulation program "oilwater_2phase.m"
  # After executed, new .mat files is created inside new directory 
  # "result_oilwater_2phase"
  subprocess.call(['octave', '-W', "/content/pyMRST/oilwater_2phase.m"])
  # octave = op.Oct2Py()
  # octave.eval("/content/pyMRST/oilwater_2phase.m", verbose=False)  

def optimize(f_objective, pbounds, init_points=5, n_iter=10):
  """
  Well placement optimization
  """
  from bayes_opt import BayesianOptimization
  optimizer = BayesianOptimization(
      f=f_objective,
      pbounds=pbounds,
      verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
      random_state=42,
  )
  optimizer.maximize(init_points=init_points, n_iter=n_iter)  
