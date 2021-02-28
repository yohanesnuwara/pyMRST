addpath /content/pyMRST
addpaths

# Execute input program. Already executed G, rock, bc, well, and fluid
INPUT

# Initialize solver                            
resSol = initResSol(G, 0.0);

# Compute transmissibility
T = computeTrans(G, rock, 'Verbose', true);

# Solve pressure using TPFA
resSol = incompTPFA(resSol, G, T, fluid, ...
                   'bc', bc, ...
                   'wells', W, ...
                   'MatrixOutput', true);

# Set directory to store results
directory = "/content/result_water_1phase";

x = any(size(dir([directory '/*.mat' ]),1));

if x==1
  # If there is file in directory, delete all contents
  rmdir(directory, "s");
endif
# Make new directory
mkdir(directory)

# Store results into directory
pressure = resSol.pressure; poro = rock.poro; perm = rock.perm;

save([directory,"/pressure.mat"], "pressure")
save([directory,"/poro.mat"], "poro")
save([directory,"/perm.mat"], "perm")
