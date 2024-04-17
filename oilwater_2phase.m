addpath /content/pyMRST
addpaths

# Execute input program. Already executed G, rock, bc, well, and fluid
INPUT

# Transmissibility
hT = computeTrans(G, rock); % compute transmissibility 

# Initialize and compute OOIP
x0 = initState(G,W,100*barsa, [0 1]); % initialize
x0 = incompTPFA(x0, G, hT, fluid, 'wells', W);
pv = poreVolume(G, rock);
ooip = sum(x0.s(:,2).*pv);

# Simulation
[M,T] = deal(numSteps, totTime);
[dt,dT] = deal(zeros(M,1), T/M);
timestep = linspace(1,T/year,M);

x = x0;
wellSol = cell(M,3);
oip     = zeros(M,3);
n = 1;

# Prepare matrix to store reservoir pressure and saturation results 
sol = repmat(struct('time',[], 'pressure',[], 'Sw',[]), [numSteps, 1]);
# sol(1) = struct('time', 0, 'pressure', value(p_ad), 'bhp', value(bhp_ad));

for i=1:M
    fprintf('Time step %d: %.2f years \n', i, timestep(i));
    x  = incompTPFA(x, G, hT, fluid, 'wells', W);
    # x  = incompTPFA(x, G, hT, fluid, 'bc', bc, 'wells', W);    
    x  = implicitTransport(x, G, dT, rock, fluid, 'wells', W);  
    # x  = implicitTransport(x, G, dT, rock, fluid, 'bc', bc, 'wells', W);        
        
    dt(i) = dT;
    oip(i,n) = sum(x.s(:,2).*pv);
    wellSol{i,n} = getWellSol(W,x, fluid);

    # Store results
    pressure = x.pressure;
    Sw = x.s(:,1);
    sol(i)  = struct('time', timestep(i), 'pressure', pressure, 'Sw', Sw);
end

# # Compute F-Phi
# D       = computeTOFandTracer(x, G, rock, 'wells', W, 'maxTOF', inf);
# [F,Phi] = computeFandPhi(pv, D.tof);

# Print the cum. oil production as OOIP minus OIP
Np = ooip - oip(numSteps);
display(Np);

# Set directory to store results
directory = "/content/result_oilwater_2phase";

x = any(size(dir([directory '/*.mat' ]),1));

if x==1
  # If there is file in directory, delete all contents
  rmdir(directory, "s");
endif
# Make new directory
mkdir(directory)

# Save oil prod
fid = fopen ("/content/result_oilwater_2phase/Np.txt", "w");
fdisp (fid, Np);
fclose (fid);

# Save pressure results into .mat files
for i=1:length(steps)
  step = steps(i);
  pressure = sol(step).pressure;
  Sw = sol(step).Sw;

  # give number (step) to filename 
  filename = sprintf([directory,"/pressure%d.mat"], step); 
  save(filename, "pressure");

  # give number (step) to filename 
  filename = sprintf([directory,"/Sw%d.mat"], step); 
  save(filename, "Sw");

# Well solutions
wellSols = wellSol(:,1);

for i = 1:numel(W)-1
  Sw = cellfun(@(x) abs(x(i).Sw), wellSols);
  Wc = cellfun(@(x) abs(x(i).wcut), wellSols); # water cut  
  qOs = cellfun(@(x) abs(x(i).qOs), wellSols); # oil rate
  qWs = cellfun(@(x) abs(x(i).qWs), wellSols); # water rate

  filename = sprintf([directory,"/well%d_Sw.mat"], i);
  save(filename, "Sw");

  filename = sprintf([directory,"/well%d_Wc.mat"], i);
  save(filename, "Wc"); 

  filename = sprintf([directory,"/well%d_qOs.mat"], i);
  save(filename, "qOs"); 

  filename = sprintf([directory,"/well%d_qWs.mat"], i);
  save(filename, "qWs");      
end

endfor
