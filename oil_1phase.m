%%writefile oil_1phase.m
addpath /content/pyMRST
addpaths

# Execute input program. Already executed G, rock, bc, well, and fluid
INPUT

# solve initial hydrostatic pressure
gravity reset on; 
g = norm(gravity);
[z_0, z_max] = deal(0, max(G.cells.centroids(:,3)));
equil = ode23(@(z,p) g .* rho(p), [z_0, z_max], p_r);
p_init = reshape(deval(equil, G.cells.centroids(:,3)), [], 1);

# Compute map between interior faces and cells
C = double(G.faces.neighbors);
intInx = all(C ~= 0, 2);
C = C(intInx, :);

# Define averaging operator
n = size(C,1);
D = sparse([(1:n)'; (1:n)'], C, ...
ones(n,1)*[-1 1], n, G.cells.num);
grad = @(x) D*x;
div = @(x) -D'*x;
avg = @(x) 0.5 * (x(C(:,1)) + x(C(:,2)));

# Compute transmissibilities
hT = computeTrans(G, rock); % Half-transmissibilities
cf = G.cells.faces(:,1);
nf = G.faces.num;
T = 1 ./ accumarray(cf, 1 ./ hT, [nf, 1]); % Harmonic average
T = T(intInx); % Restricted to interior

# Darcy's equation
gradz = grad(G.cells.centroids(:,3));
v = @(p) -(T/mu).*( grad(p) - g*avg(rho(p)).*gradz );

# Continuity equation for each cell C
presEq = @(p,p0,dt) (1/dt)*(pv(p).*rho(p) - pv(p0).*rho(p0)) ...
+ div( avg(rho(p)).*v(p) );

## Define well model
wc = W(1).cells; % connection grid cells
WI = W(1).WI; % well-indices
dz = W(1).dZ; % depth relative to bottom-hole
p_conn = @(bhp) bhp + g*dz.*rho(bhp); %connection pressures
q_conn = @(p, bhp) WI .* (rho(p(wc)) / mu) .* (p_conn(bhp) - p(wc));

# Compute total volumetric well rate
rateEq = @(p, bhp, qS) qS-sum(q_conn(p, bhp))/rhoS;

# Declare the well condition as constant BHP
ctrlEq = @(bhp) bhp-pwf;

## Initialize simulation loop

# Initialize AD variables
[p_ad, bhp_ad, qS_ad] = initVariablesADI(p_init, p_init(wc(1)), 0);

# Set indices
[p_ad, bhp_ad, qS_ad] = initVariablesADI(p_init, p_init(wc(1)), 0);
nc = G.cells.num;
[pIx, bhpIx, qSIx] = deal(1:nc, nc+1, nc+2);

# Set timesteps
[tol, maxits] = deal(1e-5, 10); % Newton tolerance / maximum Newton its
dt = totTime / numSteps;

sol = repmat(struct('time',[], 'pressure',[], 'bhp',[], 'qS',[]), [numSteps+1, 1]);
sol(1) = struct('time', 0, 'pressure', value(p_ad), ...
'bhp', value(bhp_ad), 'qS', value(qS_ad));

## Main simulation loop
t = 0; step = 0;
while t < totTime
   t = t + dt;
   step = step + 1;
   fprintf('\nTime step %d: Time %.2f -> %.2f days\n', ...
      step, convertTo(t - dt, day), convertTo(t, day));
   % Newton loop
   resNorm = 1e99;
   p0  = value(p_ad); % Previous step pressure
   nit = 0;
   while (resNorm > tol) && (nit <= maxits)
      % Add source terms to homogeneous pressure equation:
      eq1     = presEq(p_ad, p0, dt);
      eq1(wc) = eq1(wc) - q_conn(p_ad, bhp_ad);
      % Collect all equations
      eqs = {eq1, rateEq(p_ad, bhp_ad, qS_ad), ctrlEq(bhp_ad)};
      % Concatenate equations and solve for update:
      eq  = cat(eqs{:});
      J   = eq.jac{1};  % Jacobian
      res = eq.val;     % residual
      upd = -(J \ res); % Newton update
      % Update variables
      p_ad.val   = p_ad.val   + upd(pIx);
      bhp_ad.val = bhp_ad.val + upd(bhpIx);
      qS_ad.val  = qS_ad.val  + upd(qSIx);

      resNorm = norm(res);
      nit     = nit + 1;
      fprintf('  Iteration %3d:  Res = %.4e\n', nit, resNorm);
   end

   if nit > maxits
      error('Newton solves did not converge')
   else % store solution
      sol(step+1)  = struct('time', t, 'pressure', value(p_ad), ...
                            'bhp', value(bhp_ad), 'qS', value(qS_ad));
   end
end

# Set directory to store results
directory = "/content/result_oil_1phase";

x = any(size(dir([directory '/*.mat' ]),1));

if x==1
  # If there is file in directory, delete all contents
  rmdir(directory, "s");
endif
# Make new directory
mkdir(directory)

# Save pressure results into .mat files
for i=1:length(steps)
  step = steps(i);
  pressure = sol(step).pressure/barsa;
  # give number (step) to filename
  # filename = sprintf('/content/result_compressible_oil/pressure%d.mat', step); 
  filename = sprintf([directory,"/pressure%d.mat"], step); 
  save(filename, "pressure");
endfor

# Save well solutions into .mat files
time = [sol(2:end).time]/day; # day
qo = -[sol(2:end).qS]*day; # m3/day
Pwf = mean([sol(2:end).pressure]/barsa); # bar

save([directory,"/time.mat"], "time");
save([directory,"/qo.mat"], "qo");
save([directory,"/Pwf.mat"], "Pwf");
