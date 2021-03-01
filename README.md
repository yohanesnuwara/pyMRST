# pyMRST

Python wrappers to run simple reservoir simulations from MATLAB Reservoir Simulation Toolbox (MRST)

<div>
<img src="https://user-images.githubusercontent.com/51282928/109408566-7d209800-79bd-11eb-89b3-294343680217.png" width="500"/>
</div>

MATLAB Reservoir Simulation Toolbox (MRST) is a free open-source software for reservoir modelling and simulation, developed primarily by the Computational Geosciences group in the Department of Mathematics and Cybernetics at SINTEF Digital.

While pyMRST is a wrapper that allows users to run simulations from MRST in a Python environment. Currently, it has single-phase reservoir simulations for water, oil, gas, polymer, and thermal effect; and two-phase oil-water.

|Simulations|Python notebook|
|:--:|:--:|
|1-phase water|[Notebook]()|
|1-phase oil *)|[Notebook]()|
|1-phase gas *)|[Notebook]()|
|1-phase polymer *)|[Notebook]()|
|1-phase compressible with thermal effect|[Notebook]()|
|2-phase oil-water|[Notebook]()|

*) No flow boundary condition and constant BHP well assumption

<!--

## Single-phase Fluid

### Water
Example inputs: 

* mu = 1 cp
* rho = 1000 kg/m3

Formula: 

* mu, rho = constant

```
FLUID1
water
1,1000
```
### Oil
Example inputs:

* mu = 1 cp
* rho_r = 850 kg/m3 (Reference rho @ reference pressure)
* pr = 200 bar (Reference pressure)
* c = 1e-3 1/bar (Fluid compressibility)

Formula:
* mu = constant
* rho(p) = rho_r * exp(c * (p - p_r))

```
FLUID1
oil
1,850,200,0.001
```

Example inputs for gas:

* mu0 = 5 cp (Viscosity at zero)
* rho_r = 850 kg/m3
* pr = 200 bar
* c = 1e-3 1/bar
* c_mu = 2e-3 1/bar (Viscosity coefficient)

Formula:
* mu(p) = mu0*  (1 + c_mu * (p - p_r))
* rho(p) = rho_r * exp(c * (p - p_r))

```
FLUID1
gas
5,850,200,0.001,0.002
```

<!--
Experimentations of MATLAB Reservoir Simulation Toolbox in Google Colab to port it with Python and utilize free GPUs for faster computation

<p align="center">
  <img src="https://user-images.githubusercontent.com/51282928/100498951-68ebb580-3198-11eb-95c7-87ed7c1e6e9c.png" width="700" />
</p>

<!--
Experimentations of MATLAB Reservoir Simulation Toolbox in Google Colab to port it with Python and utilize free GPUs for faster computation

### Successful:

Phase 1. Full codes and documentations are preserved in [Zenodo]()
* Use Google Colab to run a MATLAB (Octave) script of MRST simulation of five-spot waterflooding in SPE10 model.
* Optimizing the well placement by coupling the simulation with a Python optimizer, such as Bayesian optimization.

### On experimentation:
* Modifying the MRST script to use `gpuArray` so that the simulation can use the free GPU in Colab and speed up simulation.
* Using `Tensorflow Probability` to improve optimizations, powered by free GPU in Colab.
* Experimenting with various optimizers (`Scipy`, `Optuna`, `Platypus`, etc.) and multi-objective optimizers (`Pymoo`).

### Future:
* Well placement optimization for different scenarios (CO2, surfactant, and polymer injection)
