# mrst-colab

Experimentations of MATLAB Reservoir Simulation Toolbox in Google Colab to port it with Python and utilize free GPUs for faster computation

<p align="center">
  <img src="https://user-images.githubusercontent.com/51282928/100498951-68ebb580-3198-11eb-95c7-87ed7c1e6e9c.png" width="700" />
</p>

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
