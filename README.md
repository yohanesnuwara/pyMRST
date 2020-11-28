# mrst-colab
Experimentations of MATLAB Reservoir Simulation Toolbox in Google Colab to port it with Python and utilize free GPUs for faster computation

Successful:
* Use Google Colab to run a MATLAB (Octave) script of MRST simulation of five-spot waterflooding in SPE10 model.
* Couple the simulation with Python optimizer such as Bayesian optimization for well placement optimization.

On experimentation:
* Modifying the MRST script to use `gpuArray` so that the simulation can use the free GPU in Colab and speed up simulation.
* Using `Tensorflow Probability` to improve optimizations, powered by free GPU in Colab.
* Experimenting with various optimizers (`Scipy`, `Optuna`, `Platypus`, etc.) and multi-objective optimizers (`Pymoo`).

Future:
* Well placement optimization for different scenarios (CO2, surfactant, and polymer injection)
