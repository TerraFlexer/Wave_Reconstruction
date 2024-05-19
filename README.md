# Wave_Reconstruction
## Program realization of Projection method of reconstruction of multifocal and spiral wavefronts by their slopes and 3 differentapproaches of searching best hyperparameters for his method on different wavefronts classes
* Optuna
* Genetic algorithms
* Adam optimisator

# Program blocks
* **main.py** - The main field for testing program and visualising the results using functions and procedures from other blocks;
* **splines.py** - Contains functions which realize spline logic (basis initialization, approximation);
* **method.py** - The core of the project, projection method is here. Functions for matrix and parameters initialization, counting devariatives and the method itslef (with 2 modes: numpy and torch);
* **srez.py** - Package for creating slice graphs of wavefronts;
* **optuna_parameters** - Package for searching best $\gamma$ and $s$ values using Optuna library;
* **gradint.py** - Package for searching best $\gamma$ and $s$ values using Adam otimizator;
* **genetic.py** - Package for searching best $\gamma$ and $s$ values using genetic algorithm;
* **metrics.py** - Contains function for counting difference metrics between source and method result wavefronts;
* **functions_package** - Package for functions which are used by other blocks;
