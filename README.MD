# From Data to Predictive Control: A Framework for Stochastic Linear Systems with Output Measurements
The code accompanying our Automatica submission that reproduces the results in the paper.

**Authors:** Haldun Balim, Andrea Carron, Melanie N. Zeilinger, and Johannes Köhler

## Setup 
From the base directory of this repository, install dependencies with:
~~~~
pip install -r requirements.txt
~~~~
Then, install this package with:
~~~~
python setup.py build  
pip install .
~~~~

## Reproducing The Figures
The files in the scripts directory can be used to replicate the figures in the paper. First run the following scripts to execute the offline phase of the algorithm:
~~~~
python scripts/spring_mass_offline.py
~~~~
This will create a time-stamped directory in the outputs directory that contains the offline phase results. Additionally, run the following scripts to compute the offline phase of the algorithm for fully-parameterized and ARX structured systems. These will also create pickled files in the corresponding time-stamped directory:
~~~~
python scripts/spring_mass_offline_lti.py
python scripts/spring_mass_offline_arx.py
~~~~
Then, run the following scripts to create figures:
~~~~
python scripts/generate_control_figs.py
python scripts/generate_pred_err_fig.py
python scripts/generate_dynof_fig.py
python scripts/sddpc_compare.py
~~~~
The figures will be saved in the corresponding time-stamped directory.

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Citation
If you use this code in your research, please cite the accompanying paper: 
Balim, H., Carron, A., Zeilinger, M. N., & Köhler, J. (2024). From Data to Predictive Control: A Framework for Stochastic Linear Systems with Output Measurements. 