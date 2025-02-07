Distributionally Robust Control and State Estimation for Linear Stochastic Systems
====================================================

This repository includes the source code for implementing 
Linear-Quadratic-Gaussian(LQG), Wasserstein Distributionally Robust Controller(WDRC), Distributionally Robust Linear Quadratic Control (DRLQC), WDRC+WKF, WDRC+DRMMSE,
and Distributionally Robust Control and Estimation(DRCE) [Ours]

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)**
- (pickle5) if relevant error occurs
- joblib (>=1.4.2)
- pykalmin (>=0.9.7)
## Additional Requirements to run DRLQC
- Pytorch 2.0
- [Pymanopt] https://pymanopt.org/

## Code explanation
---
### Comparison including DRLQC (a) Gaussian
First, generate the Total Cost data using
```
python main_param_EM.py --dist normal --noise_dist normal
```
After data generation, plot the results using
```
python plot_params4_EM.py --dist normal --noise_dist normal
```
---
### Comparison including DRLQC (b) U-Quadratic
First, generate the Total Cost data using
```
python main_param_EM.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_EM.py --dist quadratic --noise_dist quadratic
```
---
### Computation Time
First, generate the Computation_time data using
```
python main_time.py
```
Note that main_time.py is a time-consuming process.
After Data ge1neration, plot the results using
```
python plot_time.py
```
---
### Out-of-sample Performacne (a), Reliability (b)
First, generate the data using
```
python main_OS_parallel.py
```
Note that this Out-of-Sample Experiment is a time consuming process.
After data generation, plot Figure (a), (b) using
```
python plot_osp.py
```
---
### Estimator Performance (a)
First, generate the Total Cost data using
```
python main_param_filter.py
```
After data generation, plot the results using
```
python plot_params4_F.py
```
---
### Estimator Performance (b)
First, generate the Total Cost data using
```
python main_param_filter.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_F.py --dist quadratic --noise_dist quadratic
```
---
### Long Horizon (a)
First, generate the Total Cost data using
```
python main_param_longT_parallel.py
```
After data generation, plot the results using
```
python plot_params_long.py
```
---
### Long Horizon (b)
First, generate the Total Cost data using
```
python main_param_longT_parallel.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_long.py --dist quadratic --noise_dist quadratic
```
---
### 2D trajectory tracking (a) Curvy
```
python main_param_2D.py
```
Then generate plot using
```
python plot4_2d.py --use_lambda --dist normal --noise_dist normal
```
Make sure to choose which parameter to draw a plot.
For parameter plot,
```
python plot_params_2D.py --use_lambda --dist normal --noise_dist normal
```
---
### 2D trajectory tracking (b) Circular
```
python main_param_2D.py --trajectory circular
```
Then generate plot using
```
python plot4_2d.py --use_lambda --dist normal --noise_dist normal --trajectory circular
```
Make sure to choose which parameter to draw a plot.
For parameter plot,
```
python plot_params_2D.py --use_lambda --dist normal --noise_dist normal --trajectory circular
```
### Vehicle Control Problem (nx=21, nu=11, ny=10) : Total Cost - Gaussian
First, generate the Total Cost data using
```
python main_param_vehicle_EM.py --dist normal --noise_dist normal
```
After data generation, plot the results using
```
python plot_params_vehicle_EM.py --dist normal --noise_dist normal
```
---
### Vehicle Control Problem (nx=21, nu=11, ny=10) : Total Cost - U-Quadratic
First, generate the Total Cost data using
```
python main_param_vehicle_EM.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_vehicle_EM.py --dist quadratic --noise_dist quadratic
```
