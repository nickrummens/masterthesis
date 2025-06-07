"""
PINN-MT2.0: Inverse identification for the side loaded plate example

This script replicates the example from Martin et al. (2019) using a FEM reference
solution and a Physics-Informed Neural Network (PINN) to identify material properties.
"""

import os
import time
import json
import argparse
import platform
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

# =============================================================================
# 1. Utility Function: Coordinate Transformation for SPINN
# =============================================================================
def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

# =============================================================================
# 2. Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Physics Informed Neural Networks for Linear Elastic Plate")
parser.add_argument('--n_iter', type=int, default=int(1e10), help='Number of iterations')
parser.add_argument('--log_every', type=int, default=250, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=5, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='*', default=['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_fn', nargs='+', default='MSE', help='Loss functions')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1,1,1], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=10000, help='Number of collocation points for PDE evaluation')
parser.add_argument('--num_point_test', type=int, default=100000, help='Number of test points')

parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=5, help='Depth of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

parser.add_argument('--measurments_type', choices=['displacement','strain'], default='displacement', help='Type of measurements')
parser.add_argument('--num_measurments', type=int, default=16, help='Number of measurements (should be a perfect square -16)')
parser.add_argument('--noise_magnitude', type=float, default=1e-5, help='Gaussian noise magnitude (not for DIC simulated)')
parser.add_argument('--u_0', nargs='+', type=float, default=[0,0], help='Displacement scaling factor for Ux and Uy, default(=0) use measurements norm')
parser.add_argument('--params_iter_speed', nargs='+', type=float, default=[1,1], help='Scale iteration step for each parameter')
parser.add_argument('--coord_normalization', type=bool, default=False, help='Normalize the input coordinates')

parser.add_argument('--FEM_dataset', type=str, default='fem_solution_dogbone_experiments_ROI.dat', help='Path to FEM data -- fem_solution_dogbone_experiments_ROI.dat')
parser.add_argument('--DIC_dataset_path', type=str, default='DIC_data', help='If default no_dataset, use FEM model for measurements -- (DIC_data or no_dataset)')
parser.add_argument('--DIC_dataset_number', type=int, default=1, help='Only for DIC simulated measurements')
parser.add_argument('--results_path', type=str, default='results_inverse', help='Path to save results')

args = parser.parse_args()

if len(args.log_output_fields[0]) == 0:
    args.log_output_fields = [] # Empty list for no logging

# For strain measurements, extend loss weights
if args.measurments_type == "strain":
    print("STRAIN")
    args.loss_weights.append(args.loss_weights[-1])

dde.config.set_default_autodiff("forward")

# =============================================================================
# 3. Global Constants, Geometry, and Material Parameters
# =============================================================================

# /!\ Distances are in mm, forces in N, Young's modulus in N/mm^2
L_max = 10.0 # dogbone clamp width
H_max = 60.0
R = 20
theta = np.arccos(15/R)
b = 10
indent_x = R - (R * np.cos(theta))
indent_y = R * np.sin(theta)
free_length = 110
L_u = 60
L_c = free_length - (2*indent_y)
H_clamp = 0
total_points_vert = 140 #see geometry mapping code
total_points_hor = 40
x_max_FEM = (2*indent_x) + b
y_max_FEM = 2*H_clamp + 2*indent_y + L_c
t = 2 #thickness

#ROI positioning
offs_x = indent_x
offs_y = indent_y + H_clamp + (L_c-L_u)/2
x_max_ROI = b
y_max_ROI = L_u
offsets = [offs_x, offs_y]
ROI = [b, L_u]
x_max_full = [offs_x + x_max_ROI, offs_y + y_max_ROI]

#coord normalization
x_max = [1.0, 1.0] if args.coord_normalization else x_max_full

#Material parameters
E_actual  = 69e3  # Actual Young's modulus 210 GPa = 210e3 N/mm^2
nu_actual = 0.33  #0.312   # Actual Poisson's ratio
E_init    = 200e3   # Initial guess for Young's modulus
nu_init   = 0.60    # Initial guess for Poisson's ratio

#stress (FEM reference solution for 360N --> 360N/(2mmx20mm) = 9 MPa)
# force 190N --> p_stress = 4.75
# p_stress = 4.75 #9.375 #9
p_stress = 9

# Create trainable scaling factors (one per parameter)
params_factor = [dde.Variable(1 / s) for s in args.params_iter_speed]
trainable_variables = params_factor

# =============================================================================
# 4. Load FEM Data and Build Interpolation Functions
# =============================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"data_fem", args.FEM_dataset)
data = np.loadtxt(fem_file)
X_val      = data[:, :2]
u_val      = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]
solution_val = np.hstack((u_val, stress_val))

n_mesh_points = [total_points_hor,total_points_vert]

# #full specimen
# x_grid = np.linspace(0, x_max_FEM, 40)
# y_grid = np.linspace(0, y_max_FEM, 140)

#ROI
x_grid = np.linspace(offs_x, offs_x + x_max_ROI, 40)
y_grid = np.linspace(offs_y, offs_y + y_max_ROI, 140)

def create_interpolation_fn(data_array):
    num_components = data_array.shape[1]
    interpolators = []
    for i in range(num_components):
        interp = RegularGridInterpolator(
            (x_grid, y_grid),
            data_array[:, i].reshape(n_mesh_points[1], n_mesh_points[0]).T,
        )
        interpolators.append(interp)
    def interpolation_fn(x):
        x_in = transform_coords([x[0], x[1]])
        return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
    return interpolation_fn

solution_fn = create_interpolation_fn(solution_val)
strain_fn   = create_interpolation_fn(strain_val)

def strain_from_output(x, f):
    """
    Compute strain components from the network output for strain measurements.
    """
    x = transform_coords(x)
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
    return jnp.hstack([E_xx, E_yy, E_xy])

# =============================================================================
# 5.2 Stress Integral
# =============================================================================

n_integral = 1000
x_integral = np.linspace(offs_x, offs_x + x_max_ROI, n_integral).reshape(-1, 1)
y_integral = np.linspace(offs_y, offs_y + y_max_ROI, n_integral).reshape(-1, 1)
integral_points = [x_integral, y_integral]


def integral_stress(inputs, outputs, X):
    x = transform_coords(inputs)
    x_mesh = x[:,0].reshape((inputs[0].shape[0], inputs[0].shape[0]))
    Syy = outputs[0][:, 3:4].reshape(x_mesh.shape)
    return jnp.trapezoid(Syy, x_mesh, axis=0)*t

Integral_BC = dde.PointSetOperatorBC(integral_points, p_stress*((b+(2*indent_x))/b)*(b*t) , integral_stress) #stress at clamp --> scaled to stress at ROI --> times width times thickness
# Integral_BC = dde.PointSetOperatorBC(integral_points, 450 , integral_stress) #image25

bcs = [Integral_BC]

# bcs = []
# =============================================================================
# 5. Setup Measurement Data Based on Type (Displacement, Strain, DIC)
# =============================================================================
args.num_measurments = int(np.sqrt(args.num_measurments))**2
if args.measurments_type == "displacement":
    if args.DIC_dataset_path != "no_dataset":
        dic_path = os.path.join(dir_path, args.DIC_dataset_path)
        # SIMULATED DIC: speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv etc.
        # REAL-WORLD DIC: Image_0020_0.tiff_X_trans.csv
        


        X_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv"), delimiter=";",dtype=str)
        X_dic = X_dic.replace({',': '.'}, regex=True)
        X_dic = X_dic.apply(pd.to_numeric, errors='coerce')
        X_dic = X_dic.dropna(axis=1)
        X_dic = X_dic.to_numpy()
        X_dic += offs_x + 0.5

        Y_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_Y_trans.csv"), delimiter=";",dtype=str)
        Y_dic = Y_dic.replace({',': '.'}, regex=True)
        Y_dic = Y_dic.apply(pd.to_numeric, errors='coerce')
        Y_dic = Y_dic.dropna(axis=1)
        Y_dic = Y_dic.to_numpy()
        #RW
        # Y_dic += offs_y + 0.4
        #sim
        Y_dic += 0.75

        Ux_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_U_trans.csv"), delimiter=";",dtype=str)
        Ux_dic = Ux_dic.replace({',': '.'}, regex=True)
        Ux_dic = Ux_dic.apply(pd.to_numeric, errors='coerce')
        Ux_dic = Ux_dic.dropna(axis=1)
        Ux_dic = Ux_dic.to_numpy()

        Uy_dic = pd.read_csv(os.path.join(dic_path, "speckle_pattern_Numerical_1_0.synthetic.tif_V_trans.csv"), delimiter=";",dtype=str)
        Uy_dic = Uy_dic.replace({',': '.'}, regex=True)
        Uy_dic = Uy_dic.apply(pd.to_numeric, errors='coerce')
        Uy_dic = Uy_dic.dropna(axis=1)
        Uy_dic = Uy_dic.to_numpy()
        #simulated
        Uy_dic = -Uy_dic

        #---------------------- SIMULATED DIC: ROI FIX -------------------------
        rows_in_range_y = []
        rows_in_range_x = []
        rows_in_range_ux = []
        rows_in_range_uy = []

        for row_y, row_x, row_ux, row_uy in zip(Y_dic, X_dic, Ux_dic, Uy_dic):
            for value in row_y:
                if 25 <= value <= 85:
                    rows_in_range_y.append(row_y)
                    rows_in_range_x.append(row_x)
                    rows_in_range_ux.append(row_ux)
                    rows_in_range_uy.append(row_uy)
                    break
        Y_dic = np.array(rows_in_range_y)
        X_dic = np.array(rows_in_range_x)
        Ux_dic = np.array(rows_in_range_ux)
        Uy_dic = np.array(rows_in_range_uy)

        # print(Y_dic.shape)
        Y_dic = Y_dic[::5, ::2]
        X_dic = X_dic[::5, ::2]
        Ux_dic = Ux_dic[::5, ::2]
        Uy_dic = Uy_dic[::5, ::2]

        # print(Ux_dic)
        # print(Uy_dic)

        # Ux_dic = Ux_dic.reshape(-1,1)
        # Uy_dic = Uy_dic.reshape(-1,1)
        #simulated
        Ux_dic = Ux_dic.reshape(-1,1, order='F')
        Uy_dic = Uy_dic.reshape(-1,1, order='F')


        x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
        y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)

        #simulated
        y_values = y_values[::-1]

        X_DIC_input = [x_values, y_values]
        DIC_data = np.hstack([Ux_dic, Uy_dic])
        # print(X_DIC_input)
        # print(DIC_data)
        if args.num_measurments != x_values.shape[0] * y_values.shape[0]:
            print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
            args.num_measurments = x_values.shape[0] * y_values.shape[0]
    else:
        X_DIC_input = [np.linspace(offs_x, offs_x + x_max_ROI, args.num_measurments).reshape(-1, 1),
               np.linspace(offs_y, offs_y + y_max_ROI, args.num_measurments).reshape(-1, 1)]
        DIC_data = solution_fn(X_DIC_input)[:, :2]
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape) #*DIC_data
        # print(X_DIC_input)
        # print(DIC_data)
    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                          lambda x, f, x_np: f[0][:, 0:1]/DIC_norms[0])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                          lambda x, f, x_np: f[0][:, 1:2]/DIC_norms[1])
    bcs += [measure_Ux, measure_Uy]

elif args.measurments_type == "strain":
    if args.DIC_dataset_path != "no_dataset":
        print("DIC ##########################")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dic_path = os.path.join(dir_path, "DIC_data")

        # SIMULATED DIC: speckle_pattern_Numerical_1_0.synthetic.tif_X_trans.csv etc.
        # REAL-WORLD DIC: Image_0020_0.tiff_X_trans.csv

        X_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_X_trans.csv"), delimiter=";",dtype=str)
        X_dic = X_dic.replace({',': '.'}, regex=True)
        X_dic = X_dic.apply(pd.to_numeric, errors='coerce')
        X_dic = X_dic.dropna(axis=1)
        X_dic = X_dic.to_numpy()
        X_dic += offs_x + 0.4

        Y_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_Y_trans.csv"), delimiter=";",dtype=str)
        Y_dic = Y_dic.replace({',': '.'}, regex=True)
        Y_dic = Y_dic.apply(pd.to_numeric, errors='coerce')
        Y_dic = Y_dic.dropna(axis=1)
        Y_dic = Y_dic.to_numpy()
        # real world: 
        Y_dic += offs_y + 0.4
        #simulated
        # Y_dic += 0.5

        Ux_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_U_trans.csv"), delimiter=";",dtype=str)
        Ux_dic = Ux_dic.replace({',': '.'}, regex=True)
        Ux_dic = Ux_dic.apply(pd.to_numeric, errors='coerce')
        Ux_dic = Ux_dic.dropna(axis=1)
        Ux_dic = Ux_dic.to_numpy()

        Uy_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_V_trans.csv"), delimiter=";",dtype=str)
        Uy_dic = Uy_dic.replace({',': '.'}, regex=True)
        Uy_dic = Uy_dic.apply(pd.to_numeric, errors='coerce')
        Uy_dic = Uy_dic.dropna(axis=1)
        Uy_dic = Uy_dic.to_numpy()

        E_xx_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_exx.csv"), delimiter=";",dtype=str)
        E_xx_dic = E_xx_dic.replace({',': '.'}, regex=True)
        E_xx_dic = E_xx_dic.apply(pd.to_numeric, errors='coerce')
        E_xx_dic = E_xx_dic.dropna(axis=1)
        E_xx_dic = E_xx_dic.to_numpy()

        E_yy_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_eyy.csv"), delimiter=";",dtype=str)
        E_yy_dic = E_yy_dic.replace({',': '.'}, regex=True)
        E_yy_dic = E_yy_dic.apply(pd.to_numeric, errors='coerce')
        E_yy_dic = E_yy_dic.dropna(axis=1)
        E_yy_dic = E_yy_dic.to_numpy()

        E_xy_dic = pd.read_csv(os.path.join(dic_path, "Image_0020_0.tiff_exy.csv"), delimiter=";",dtype=str)
        E_xy_dic = E_xy_dic.replace({',': '.'}, regex=True)
        E_xy_dic = E_xy_dic.apply(pd.to_numeric, errors='coerce')
        E_xy_dic = E_xy_dic.dropna(axis=1)
        E_xy_dic = E_xy_dic.to_numpy()


        #---------------------- SIMULATED DIC: ROI FIX -------------------------
        rows_in_range_y = []
        rows_in_range_x = []
        rows_in_range_ux = []
        rows_in_range_uy = []
        rows_in_range_exx = []
        rows_in_range_eyy = []
        rows_in_range_exy = []

        for row_y, row_x, row_ux, row_uy, row_exx, row_eyy, row_exy in zip(Y_dic, X_dic, Ux_dic, Uy_dic, E_xx_dic, E_yy_dic, E_xy_dic):
            for value in row_y:
                if 25 <= value <= 85:
                    rows_in_range_y.append(row_y)
                    rows_in_range_x.append(row_x)
                    rows_in_range_ux.append(row_ux)
                    rows_in_range_uy.append(row_uy)
                    rows_in_range_exx.append(row_exx)
                    rows_in_range_eyy.append(row_eyy)
                    rows_in_range_exy.append(row_exy)
                    break
        Y_dic = np.array(rows_in_range_y)
        X_dic = np.array(rows_in_range_x)
        Ux_dic = np.array(rows_in_range_ux)
        Uy_dic = np.array(rows_in_range_uy)
        E_xx_dic = np.array(rows_in_range_exx)
        E_yy_dic = np.array(rows_in_range_eyy)
        E_xy_dic = np.array(rows_in_range_exy)

        # print(Y_dic.shape)
        # Y_dic = Y_dic[::3, ::2]
        # X_dic = X_dic[::3, ::2]
        # Ux_dic = Ux_dic[::3, ::2]
        # Uy_dic = Uy_dic[::3, ::2]
        # E_xx_dic = E_xx_dic[::3, ::2]
        # E_yy_dic = E_yy_dic[::3, ::2]
        # E_xy_dic = E_xy_dic[::3, ::2]


        E_xx_dic = E_xx_dic.reshape(-1,1)
        E_yy_dic = E_yy_dic.reshape(-1,1)
        E_xy_dic = E_xy_dic.reshape(-1,1)
        #simulated
        # E_xx_dic = E_xx_dic.reshape(-1,1, order='F')
        # E_yy_dic = E_yy_dic.reshape(-1,1, order='F')
        # E_xy_dic = E_xy_dic.reshape(-1,1, order='F')


        x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
        y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
        #simulated
        # y_values = y_values[::-1]
        X_DIC_input = [x_values, y_values]
        DIC_data = np.hstack([E_xx_dic, E_yy_dic, E_xy_dic])
        # print(X_DIC_input)
        # print(DIC_data)
        # print(np.mean(E_xx_dic))
        # print(np.mean(E_yy_dic))
        if args.num_measurments != x_values.shape[0] * y_values.shape[0]:
            print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
            args.num_measurments = x_values.shape[0] * y_values.shape[0]
    else:
        X_DIC_input = [np.linspace(offs_x, offs_x + x_max_ROI, args.num_measurments).reshape(-1, 1),
               np.linspace(offs_y, offs_y + y_max_ROI, args.num_measurments).reshape(-1, 1)]
        DIC_data = strain_fn(X_DIC_input)
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
        # print(X_DIC_input)
        # print(DIC_data)
    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Exx = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 0:1]/DIC_norms[0])
    measure_Eyy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 1:2]/DIC_norms[1])
    measure_Exy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 2:3]/DIC_norms[2],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 2:3]/DIC_norms[2])
    bcs += [measure_Exx, measure_Eyy, measure_Exy]

else:
    raise ValueError("Invalid measurement type. Choose 'displacement' or 'strain'.")

# Use measurements norm as the default scaling factor
if args.DIC_dataset_path != "no_dataset":
    disp_norms = np.mean(np.abs(np.hstack([Ux_dic, Uy_dic])), axis=0)
else:
    disp_norms = np.mean(np.abs(solution_fn(X_DIC_input)[:, :2]), axis=0)
args.u_0 = [disp_norms[i] if not args.u_0[i] else args.u_0[i] for i in range(2)]



# =============================================================================
# 6. PINN Implementation: Boundary Conditions and PDE Residual
# =============================================================================
# Define the domain geometry
geom = dde.geometry.Rectangle([offs_x, offs_y], [offs_x + x_max_ROI, offs_y+y_max_ROI])

def HardBC(x, f, x_max=x_max[0]):
    """
    Apply hard boundary conditions via transformation.
    If x is provided as a list of 1D arrays, transform it to a 2D meshgrid.
    """
    if isinstance(x, list):
        x = transform_coords(x)
    Ux  = f[:, 0] * args.u_0[0] 
    Uy  = f[:, 1] * args.u_0[1]
    Sxx = f[:, 2] * (x_max - x[:, 0])/x_max * (x[:,0] - offs_x)/x_max
    Syy = f[:, 3]
    Sxy = f[:, 4] * (x_max - x[:, 0])/x_max * (x[:,0] - offs_x)/x_max
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

def pde(x, f, unknowns=params_factor):
    """
    Define the PDE residuals for the linear elastic plate.
    """
    x = transform_coords(x)
    param_factors = [u * s for u, s in zip(unknowns, args.params_iter_speed)]
    E = E_init * param_factors[0]
    nu = nu_init * param_factors[1]
    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    lmbd=2*mu*lmbd/(lmbd+(2*mu)) # plane stress
    
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
    
    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)[0]
    Syy_y = dde.grad.jacobian(f, x, i=3, j=1)[0]
    Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)[0]
    Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)[0]
    
    # momentum_x = (Sxx_x + Sxy_y)*x_max_ROI
    # momentum_y = (Sxy_x + Syy_y)*y_max_ROI
    momentum_x = Sxx_x + Sxy_y
    momentum_y = Sxy_x + Syy_y
    
    f_val = f[0] # f[1] is the function 
    stress_x  = S_xx - f_val[:, 2:3]
    stress_y  = S_yy - f_val[:, 3:4]
    stress_xy = S_xy - f_val[:, 4:5]
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

def input_scaling(x):
    """
    Scale the input coordinates to the range [0, 1].
    """
    if isinstance(x, list):
        return [(x[i]-offsets[i])/ROI[i] for i in range(len(x))]
    else:
        #TODO
        return jnp.array([x[0,0]-offsets[0]/ROI[0], x[0,1]-offsets[1]/ROI[1]])
        # return jnp.vstack([(x[:,i]-offsets[i])/ROI[i] for i in range(len(x))]) ??? len(x[0]) ???
        # return(x)
# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================
layers = [2] + [args.net_width] * args.net_depth + [5]
net = dde.nn.SPINN(layers, args.activation, args.initialization, args.mlp)
batch_size = args.num_point_PDE + args.num_measurments
num_params = sum(p.size for p in jax.tree.leaves(net.init(jax.random.PRNGKey(0), jnp.ones(layers[0]))))

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=args.num_point_PDE,
    solution=solution_fn,
    num_test=args.num_point_test,
    is_SPINN=True,
)
if args.coord_normalization:
    net.apply_feature_transform(input_scaling)
net.apply_output_transform(HardBC)

model = dde.Model(data, net)
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=[1]*len(args.loss_weights), loss=args.loss_fn,
              external_trainable_variables=trainable_variables)

#loss_weights=args.loss_weights, loss=args.loss_fn,

# =============================================================================
# 8. Setup Callbacks for Logging
# =============================================================================
results_path = os.path.join(dir_path, args.results_path)
if args.DIC_dataset_path != "no_dataset":
    dic_prefix = 'dic_'
    noise_prefix = args.DIC_dataset_path.split('/')[-1]
else:
    dic_prefix = ''
    noise_prefix = f"{args.noise_magnitude}noise"
folder_name = f"{dic_prefix}{args.measurments_type}_x{args.num_measurments}_{noise_prefix}_{args.available_time if args.available_time else args.n_iter}{'min' if args.available_time else 'iter'}"
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    folder_name = f"{folder_name}-{max(suffixes)+1}" if suffixes else f"{folder_name}-1"
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

callbacks = []
if args.available_time:
    callbacks.append(dde.callbacks.Timer(args.available_time))
callbacks.append(dde.callbacks.VariableValue(params_factor, period=args.log_every,
                                               filename=os.path.join(new_folder_path, "variables_history.dat"),
                                               precision=8))

# Log the history of the output fields
def output_log(x, output, field):
    if field in ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy']:
        return output[0][:, ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'].index(field)]
    if field in ['Exx', 'Eyy', 'Exy']:
        return strain_from_output(x, output)[:, ['Exx', 'Eyy', 'Exy'].index(field)]
    raise ValueError(f"Invalid field name: {field}")
        
# X_plot = [np.linspace(offs_x, offs_x + x_max_ROI, n_mesh_points[0]),
#                np.linspace(offs_y, offs_y + y_max_ROI, n_mesh_points[1])]
X_plot = [np.linspace(offs_x, offs_x + x_max_ROI, n_mesh_points[0]).reshape(-1, 1),
               np.linspace(offs_y, offs_y + y_max_ROI, n_mesh_points[1]).reshape(-1, 1)]
for i, field in enumerate(args.log_output_fields): # Log output fields
    callbacks.append(
        dde.callbacks.OperatorPredictor(
            X_plot,
            lambda x, output, field=field: output_log(x, output, field),
            period=args.log_every,
            filename=os.path.join(new_folder_path, f"{field}_history.dat"),
            precision=6
        )
    )

# =============================================================================
# 9. Calculate Loss Weights based on the Gradient of the Loss Function
# =============================================================================
from jax.flatten_util import ravel_pytree

# def loss_function(params,comp=0,inputs=[X_DIC_input]*len(bcs)+[X_plot]):
#     return model.outputs_losses_train(params, inputs, None)[1][comp]

def loss_function(params,comp=0,inputs=[integral_points]+[X_DIC_input]*(len(bcs)-1)+[X_plot]):
    return model.outputs_losses_train(params, inputs, None)[1][comp]

n_loss = len(args.loss_weights)

def calc_loss_weights(model):
    loss_grads = [1]*n_loss

    for i in range(n_loss):
        grad_fn = jax.grad(lambda params,comp=i: loss_function(params,comp))
        grads = grad_fn(model.params)[0]
        flattened_grad = ravel_pytree(list(grads.values())[0])[0]
        loss_grads[i] = jnp.linalg.norm(flattened_grad)

    loss_grads = jnp.array(loss_grads)
    loss_weights_grads = jnp.sqrt(jnp.sum(loss_grads)/loss_grads) # Caution: ad-hoc sqrt
    return loss_weights_grads, loss_grads

loss_weights_grads, loss_grads = calc_loss_weights(model)
new_loss_weights = [w * g for w, g in zip(args.loss_weights, loss_weights_grads)]
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=new_loss_weights, loss=args.loss_fn,
              external_trainable_variables=trainable_variables)

# =============================================================================
# 10. Training
# =============================================================================
start_time = time.time()
print(f"E(GPa): {E_init * params_factor[0].value * args.params_iter_speed[0]/1e3:.3f}, nu: {nu_init * params_factor[1].value * args.params_iter_speed[1]:.3f}")
losshistory, train_state = model.train(iterations=args.n_iter, callbacks=callbacks, display_every=args.log_every)
elapsed = time.time() - start_time

# =============================================================================
# 11. Logging
# =============================================================================
dde.utils.save_loss_history(losshistory, os.path.join(new_folder_path, "loss_history.dat"))

params_init = [E_init, nu_init]
variables_history_path = os.path.join(new_folder_path, "variables_history.dat")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()

# Update the variables history with scaled values
with open(variables_history_path, "w") as f:
    for line in lines:
        step, value = line.strip().split(' ', 1)
        values = [scale * init * val for scale, init, val in zip(args.params_iter_speed, params_init, eval(value))]
        f.write(f"{step} " + dde.utils.list_to_str(values, precision=8) + "\n")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()
# Final E and nu values as the average of the last 10 values 
E_final = np.mean([eval(line.strip().split(' ', 1)[1])[0] for line in lines[-10:]])
nu_final = np.mean([eval(line.strip().split(' ', 1)[1])[1] for line in lines[-10:]])
print(f"Final E(GPa): {E_final/1e3:.3f}, nu: {nu_final:.3f}")

def log_config(fname):
    """
    Save configuration and execution details to a JSON file, grouped by category.
    """
    system_info = {"OS": platform.system(), "Release": platform.release()}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        gpu_name, total_memory_mb = output.stdout.strip().split(", ")
        total_memory_gb = round(float(total_memory_mb.split(' ')[0]) / 1024, 2)
        gpu_info = {"GPU": gpu_name, "Total GPU Memory": f"{total_memory_gb:.2f} GB"}
    except subprocess.CalledProcessError:
        gpu_info = {"GPU": "No GPU found", "Total GPU Memory": "N/A"}
    
    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
    }
    network_info = {
        "net_width": args.net_width,
        "net_depth": args.net_depth,
        "num_params": num_params,
        "activation": args.activation,
        "mlp_type": args.mlp,
        "optimizer": args.optimizer,
        "initializer": args.initialization,
        "batch_size": batch_size,
        "lr": args.lr,
        "loss_weights": args.loss_weights,
        "params_iter_speed": args.params_iter_speed,
        "u_0": args.u_0,
        "logged_fields": args.log_output_fields,
    }
    problem_info = {
        "L_max": L_max,
        "H_max": H_max,
        "E_actual": E_actual,
        "nu_actual": nu_actual,
        "E_init": E_init,
        "nu_init": nu_init,
        "E_final": E_final,
        "nu_final": nu_final,
    }
    data_info = {
        "num_measurments": (int(np.sqrt(args.num_measurments)))**2,
        "noise_magnitude": args.noise_magnitude,
        "measurments_type": args.measurments_type,
        "DIC_dataset_path": args.DIC_dataset_path,
        "DIC_dataset_number": args.DIC_dataset_number,
        "FEM_dataset": args.FEM_dataset, 
    }
    info = {"system": system_info, "gpu": gpu_info, "execution": execution_info,
            "network": network_info, "problem": problem_info, "data": data_info}
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

log_config(os.path.join(new_folder_path, "config.json"))