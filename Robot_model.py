#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import sympy as sp
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import threading
from scipy.optimize import minimize
import warnings as warnings

# Enable interactive mode so plots update non-blocking
plt.ion()
from sympy import cos, sin, lambdify
from sympy.physics.mechanics import dynamicsymbols
from pydy.codegen.ode_function_generators import generate_ode_function
from functools import reduce, lru_cache
import dill as pickle  # Use dill instead of standard pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------
# Global Logger for utility functions
# ------------------------------------------------------------------------
GLOBAL_LOGGER = lambda msg: print(f"[GLOBAL DEBUG]: {msg}")
# If you wish to disable global logging completely, uncomment the following:
GLOBAL_LOGGER = lambda msg: None


# ------------------------------------------------------------------------
# Simple Logger (replace with Python's logging if desired)
# ------------------------------------------------------------------------
class Logger:
    def __init__(self, name: str, debug_enabled=False):
        self.name = name
        self.debug_enabled = debug_enabled

    def debug(self, msg: str):
        if self.debug_enabled:
            print(f"[DEBUG] {self.name}: {msg}")


# ------------------------------------------------------------------------
# Dynamic Cache Utility Functions using dill (with symbolic inputs)
# ------------------------------------------------------------------------
def get_cache_directory():
    """Ensure a dedicated directory exists for cached files."""
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def get_pickle_filename(identifier, ndof, version, sym_vars=None, flags=None, ext=".pkl"):
    filename = f"{identifier}_{ndof}R_{version}"
    if flags is not None:
        # Append the flags in a consistent sorted order.
        flag_str = "_".join(f"{k}{flags[k]}" for k in sorted(flags))
        filename += "_" + flag_str
    if sym_vars is not None:
        # Assume each symbol has a 'name' attribute; sort for consistency.
        var_names = "_".join(sorted([v.name for v in sym_vars if hasattr(v, "name")]))
        filename += "_" + var_names
    filename += ext
    full_path = os.path.join(get_cache_directory(), filename)
    GLOBAL_LOGGER(f"Generated filename: {full_path}")
    return full_path


def save_pickled(identifier, ndof, version, data, sym_vars=None, flags=None):
    filename = get_pickle_filename(identifier, ndof, version, sym_vars, flags)
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        GLOBAL_LOGGER(f"Saved {identifier} to {filename}")
    except Exception as e:
        GLOBAL_LOGGER(f"Error saving {filename}: {e}")


def load_pickled(identifier, ndof, version, sym_vars=None, flags=None, force_update=False):
    filename = get_pickle_filename(identifier, ndof, version, sym_vars, flags)
    if force_update:
        GLOBAL_LOGGER(f"Force update enabled; skipping cache for {filename}.")
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        GLOBAL_LOGGER(f"Loaded {identifier} from {filename}")
        return data
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        GLOBAL_LOGGER(f"Could not load {filename}: {e}")
        return None


# ------------------------------------------------------------------------
# Symbolic Utilities (CPU-side) with debug prints
# ------------------------------------------------------------------------
def apply_generalized_force(f):
    GLOBAL_LOGGER("Applying generalized force.")
    n = len(f)
    result = [f[i] if i == n - 1 else f[i] - f[i + 1] for i in range(n)]
    GLOBAL_LOGGER("Applied generalized force.")
    return result


def custom_exponent(q, A, k, q_lim):
    GLOBAL_LOGGER("Calculating custom exponent.")
    result = A * sp.exp(k * (q - q_lim)) / (148.42) ** k
    GLOBAL_LOGGER("Calculated custom exponent.")
    return result


def coordinate_limiting_force(q, q_low, q_up, a, b):
    GLOBAL_LOGGER("Calculating coordinate limiting force.")
    result = custom_exponent(q_low + 5, a, b, q) - custom_exponent(q, a, b, q_up - 5)
    GLOBAL_LOGGER("Calculated coordinate limiting force.")
    return result


def christoffel_symbols(M, q, i, j, k):
    return sp.Rational(1, 2) * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) - sp.diff(M[k, j], q[i]))


def coriolis_matrix(M, q, dq):
    GLOBAL_LOGGER("Computing coriolis matrix.")
    n = M.shape[0]
    C = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += christoffel_symbols(M, q, i, j, k) * dq[k]
    result = sp.trigsimp(C)
    GLOBAL_LOGGER("Computed coriolis matrix.")
    return result


def substitute(expression, constants):
    GLOBAL_LOGGER("Substituting constants into expression.")
    if isinstance(expression, (list, tuple)):
        return type(expression)(substitute(exp, constants) for exp in expression)
    elif isinstance(expression, dict):
        return {key: substitute(value, constants) for key, value in expression.items()}
    elif hasattr(expression, "subs"):
        return expression.subs(constants)
    else:
        return expression


# ------------------------------------------------------------------------
# PyTorch damped pseudoinverse function with debug prints
# ------------------------------------------------------------------------
def damped_pinv_torch(J, damping=0.01):
    GLOBAL_LOGGER("Computing damped pseudoinverse.")
    U, s, V = torch.svd(J, some=False)
    m = s.shape[0]
    S_damped = s / (s**2 + damping**2)
    V_t = V[:, :m]
    result = V_t @ torch.diag(S_damped) @ U.t()
    GLOBAL_LOGGER("Computed damped pseudoinverse.")
    return result


def damped_pinv_np(J, damping=0.01):
    """
    Compute a damped pseudoinverse of a NumPy matrix J.
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_damped = s / (s**2 + damping**2)
    return Vt.T @ np.diag(s_damped) @ U.T


# ------------------------------------------------------------------------
# ArmModel Class using PyTorch for GPU Numerical Computation with extensive logging
# (Dummy coordinate removed: all joint‐related arrays now have exactly nd entries)
# ------------------------------------------------------------------------
class ArmModel:
    """
    A planar arm model with either 3 DoFs (9 muscles) or 2 DoFs (6 muscles).

    Symbolic derivations (kinematics, dynamics, etc.) are performed on the CPU with Sympy.
    Numerical computations (forward kinematics, dynamics, and control) are executed on the GPU using PyTorch.

    Key symbolic expressions and lambdified functions are cached to disk using filenames that
    include the model's DoF and version. (Note: The dummy coordinate has been removed.)
    Extensive logging is added for debugging.
    """

    # --- Property Getters and Setters for L1 and L2 ---
    @property
    def L1(self):
        return float(self.constants[self.L[0]])

    @L1.setter
    def L1(self, value):
        self.constants[self.L[0]] = value
        self.pre_substitute_parameters()
        self.update_lambdified_functions()

    @property
    def L2(self):
        return float(self.constants[self.L[1]])

    @L2.setter
    def L2(self, value):
        self.constants[self.L[1]] = value
        self.pre_substitute_parameters()
        self.update_lambdified_functions()

    def get_flags(self):
        return {"gravity": self.use_gravity, "coordLim": self.use_coordinate_limits, "visc": self.use_viscosity}

    # --- Utility Methods for Active Variables ---
    def Q(self):
        self.logger.debug("Returning joint coordinates.")
        return self.q

    def QDot(self):
        self.logger.debug("Returning joint speeds.")
        return self.dq

    def U(self):
        self.logger.debug("Returning joint speeds (alias U).")
        return self.u

    def Tau(self):
        self.logger.debug("Returning generalized forces.")
        return self.tau

    # --------------------------------------------------------------------
    # Public Method: model_parameters
    # --------------------------------------------------------------------
    def model_parameters(self, **kwargs):
        self.logger.debug("Getting model parameters.")
        expected = {"q", "u", "in_deg"}
        for key in kwargs:
            if key not in expected:
                raise Exception("Unexpected Argument: " + key)
        result = self.__model_parameters(kwargs)
        self.logger.debug("Obtained model parameters.")
        return result

    def __model_parameters(self, dic):
        constants = self.constants.copy()
        q, dq, u = self.Q(), self.QDot(), self.U()
        in_deg = dic.get("in_deg", False)
        if "q" in dic:
            qs = np.array(dic["q"])
            if in_deg:
                qs = np.deg2rad(qs)
            for i in range(self.nd):
                constants[q[i]] = qs[i]
        if "u" in dic:
            us = np.array(dic["u"])
            for i in range(self.nd):
                constants[dq[i]] = us[i]
                constants[u[i]] = us[i]
        constants[self.t] = 0
        return constants

    # --------------------------------------------------------------------
    # Initialization and Model Construction
    # --------------------------------------------------------------------
    def __init__(
        self,
        use_gravity=0,
        use_coordinate_limits=0,
        use_viscosity=0,
        nd=3,
        version="V0",
    ):
        self.logger = Logger("ArmModel")
        self.logger.debug("Initializing ArmModel.")
        self.nd = nd
        self.md = 9 if nd == 3 else 6


        self.s = self.nd  # Removed dummy coordinate
        self.dim = list(range(self.s))
        self.version = version

        self.use_gravity = use_gravity
        self.use_coordinate_limits = use_coordinate_limits
        self.use_viscosity = use_viscosity
        self.sub_constants = True
 
        if self.nd == 3:
            self.state0 = np.deg2rad([45, 45, 45, 0, 0, 0])
            self.reference_pose = np.deg2rad([60, 70, 50])
            
        
        else:
            self.state0 = np.deg2rad([45, 45, 0, 0])
            self.reference_pose = np.deg2rad([60, 70])


        self.activation = np.zeros(self.md)   # initial activation for each muscle
        #self.fatigue = np.zeros(self.md)        # initial fatigue for each muscle
        self.Fmax = 1000 * np.ones(self.nd)

        if hasattr(self, 'Fmax'):
            self.x_max = np.max(self.Fmax)
        else:
            self.x_max = 1e6  # or any large number as default

        self.joint_low = np.deg2rad([0] * self.nd)
        self.joint_high = np.deg2rad([180] * self.nd)
        self.joint_vel_limits = 100 * np.ones(self.nd)
        self.joint_acc_limits = 200 * np.ones(self.nd)
        self.joint_jerk_limits = 50 * np.ones(self.nd)
    
        self.logger.debug("Constructing model components...")
        self.__construct_symbols()
        self.__construct_kinematics()
        self.__construct_kinetics()
        self.__construct_coordinate_limiting_forces()
        self.__construct_drawables()
        #self.construct_muscle_geometry()
        self.construct_muscle_geometry(use_parabolic=True, h=0.09, k=0.1, use_alt_representation=False)
        self.__define_muscle_parameters()
        self.__construct_rhs()
        self.__construct_models_Kinematics()
        self.__construct_models_jacobianderivatives()

        self.pre_substitute_parameters()
        self.__initialize_lambdified_functions()
        self.logger.debug("ArmModel initialization complete.")

    # --------------------------------------------------------------------
    # Public method to update lambdified functions
    # --------------------------------------------------------------------
    def update_lambdified_functions(self):
        self.__initialize_lambdified_functions()

    # --------------------------------------------------------------------
    # Pre-Substitution and Workspace Bounds Methods
    # --------------------------------------------------------------------
    def pre_substitute_parameters(self):
        self.logger.debug("Pre-substituting constant parameters into symbolic expressions.")
        self.sub_constants = False
        C = self.constants
        self.M = self.M.subs(C)
        self.tau_c = self.tau_c.subs(C)
        self.tau_g = self.tau_g.subs(C)
        self.tau_l = self.tau_l.subs(C)
        self.tau_b = self.tau_b.subs(C)
        self.f = self.f.subs(C)
        self.lm = self.lm.subs(C)
        self.lmd = self.lmd.subs(C)
        self.lmdd = self.lmdd.subs(C)
        self.R = self.R.subs(C)
        self.RDot = self.RDot.subs(C)
        self.RDotQDot = self.RDotQDot.subs(C)
        self.RTDq = self.RTDq.subs(C)
        self.ap = substitute(self.ap, C)
        self.bp = substitute(self.bp, C)
        self.bc = substitute(self.bc, C)
        self.jc = substitute(self.jc, C)
        self.ee = substitute(self.ee, C)
        if self._fk_expr is not None:
            self._fk_expr = sp.trigsimp(self._fk_expr.subs(C))
        if self._ik_expr is not None:
            for key in self._ik_expr:
                self._ik_expr[key] = sp.trigsimp(self._ik_expr[key].subs(C))
        self.logger.debug("Pre-substitution complete.")

    def allowed_workspace_bounds(self):
        self.logger.debug("Calculating allowed workspace bounds.")
        grids = [np.linspace(self.joint_low[i], self.joint_high[i], num=50) for i in range(self.nd)]
        mesh = np.meshgrid(*grids)
        joint_configs = np.vstack([m.flatten() for m in mesh]).T
        positions = np.array([self.forward_kinematics(config.tolist()) for config in joint_configs])
        x_min = positions[:, 0].min()
        x_max = positions[:, 0].max()
        y_min = positions[:, 1].min()
        y_max = positions[:, 1].max()
        self.logger.debug("Calculated workspace bounds.")
        return x_min, x_max, y_min, y_max

    # --------------------------------------------------------------------
    # Symbolic Construction Methods (CPU-side)
    # --------------------------------------------------------------------
    def __construct_symbols(self):
        self.logger.debug("Constructing symbols.")
        self.a = sp.Matrix(sp.symbols("a0:10"))
        self.b = sp.Matrix(sp.symbols("b0:10"))
        self.L = sp.Matrix(sp.symbols(f"L0:{self.nd}"))
        self.Lc = sp.Matrix(sp.symbols(f"Lc0:{self.nd}"))
        self.Iz = sp.Matrix(sp.symbols(f"Iz0:{self.nd}"))
        self.m = sp.Matrix(sp.symbols(f"m0:{self.nd}"))
        self.t = sp.symbols("t")
        self.g = sp.symbols("g")
        self.q = sp.Matrix(dynamicsymbols(f"theta0:{self.nd}"))
        self.u = sp.Matrix(dynamicsymbols(f"u0:{self.nd}"))
        self.dq = sp.Matrix([sp.diff(x, self.t) for x in self.q])
        self.ddq = sp.Matrix([sp.diff(x, self.t) for x in self.dq])
        self.tau = sp.Matrix(dynamicsymbols(f"tau0:{self.nd}"))
        # --- Updated constants dictionary with re-indexed muscle parameters ---
        constants = {
            self.a[0]: 0.055,
            self.a[1]: 0.055,
            self.a[2]: 0.220,
            self.a[3]: 0.24,
            self.a[4]: 0.040,
            self.a[5]: 0.040,
            self.a[6]: 0.220,
            self.a[7]: 0.06,
            self.a[8]: 0.26,
            self.b[0]: 0.080,
            self.b[1]: 0.11,
            self.b[2]: 0.030,
            self.b[3]: 0.03,
            self.b[4]: 0.045,
            self.b[5]: 0.045,
            self.b[6]: 0.048,
            self.b[7]: 0.050,
            self.b[8]: 0.03,
            self.g: 9.81,
        }
        if self.nd == 3:
            constants[self.L[0]] = 0.310
            constants[self.L[1]] = 0.270
            constants[self.L[2]] = 0.150
            constants[self.Lc[0]] = 0.165
            constants[self.Lc[1]] = 0.135
            constants[self.Lc[2]] = 0.075
            constants[self.m[0]] = 1.93
            constants[self.m[1]] = 1.32
            constants[self.m[2]] = 0.35
            constants[self.Iz[0]] = 0.0141
            constants[self.Iz[1]] = 0.0120
            constants[self.Iz[2]] = 0.001
        else:
            constants[self.L[0]] = 0.310
            constants[self.L[1]] = 0.270
            constants[self.Lc[0]] = 0.165
            constants[self.Lc[1]] = 0.135
            constants[self.m[0]] = 1.93
            constants[self.m[1]] = 1.32
            constants[self.Iz[0]] = 0.0141
            constants[self.Iz[1]] = 0.0120
        self.constants = constants
        self.logger.debug("Symbols constructed.")

        self.logger.debug("Constructing centers of mass (xc).")
        self.xc = load_pickled("xc", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())

        if self.xc is None:
            if self.nd == 3:
                xc1 = sp.Matrix(
                    [
                        self.Lc[0] * cos(self.q[0]),
                        self.Lc[0] * sin(self.q[0]),
                        0,
                        0,
                        0,
                        self.q[0],
                    ]
                )
                xc2 = sp.Matrix(
                    [
                        self.L[0] * cos(self.q[0]) + self.Lc[1] * cos(self.q[0] + self.q[1]),
                        self.L[0] * sin(self.q[0]) + self.Lc[1] * sin(self.q[0] + self.q[1]),
                        0,
                        0,
                        0,
                        self.q[0] + self.q[1],
                    ]
                )
                xc3 = sp.Matrix(
                    [
                        self.L[0] * cos(self.q[0]) + self.L[1] * cos(self.q[0] + self.q[1]) + self.Lc[2] * cos(self.q[0] + self.q[1] + self.q[2]),
                        self.L[0] * sin(self.q[0]) + self.L[1] * sin(self.q[0] + self.q[1]) + self.Lc[2] * sin(self.q[0] + self.q[1] + self.q[2]),
                        0,
                        0,
                        0,
                        self.q[0] + self.q[1] + self.q[2],
                    ]
                )
                self.xc = [xc1, xc2, xc3]
            else:
                xc1 = sp.Matrix(
                    [
                        self.Lc[0] * cos(self.q[0]),
                        self.Lc[0] * sin(self.q[0]),
                        0,
                        0,
                        0,
                        self.q[0],
                    ]
                )
                xc2 = sp.Matrix(
                    [
                        self.L[0] * cos(self.q[0]) + self.Lc[1] * cos(self.q[0] + self.q[1]),
                        self.L[0] * sin(self.q[0]) + self.Lc[1] * sin(self.q[0] + self.q[1]),
                        0,
                        0,
                        0,
                        self.q[0] + self.q[1],
                    ]
                )
                self.xc = [xc1, xc2]

            save_pickled("xc", self.nd, self.version, self.xc, sym_vars=list(self.q), flags=self.get_flags())

        self.logger.debug("Centers of mass (xc) constructed.")

    def __construct_kinematics(self):
        self.logger.debug("Constructing kinematics.")
        self._fk_expr = load_pickled("fk_expr", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if self._fk_expr is None:
            if self.nd == 2:
                self._fk_expr = sp.Matrix(
                    [
                        self.L[0] * cos(self.q[0]) + self.L[1] * cos(self.q[0] + self.q[1]),
                        self.L[0] * sin(self.q[0]) + self.L[1] * sin(self.q[0] + self.q[1]),
                    ]
                )
            elif self.nd == 3:
                self._fk_expr = sp.Matrix(
                    [
                        self.L[0] * cos(self.q[0]) + self.L[1] * cos(self.q[0] + self.q[1]) + self.L[2] * cos(self.q[0] + self.q[1] + self.q[2]),
                        self.L[0] * sin(self.q[0]) + self.L[1] * sin(self.q[0] + self.q[1]) + self.L[2] * sin(self.q[0] + self.q[1] + self.q[2]),
                    ]
                )
            save_pickled("fk_expr", self.nd, self.version, self._fk_expr, sym_vars=list(self.q), flags=self.get_flags())
        self.logger.debug("Forward kinematics expression constructed.")
        self.vc = load_pickled("vc", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if self.vc is None:
            self.vc = [sp.diff(x, self.t) for x in self.xc]
            save_pickled("vc", self.nd, self.version, self.vc, sym_vars=list(self.q), flags=self.get_flags())
        self.logger.debug("Velocity expressions (vc) constructed.")
        self.Jc = load_pickled("Jc", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if self.Jc is None:
            self.Jc = [x.jacobian(self.QDot()) for x in self.vc]
            save_pickled("Jc", self.nd, self.version, self.Jc, sym_vars=list(self.q), flags=self.get_flags())
        self.logger.debug("Jacobian of centers (Jc) constructed.")

    def __construct_kinetics(self):
        self.logger.debug("Constructing kinetics.")
        self.M = load_pickled("mass_matrix", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if self.M is None:
            M_list = []
            for i in self.dim:
                spatial_inertia = sp.diag(self.m[i], self.m[i], self.m[i], 0, 0, self.Iz[i])
                M_list.append(self.Jc[i].T * spatial_inertia * self.Jc[i])
            self.M = sp.trigsimp(reduce(lambda x, y: x + y, M_list))
            save_pickled("mass_matrix", self.nd, self.version, self.M, sym_vars=list(self.q), flags=self.get_flags())

        self.logger.debug("Mass matrix constructed.")
        self.C = coriolis_matrix(self.M, self.Q(), self.QDot())
        self.tau_c = sp.trigsimp(self.C * sp.Matrix(self.QDot()))
        self.V = sum(self.m[i] * self.g * self.xc[i][1] for i in self.dim)
        # Apply use_gravity flag here so that tau_g is pre-scaled.
        self.tau_g = self.use_gravity * sp.Matrix([sp.diff(self.V, x) for x in self.Q()])
        self.logger.debug("Kinetics constructed.")

    def __construct_coordinate_limiting_forces(self):
        self.logger.debug("Constructing coordinate limiting forces.")
        self.__tau_l = load_pickled("tau_l", self.nd, self.version, sym_vars=list(self.q))
        if self.__tau_l is None:
            a_val, b_val = 5, 50
            if self.nd == 3:
                q_low = [np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]
                q_up = [np.deg2rad(175), np.pi, np.deg2rad(100)]
            else:
                q_low = [np.deg2rad(5), np.deg2rad(5)]
                q_up = [np.deg2rad(175), np.deg2rad(175)]
            # self.__tau_l = [coordinate_limiting_force(self.q[i], q_low[i], q_up[i], a_val, b_val) for i in range(self.nd)]
            self.__tau_l = [self.use_coordinate_limits * coordinate_limiting_force(self.q[i], q_low[i], q_up[i], a_val, b_val) for i in range(self.nd)]
            save_pickled("tau_l", self.nd, self.version, self.__tau_l, sym_vars=list(self.q), flags=self.get_flags())
        self.logger.debug("Coordinate limiting forces constructed.")

    def __construct_drawables(self):
        self.logger.debug("Constructing drawables.")
        saved = load_pickled("drawables", self.nd, self.version, sym_vars=list(self.q) + list(self.L) + list(self.a) + list(self.b), flags=self.get_flags())

        if saved is not None:
            self.ap, self.bp, self.jc, self.ee, self.bc = saved
        else:
            a, b, q, L = self.a, self.b, self.q, self.L
            if self.nd == 3:
                self.ap = [
                    [-a[0], sp.Rational(0)],
                    [a[1], sp.Rational(0)],
                    [a[2] * cos(q[0]), a[2] * sin(q[0])],
                    [a[3] * cos(q[0]), a[3] * sin(q[0])],
                    [-a[4], sp.Rational(0)],
                    [a[5], sp.Rational(0)],
                ]
                self.bp = [
                    [b[0] * cos(q[0]), b[0] * sin(q[0])],
                    [b[1] * cos(q[0]), b[1] * sin(q[0])],
                    [
                        L[0] * cos(q[0]) + b[2] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + b[2] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) - b[3] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) - b[3] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) + b[4] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + b[4] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) - b[5] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) - b[5] * sin(q[0] + q[1]),
                    ],
                ]
                self.ap.extend(
                    [
                        [
                            L[0] * cos(q[0]) + a[6] * cos(q[0] + q[1]),
                            L[0] * sin(q[0]) + a[6] * sin(q[0] + q[1]),
                        ],
                        [
                            L[0] * cos(q[0]) + a[7] * cos(q[0] + q[1]),
                            L[0] * sin(q[0]) + a[7] * sin(q[0] + q[1]),
                        ],
                        [a[8] * cos(q[0]), a[8] * sin(q[0])],
                    ]
                )
                self.bp.extend(
                    [
                        [
                            L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) + b[6] * cos(q[0] + q[1] + q[2]),
                            L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]) + b[6] * sin(q[0] + q[1] + q[2]),
                        ],
                        [
                            L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) - b[7] * cos(q[0] + q[1] + q[2]),
                            L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]) - b[7] * sin(q[0] + q[1] + q[2]),
                        ],
                        [
                            L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) + b[8] * cos(q[0] + q[1] + q[2]),
                            L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]) + b[8] * sin(q[0] + q[1] + q[2]),
                        ],
                    ]
                )
                self.jc = [
                    [sp.Rational(0), sp.Rational(0)],
                    [L[0] * cos(q[0]), L[0] * sin(q[0])],
                    [
                        L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]),
                    ],
                ]
                self.jc.append(
                    [
                        L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) + L[2] * cos(q[0] + q[1] + q[2]),
                        L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]) + L[2] * sin(q[0] + q[1] + q[2]),
                    ]
                )
                self.ee = sp.Matrix(
                    [
                        L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) + L[2] * cos(q[0] + q[1] + q[2]),
                        L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]) + L[2] * sin(q[0] + q[1] + q[2]),
                    ]
                )
                self.bc = [
                    [self.xc[0][0], self.xc[0][1]],
                    [self.xc[1][0], self.xc[1][1]],
                    [self.xc[2][0], self.xc[2][1]],
                ]
            else:
                self.ap = [
                    [-a[0], sp.Rational(0)],
                    [a[1], sp.Rational(0)],
                    [a[2] * cos(q[0]), a[2] * sin(q[0])],
                    [a[3] * cos(q[0]), a[3] * sin(q[0])],
                    [-a[4], sp.Rational(0)],
                    [a[5], sp.Rational(0)],
                ]
                self.bp = [
                    [b[0] * cos(q[0]), b[0] * sin(q[0])],
                    [b[1] * cos(q[0]), b[1] * sin(q[0])],
                    [
                        L[0] * cos(q[0]) + b[2] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + b[2] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) - b[3] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) - b[3] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) + b[4] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + b[4] * sin(q[0] + q[1]),
                    ],
                    [
                        L[0] * cos(q[0]) - b[5] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) - b[5] * sin(q[0] + q[1]),
                    ],
                ]
                self.jc = [
                    [sp.Rational(0), sp.Rational(0)],
                    [L[0] * cos(q[0]), L[0] * sin(q[0])],
                    [
                        L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]),
                    ],
                ]
                self.ee = sp.Matrix(
                    [
                        L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]),
                        L[0] * sin(q[0]) + L[1] * sin(q[0] + q[1]),
                    ]
                )
                self.bc = [
                    [self.xc[0][0], self.xc[0][1]],
                    [self.xc[1][0], self.xc[1][1]],
                ]
            save_pickled("drawables", self.nd, self.version, (self.ap, self.bp, self.jc, self.ee, self.bc), sym_vars=list(self.q) + list(self.L) + list(self.a) + list(self.b), flags=self.get_flags())

        self.logger.debug("Drawables constructed.")

    def __define_muscle_parameters(self):
        self.logger.debug("Defining muscle parameters.")
        params = self.model_parameters(q=self.reference_pose, in_deg=False)
        self.lm0 = self.lm.subs(params)
        self.RTDq = sp.derive_by_array(self.R.transpose(), self.Q())
        self.logger.debug("Muscle parameters defined.")

    def approx_muscle_length_parabolic(self, A_sym, B_sym, h, k, w):
        """
        Compute an approximate muscle length along a parabolic arc between attachment points A_sym and B_sym,
        with a bending offset that depends linearly on the joint coordinates.

        We define the q-dependent offset as:
            H(q) = h + k * (w^T * q)
        where w is a predetermined weight (or sensitivity) for the muscle.
        
        Then the muscle length is given by:
            L(q) = sqrt(D^2+H(q)^2)/2 + (D^2/(2*H(q)))*asinh(H(q)/D)
        where D is the Euclidean distance between A_sym and B_sym.
        
        This formulation is elegant in that H(q) is differentiable and its derivative at q = 0 is k*w.
        Thus, even when D is constant at q = 0, we obtain a nonzero derivative:
            dL/dq = (∂L/∂H)(k*w)  ≠ 0,  if k*w ≠ 0.
        
        Parameters:
        A_sym (sp.Matrix): 2×1 symbolic attachment point A(q).
        B_sym (sp.Matrix): 2×1 symbolic attachment point B(q).
        h (number or sp.Expr): Baseline bending offset.
        k (number): Modulation coefficient.
        w (number or sp.Expr): Weight (or sensitivity) for the q–dependence.
        
        Returns:
        sp.Expr: The symbolic muscle length.
        """
        # Define the effective bending offset as a linear function of q.
        # Here we assume that w is chosen appropriately (scalar or a suitable linear combination).
        H = h + k * w  # If q is a scalar or if w already represents w^T*q
        # If q is a vector and you wish to use a linear combination, you might instead do:
        # H = h + k * (sp.Matrix(self.q).dot(sp.Matrix(w)))
        
        # Compute the straight-line distance between attachment points.
        D = sp.sqrt((B_sym[0] - A_sym[0])**2 + (B_sym[1] - A_sym[1])**2)
        
        # Compute the muscle length using the parabolic formula.
        L_parabolic = sp.sqrt(D**2 + H**2)/2 + (D**2/(2*H)) * sp.asinh(H/D)
        
        return sp.simplify(L_parabolic)

    def construct_muscle_geometry(self, use_parabolic=False, h=0.05, k=0.05, use_alt_representation=False):
        """
        Construct muscle geometry for either 2-DoF (6 muscles) or 3-DoF (9 muscles).

        - If use_parabolic is False (the default), the muscle lengths are computed using
        a sqrt‐based formulation (or an alternate arcsin/arccos method if use_alt_representation=True).
        - If use_parabolic is True, the muscle lengths are computed via a parabolic approximation,
        using the helper approx_muscle_length_parabolic. In this updated version the bending offset
        is made variable by automatically detecting the joint variables affecting the muscle attachments,
        and using an effective offset h_expr = h + k * (sum of affecting joints).

        The computed symbolic expressions for lengths, their time derivatives, and the Jacobians are
        stored in self.lm, self.lmd, self.lmdd, self.R, self.RDot, and self.RDotQDot.
        """
        import sympy as sp
        self.logger.debug(f"Constructing muscle geometry (use_parabolic={use_parabolic}, h={h}, k={k}, use_alt_representation={use_alt_representation}).")
        
        # Create a caching flag that distinguishes between the two methods.
        flags = {"use_parabolic": use_parabolic, "h": h, "k": k, "use_alt": use_alt_representation, **self.get_flags()}
        saved_geom = load_pickled("muscle_geometry", self.nd, self.version, sym_vars=list(self.q), flags=flags)
        if saved_geom is not None:
            (self.lm, self.lmd, self.lmdd, self.R, self.RDot, self.RDotQDot) = saved_geom
            self.logger.debug("Loaded muscle geometry from file.")
        else:
            a, b, L, q = self.a, self.b, self.L, self.q
            cos = sp.cos
            sqrt = sp.sqrt
            acos = sp.acos
            if use_parabolic:
                # PARABOLIC APPROXIMATION with variable bending offset.
                # Ensure the attachment points (self.ap and self.bp) are computed.
                self.__construct_drawables()
                self.logger.debug("Attachment points (ap, bp) constructed for parabolic approximation.")
                lm_list = []
                for i in range(self.md):
                    A_sym = sp.Matrix(self.ap[i])
                    B_sym = sp.Matrix(self.bp[i])
                    # Compute the parabolic muscle length using the new helper function,
                    # which automatically detects the affecting joints.
                    if i < self.nd:
                        w_i = self.q[i]  # or another predetermined nonzero value
                    else:
                         #w_i = (self.q[0] + self.q[1]) / 2
                         w_i = self.q[0] + self.q[1]

                    L_expr = self.approx_muscle_length_parabolic(A_sym, B_sym, h, k, w_i)
                    
                    lm_list.append(L_expr)
                self.lm = sp.Matrix(lm_list)
            else:
                # DEFAULT GEOMETRY (sqrt‐based or alternate)
                if self.nd == 3:
                    if not use_alt_representation:
                        self.lm = sp.Matrix([
                            sqrt(a[0]**2 + b[0]**2 + 2*a[0]*b[0]*cos(q[0])),
                            sqrt(a[1]**2 + b[1]**2 - 2*a[1]*b[1]*cos(q[0])),
                            sqrt((L[0]-a[2])**2 + b[2]**2 + 2*(L[0]-a[2])*b[2]*cos(q[1])),
                            sqrt((L[0]-a[3])**2 + b[3]**2 - 2*(L[0]-a[3])*b[3]*cos(q[1])),
                            sqrt(a[4]**2 + b[4]**2 + L[0]**2 + 2*a[4]*L[0]*cos(q[0]) +
                                2*b[4]*L[0]*cos(q[1]) + 2*a[4]*b[4]*cos(q[0]+q[1])),
                            sqrt(a[5]**2 + b[5]**2 + L[0]**2 - 2*a[5]*L[0]*cos(q[0]) -
                                2*b[5]*L[0]*cos(q[1]) + 2*a[5]*b[5]*cos(q[0]+q[1])),
                            sqrt((L[1]-a[6])**2 + b[6]**2 + 2*(L[1]-a[6])*b[6]*cos(q[2])),
                            sqrt((L[1]-a[7])**2 + b[7]**2 - 2*(L[1]-a[7])*b[7]*cos(q[2])),
                            sqrt((L[0]-a[8])**2 + b[8]**2 + L[1]**2 + 2*(L[0]-a[8])*L[1]*cos(q[1]) +
                                2*b[8]*L[1]*cos(q[2]) + 2*(L[0]-a[8])*b[8]*cos(q[0]+q[1]))
                        ])
                    else:
                        self.lm = sp.Matrix([
                            -a[0]*q[0] + sqrt(b[0]**2 - a[0]**2) + a[0]*(sp.pi - acos(a[0]/b[0])),
                            -a[1]*q[0] + sqrt(b[1]**2 - a[1]**2) + a[1]*(sp.pi/2 - acos(a[1]/b[1])),
                            -a[2]*q[1] + sqrt(b[2]**2 - a[2]**2) + a[2]*(sp.pi - acos(a[2]/b[2])),
                            -a[3]*q[1] + sqrt(b[3]**2 - a[3]**2) + a[3]*(sp.pi/2 - acos(a[3]/b[3])),
                            -a[4]*(q[0]+q[1]) + sqrt(b[4]**2 - a[4]**2) + a[4]*(sp.pi - acos(a[4]/b[4])),
                            a[5]*(q[0]+q[1]) + sqrt(b[5]**2 - a[5]**2) + a[5]*(sp.pi/2 - acos(a[5]/b[5])),
                            -a[6]*q[2] + sqrt(b[6]**2 - a[6]**2) + a[6]*(sp.pi - acos(a[6]/b[6])),
                            -a[7]*q[2] + sqrt(b[7]**2 - a[7]**2) + a[7]*(sp.pi/2 - acos(a[7]/b[7])),
                            a[8]*(q[1]+q[2]) + sqrt(b[8]**2 - a[8]**2) + a[8]*(sp.pi - acos(a[8]/b[8]))
                        ])
                else:
                    if not use_alt_representation:
                        self.lm = sp.Matrix([
                            sqrt(a[0]**2 + b[0]**2 + 2*a[0]*b[0]*cos(q[0])),
                            sqrt(a[1]**2 + b[1]**2 - 2*a[1]*b[1]*cos(q[0])),
                            sqrt((L[0]-a[2])**2 + b[2]**2 + 2*(L[0]-a[2])*b[2]*cos(q[1])),
                            sqrt((L[0]-a[3])**2 + b[3]**2 - 2*(L[0]-a[3])*b[3]*cos(q[1])),
                            sqrt(a[4]**2 + b[4]**2 + L[0]**2 + 2*a[4]*L[0]*cos(q[0]) +
                                2*b[4]*L[0]*cos(q[1]) + 2*a[4]*b[4]*cos(q[0]+q[1])),
                            sqrt(a[5]**2 + b[5]**2 + L[0]**2 - 2*a[5]*L[0]*cos(q[0]) -
                                2*b[5]*L[0]*cos(q[1]) + 2*a[5]*b[5]*cos(q[0]+q[1]))
                        ])
                    else:
                        self.lm = sp.Matrix([
                            -a[0]*q[0] + sqrt(b[0]**2 - a[0]**2) + a[0]*(sp.pi - acos(a[0]/b[0])),
                            -a[1]*q[0] + sqrt(b[1]**2 - a[1]**2) + a[1]*(sp.pi/2 - acos(a[1]/b[1])),
                            -a[2]*q[1] + sqrt(b[2]**2 - a[2]**2) + a[2]*(sp.pi - acos(a[2]/b[2])),
                            -a[3]*q[1] + sqrt(b[3]**2 - a[3]**2) + a[3]*(sp.pi/2 - acos(a[3]/b[3])),
                            -a[4]*(q[0]+q[1]) + sqrt(b[4]**2 - a[4]**2) + a[4]*(sp.pi - acos(a[4]/b[4])),
                            a[5]*(q[0]+q[1]) + sqrt(b[5]**2 - a[5]**2) + a[5]*(sp.pi/2 - acos(a[5]/b[5]))
                        ])
                self.lm = self.lm.subs(self.constants)
            # Compute time derivatives and Jacobians.
            self.lmd = sp.diff(self.lm, self.t)
            self.lmdd = sp.diff(self.lmd, self.t)
            #print(self.lm)
            #(self.Q())
            self.R = self.lm.jacobian(self.Q())
            self.RDot = sp.diff(self.R, self.t)
            self.RDotQDot = self.RDot * sp.Matrix(self.QDot())

            self.lm = self.lm.subs(self.constants)
            self.lmd = self.lmd.subs(self.constants)
            self.lmdd = self.lmdd.subs(self.constants)
            self.R = self.R.subs(self.constants)
            self.RDot = self.RDot.subs(self.constants)
            self.RDotQDot = self.RDotQDot.subs(self.constants)

            # Cache the computed geometry.
            save_pickled("muscle_geometry", self.nd, self.version,
                        (self.lm, self.lmd, self.lmdd, self.R, self.RDot, self.RDotQDot),
                        sym_vars=list(self.q), flags=flags)
            self.logger.debug("Muscle geometry constructed and saved.")
        # Lambdify the resulting expressions using pure sympy.
        self.muscle_lengths_func = lambdify(self.Q(), self.lm, modules="sympy")
        self.muscle_velocities_func = lambdify((self.Q(), self.QDot()), self.lmd, modules="sympy")
        self.muscle_acceleration_func = lambdify((self.Q(), self.QDot(), self.ddq), self.lmdd, modules="sympy")
        self.R_func = lambdify(self.Q(), self.R, modules="sympy")
        self.RDot_func = lambdify((self.Q(), self.QDot()), self.RDot, modules="sympy")
        self.RDotQDot_func = lambdify((self.Q(), self.QDot()), self.RDotQDot, modules="sympy")

    def __construct_rhs(self):
        self.logger.debug("Constructing RHS (dynamics function).")
        saved_rhs = load_pickled("rhs", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if saved_rhs is not None:
            (
                self.f,
                self.forcing,
                self.coordinates,
                self.speeds,
                self.coordinates_derivatives,
                self.specifieds,
                self.rhs,
                self.tau_b,
            ) = saved_rhs
            self.tau_l = sp.Matrix(apply_generalized_force(self.__tau_l))
            self.logger.debug("Loaded RHS from file.")
        else:
            b_damping = 0.05
            tau = sp.Matrix(apply_generalized_force(self.Tau()))
            # If you wish, you can also multiply your coordinate-limiting force here:
            self.tau_l = self.use_coordinate_limits * sp.Matrix(apply_generalized_force(self.__tau_l))
            # Scale the damping force by use_viscosity
            self.tau_b = self.use_viscosity * (-b_damping * sp.Matrix(apply_generalized_force(self.QDot())))
            # Then, build f without re-multiplying:
            self.f = self.tau_c + self.tau_g - self.tau_l - self.tau_b

            self.forcing = tau
            for i in range(self.forcing.shape[0]):
                self.forcing = self.forcing.subs(self.dq[i], self.u[i])
            self.coordinates = sp.Matrix(self.Q())
            self.speeds = sp.Matrix(self.QDot())
            self.coordinates_derivatives = self.speeds
            self.specifieds = sp.Matrix(self.Tau())
            self.rhs = generate_ode_function(
                self.forcing,
                self.coordinates,
                self.speeds,
                list(self.constants.keys()),
                mass_matrix=self.M,
                coordinate_derivatives=self.coordinates_derivatives,
                specifieds=self.specifieds,
            )
            save_pickled(
                "rhs",
                self.nd,
                self.version,
                (
                    self.f,
                    self.forcing,
                    self.coordinates,
                    self.speeds,
                    self.coordinates_derivatives,
                    self.specifieds,
                    self.rhs,
                    self.tau_b,
                ),
                sym_vars=list(self.q),
                flags=self.get_flags(),
            )

            self.logger.debug("RHS constructed and saved.")

    def __construct_models_Kinematics(self):
        self.logger.debug("Constructing kinematics models.")
        if self.nd == 2:
            x_ee = self.L[0] * cos(self.q[0]) + self.L[1] * cos(self.q[0] + self.q[1])
            y_ee = self.L[0] * sin(self.q[0]) + self.L[1] * sin(self.q[0] + self.q[1])
            self._fk_expr = sp.Matrix([sp.trigsimp(x_ee), sp.trigsimp(y_ee)])
            x_d, y_d = sp.symbols("x_d y_d", real=True)
            L1, L2 = self.L[0], self.L[1]
            D = sp.trigsimp((x_d**2 + y_d**2 - L1**2 - L2**2) / (2 * L1 * L2))
            D_clip = sp.Min(1, sp.Max(-1, D))
            q2_sol = sp.acos(D_clip)
            q1_sol = sp.atan2(y_d, x_d) - sp.atan2(L2 * sp.sin(q2_sol), L1 + L2 * sp.cos(q2_sol))
            self._ik_expr = {
                "q1": sp.trigsimp(q1_sol),
                "q2": sp.trigsimp(q2_sol),
                "x_d": x_d,
                "y_d": y_d,
            }
            save_pickled("ik_expr", self.nd, self.version, self._ik_expr, sym_vars=list(self.q), flags=self.get_flags())
        elif self.nd == 3:
            x_ee = self.L[0] * cos(self.q[0]) + self.L[1] * cos(self.q[0] + self.q[1]) + self.L[2] * cos(self.q[0] + self.q[1] + self.q[2])
            y_ee = self.L[0] * sin(self.q[0]) + self.L[1] * sin(self.q[0] + self.q[1]) + self.L[2] * sin(self.q[0] + self.q[1] + self.q[2])
            self._fk_expr = sp.Matrix([sp.trigsimp(x_ee), sp.trigsimp(y_ee)])
            self._ik_expr = None
            save_pickled("ik_expr", self.nd, self.version, self._ik_expr, sym_vars=list(self.q), flags=self.get_flags())
        self.logger.debug("Kinematics models constructed.")

    def __construct_models_jacobianderivatives(self):
        self.logger.debug("Constructing Jacobian and its derivatives.")
        saved_jac = load_pickled("jacobian", self.nd, self.version, sym_vars=list(self.q), flags=self.get_flags())
        if saved_jac is not None:
            self.J_symbolic, self.J_dot_symbolic, self.J_ddot_symbolic = saved_jac
            self.logger.debug("Loaded Jacobian derivatives from file.")
        else:
            self.J_symbolic = sp.trigsimp(self._fk_expr.jacobian(self.Q()))
            self.J_dot_symbolic = sp.zeros(self.J_symbolic.shape[0], self.J_symbolic.shape[1])
            for i, q_sym in enumerate(self.Q()):
                self.J_dot_symbolic += self.J_symbolic.diff(q_sym) * self.QDot()[i]
            self.J_dot_symbolic = sp.trigsimp(self.J_dot_symbolic)
            self.J_ddot_symbolic = sp.diff(self.J_dot_symbolic, self.t)
            save_pickled("jacobian", self.nd, self.version, (self.J_symbolic, self.J_dot_symbolic, self.J_ddot_symbolic), sym_vars=list(self.q), flags=self.get_flags())
            self.logger.debug("Jacobian derivatives constructed and saved.")

    def jacobian(self, q_vals=None):
        self.logger.debug("Calculating Jacobian.")
        J_sym = sp.trigsimp(self._fk_expr.jacobian(self.Q()))
        if q_vals is not None:
            subs_dict = self.model_parameters(q=q_vals, in_deg=False)
            result = sp.trigsimp(J_sym.subs(subs_dict))
        else:
            result = J_sym
        self.logger.debug("Jacobian calculated.")
        return result

    def jacobian_dot(self, q_vals=None, q_dot_vals=None):
        self.logger.debug("Calculating Jacobian dot.")
        J = self.jacobian(q_vals)
        q_syms = self.Q()
        if q_dot_vals is None:
            q_dot_expr = self.QDot()
        else:
            q_dot_expr = sp.Matrix(q_dot_vals)
        J_dot = sp.zeros(J.shape[0], J.shape[1])
        for i in range(len(q_syms)):
            J_dot += J.diff(q_syms[i]) * q_dot_expr[i]
        result = sp.trigsimp(J_dot)
        self.logger.debug("Jacobian dot calculated.")
        return result

    def jacobian_dotdot(self, q_vals=None, q_dot_vals=None, q_ddot_vals=None):
        self.logger.debug("Calculating Jacobian double dot.")
        J_dot = self.jacobian_dot(q_vals, q_dot_vals)
        if q_ddot_vals is None:
            J_ddot = sp.diff(J_dot, self.t)
        else:
            J_ddot = sp.diff(J_dot, self.t)
            subs_dict = {}
            for i, q_sym in enumerate(self.Q()):
                subs_dict[sp.diff(q_sym, self.t, self.t)] = q_ddot_vals[i]
            J_ddot = J_ddot.subs(subs_dict)
        result = sp.trigsimp(J_ddot)
        self.logger.debug("Jacobian double dot calculated.")
        return result

    def damped_pinv(self, J, damping=0.01):
        self.logger.debug("Calculating damped pseudoinverse of a matrix.")
        J_torch = torch.as_tensor(J, dtype=torch.float32, device=device)
        result = damped_pinv_torch(J_torch, damping).cpu().numpy()
        self.logger.debug("Calculated damped pseudoinverse.")
        return result

    def forward_kinematics(self, q):
        self.logger.debug("Calculating forward kinematics.")
        pos = self.fk_func(*q)
        pos_arr = np.array(pos, dtype=float).flatten()
        result = (float(pos_arr[0]), float(pos_arr[1]))
        self.logger.debug(f"Forward kinematics result: {result}")
        return result

    def inverse_kinematics(self, pos, phi=None, elbow_up=True):
        self.logger.debug("Calculating inverse kinematics.")
        if self.nd == 2:
            x, y = pos
            subs_dict = {self._ik_expr["x_d"]: x, self._ik_expr["y_d"]: y}
            q1_val = float(self._ik_expr["q1"].subs(subs_dict).evalf())
            q2_val = float(self._ik_expr["q2"].subs(subs_dict).evalf())
            result = (q1_val, q2_val)
            self.logger.debug(f"Inverse kinematics result (2R): {result}")
            return result
        elif self.nd == 3:
            if phi is not None:
                params = self.model_parameters(in_deg=False)
                L1_val = float(params[self.L[0]])
                L2_val = float(params[self.L[1]])
                L3_val = float(params[self.L[2]])
                x, y = pos
                x_w = x - L3_val * np.cos(phi)
                y_w = y - L3_val * np.sin(phi)
                r = np.hypot(x_w, y_w)
                cos_q2 = (r**2 - L1_val**2 - L2_val**2) / (2 * L1_val * L2_val)
                cos_q2 = np.clip(cos_q2, -1, 1)
                sin_q2 = np.sqrt(1 - cos_q2**2) if elbow_up else -np.sqrt(1 - cos_q2**2)
                q2_val = np.arctan2(sin_q2, cos_q2)
                q1_val = np.arctan2(y_w, x_w) - np.arctan2(L2_val * np.sin(q2_val), L1_val + L2_val * np.cos(q2_val))
                q3_val = phi - q1_val - q2_val
                result = (q1_val, q2_val, q3_val)
                self.logger.debug(f"Inverse kinematics result (3R): {result}")
                return result
            else:
                raise ValueError("For a 3-DoF arm, please provide a desired overall orientation phi.")
        else:
            raise NotImplementedError("Inverse kinematics is only implemented for 2 and 3 DoF models.")

    def forward_speed_kinematics(self, q, q_dot):
        self.logger.debug("Calculating forward speed kinematics.")
        result = self.forward_speed_kinematics_lambdified(q, q_dot)
        self.logger.debug("Calculated forward speed kinematics.")
        return result

    def inverse_speed_kinematics(self, q, v_des, damping=0.01):
        self.logger.debug("Calculating inverse speed kinematics.")
        result = self.inverse_speed_kinematics_lambdified(q, v_des, damping)
        self.logger.debug("Calculated inverse speed kinematics.")
        return result

    def forward_dynamics(self, state, torque):
        self.logger.debug("Calculating forward dynamics.")
        result = self.forward_dynamics_lambdified(state, torque)
        self.logger.debug("Calculated forward dynamics.")
        return result

    def inverse_dynamics(self, state, desired_acc):
        self.logger.debug("Calculating inverse dynamics.")
        result = self.inverse_dynamics_lambdified(state, desired_acc)
        self.logger.debug("Calculated inverse dynamics.")
        return result

    # --------------------------------------------------------------------
    # Lambdified Functions Setup
    # --------------------------------------------------------------------

    def __initialize_lambdified_functions(self) -> None:
        self.logger.debug("Initializing lambdified functions.")

        # Prepare standard symbolic variables for lambdifying.
        Q = self.Q()  # [theta0(t), theta1(t), (theta2(t))]
        QDot = self.QDot()  # [dtheta0(t), dtheta1(t), (dtheta2(t))]
        QDDot = self.ddq  # [ddtheta0(t), ddtheta1(t), (ddtheta2(t))] if nd=3

        # -------------------------------------------------------------
        # 1) Mass Matrix
        # -------------------------------------------------------------
        self.mass_matrix_func = load_pickled("mass_matrix_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())

        if self.mass_matrix_func is None:
            self.logger.debug("Mass matrix function not found; generating new one.")
            self.mass_matrix_func = lambdify(Q, self.M, modules="numpy")
            save_pickled("mass_matrix_func", self.nd, self.version, self.mass_matrix_func, sym_vars=list(Q), flags=self.get_flags())

        # -------------------------------------------------------------
        # 2) Coriolis / Centrifugal Forces
        # -------------------------------------------------------------
        self.coriolis_forces_func = load_pickled("coriolis_forces_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())
        if self.coriolis_forces_func is None:
            self.logger.debug("Coriolis forces function not found; generating new one.")
            self.coriolis_forces_func = lambdify((Q, QDot), self.tau_c, modules="numpy")
            save_pickled("coriolis_forces_func", self.nd, self.version, self.coriolis_forces_func, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        # -------------------------------------------------------------
        # 3) Gravitational Forces
        # -------------------------------------------------------------
        self.gravitational_forces_func = load_pickled("gravitational_forces_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.gravitational_forces_func is None:
            self.logger.debug("Gravitational forces function not found; generating new one.")
            self.gravitational_forces_func = lambdify(Q, self.tau_g, modules="numpy")
            save_pickled("gravitational_forces_func", self.nd, self.version, self.gravitational_forces_func, sym_vars=list(Q), flags=self.get_flags())

        # -------------------------------------------------------------
        # 4) Muscle Lengths (lm)
        # -------------------------------------------------------------
        self.muscle_lengths_func = load_pickled("muscle_lengths_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.muscle_lengths_func is None:
            self.logger.debug("Muscle lengths function not found; generating new one.")
            self.muscle_lengths_func = lambdify(Q, self.lm, modules="numpy")
            save_pickled("muscle_lengths_func", self.nd, self.version, self.muscle_lengths_func, sym_vars=list(Q), flags=self.get_flags())

        # --- NEW BLOCK 1: muscle velocity (lmd) ---
        self.muscle_velocities_func = load_pickled("muscle_velocities_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())
        if self.muscle_velocities_func is None:
            self.logger.debug("Muscle velocity function not found; generating new one.")
            # lmd depends on (q, q_dot)
            self.muscle_velocities_func = lambdify((Q, QDot), self.lmd, modules="numpy")
            save_pickled("muscle_velocities_func", self.nd, self.version, self.muscle_velocities_func, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        # --- NEW BLOCK 2: muscle acceleration (lmdd) ---
        self.muscle_acceleration_func = load_pickled("muscle_acceleration_func", self.nd, self.version, sym_vars=list(Q) + list(QDot) + list(QDDot), flags=self.get_flags())
        if self.muscle_acceleration_func is None:
            self.logger.debug("Muscle acceleration function not found; generating new one.")
            # lmdd depends on (q, q_dot, q_ddot)
            self.muscle_acceleration_func = lambdify((Q, QDot, QDDot), self.lmdd, modules="numpy")
            save_pickled("muscle_acceleration_func", self.nd, self.version, self.muscle_acceleration_func, sym_vars=list(Q) + list(QDot) + list(QDDot), flags=self.get_flags())

        # --- NEW BLOCK 3: muscle Jacobian R ---
        self.R_func = load_pickled("R_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.R_func is None:
            self.logger.debug("R (muscle Jacobian) function not found; generating new one.")
            self.R_func = lambdify(Q, self.R, modules="numpy")
            save_pickled("R_func", self.nd, self.version, self.R_func, sym_vars=list(Q), flags=self.get_flags())

        # --- NEW BLOCK 4: muscle Jacobian time derivative (RDot) ---
        self.RDot_func = load_pickled("RDot_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())
        if self.RDot_func is None:
            self.logger.debug("RDot function not found; generating new one.")
            self.RDot_func = lambdify((Q, QDot), self.RDot, modules="numpy")
            save_pickled("RDot_func", self.nd, self.version, self.RDot_func, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        # --- NEW BLOCK 5: joint-limit torque (tau_l) ---
        self.tau_l_func = load_pickled("tau_l_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.tau_l_func is None:
            self.logger.debug("tau_l function not found; generating new one.")
            # Usually depends on q only (if purely position-based limits)
            self.tau_l_func = lambdify(Q, self.tau_l, modules="numpy")
            save_pickled("tau_l_func", self.nd, self.version, self.tau_l_func, sym_vars=list(Q), flags=self.get_flags())

        # --- NEW BLOCK 6: damping torque (tau_b) ---
        # If your damping is only velocity-based, depends on QDot.
        self.tau_b_func = load_pickled("tau_b_func", self.nd, self.version, sym_vars=list(QDot), flags=self.get_flags())
        if self.tau_b_func is None:
            self.logger.debug("tau_b function not found; generating new one.")
            self.tau_b_func = lambdify(QDot, self.tau_b, modules="numpy")
            save_pickled("tau_b_func", self.nd, self.version, self.tau_b_func, sym_vars=list(QDot), flags=self.get_flags())

        # --- NEW BLOCK for RDotQDot (time derivative of R times q_dot) ---
        self.RDotQDot_func = load_pickled("RDotQDot_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())
        if self.RDotQDot_func is None:
            self.logger.debug("RDotQDot function not found; generating new one.")
            self.RDotQDot_func = lambdify((Q, QDot), self.RDotQDot, modules="numpy")
            save_pickled("RDotQDot_func", self.nd, self.version, self.RDotQDot_func, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        # -------------------------------------------------------------
        # 5) Forward Kinematics (already present)
        # -------------------------------------------------------------
        self.fk_func = load_pickled("fk_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.fk_func is None:
            self.logger.debug("Forward kinematics function not found; generating new one.")
            self.fk_func = lambdify(Q, self._fk_expr, modules="numpy")
            save_pickled("fk_func", self.nd, self.version, self.fk_func, sym_vars=list(Q), flags=self.get_flags())

        # IK function for 2R or 3R
        if self.nd == 2:
            x_d, y_d = sp.symbols("x_d y_d", real=True)
            self.ik_func = lambdify((x_d, y_d), (self._ik_expr["q1"], self._ik_expr["q2"]), modules="numpy")
        else:
            self.ik_func = lambda x, y, phi, elbow_up=True: self.inverse_kinematics((x, y), phi, elbow_up)

        # -------------------------------------------------------------
        # 6) End-Effector Jacobian J(q), J_dot, J_ddot
        # -------------------------------------------------------------
        self.J_func = load_pickled("J_func", self.nd, self.version, sym_vars=list(Q), flags=self.get_flags())
        if self.J_func is None:
            self.logger.debug("Jacobian function not found; generating new one.")
            J_sym = sp.trigsimp(self._fk_expr.jacobian(Q))
            self.J_symbolic = J_sym
            self.J_func = lambdify(Q, J_sym, modules="numpy")
            save_pickled("J_func", self.nd, self.version, self.J_func, sym_vars=list(Q), flags=self.get_flags())

        self.J_dot_func = load_pickled("J_dot_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())
        if self.J_dot_func is None:
            self.logger.debug("Jacobian dot function not found; generating new one.")
            J_dot_sym = sp.zeros(self.J_symbolic.shape[0], self.J_symbolic.shape[1])
            for i, q_sym in enumerate(Q):
                J_dot_sym += self.J_symbolic.diff(q_sym) * QDot[i]
            J_dot_sym = sp.trigsimp(J_dot_sym)
            self.J_dot_symbolic = J_dot_sym
            self.J_dot_func = lambdify((Q, QDot), J_dot_sym, modules="numpy")
            save_pickled("J_dot_func", self.nd, self.version, self.J_dot_func, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        self.J_ddot_func = load_pickled("J_ddot_func", self.nd, self.version, sym_vars=list(Q) + list(QDot) + list(QDDot), flags=self.get_flags())
        if self.J_ddot_func is None:
            self.logger.debug("Jacobian double dot function not found; generating new one.")
            self.J_ddot_func = lambdify((Q, QDot, QDDot), self.jacobian_dotdot(), modules="numpy")
            save_pickled("J_ddot_func", self.nd, self.version, self.J_ddot_func, sym_vars=list(Q) + list(QDot) + list(QDDot), flags=self.get_flags())

        # -------------------------------------------------------------
        # 7) Forward Speed Kinematics (EE velocity)
        # -------------------------------------------------------------
        # v_sym = J(q) * qdot
        v_sym = self.J_symbolic * sp.Matrix(QDot)
        self.forward_speed_kinematics_lambdified = load_pickled("forward_speed_kinematics_func", self.nd, self.version, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        if self.forward_speed_kinematics_lambdified is None:
            self.logger.debug("Forward speed kinematics function not found; generating new one.")
            self.forward_speed_kinematics_lambdified = lambdify((Q, QDot), v_sym, modules="numpy")
            save_pickled("forward_speed_kinematics_func", self.nd, self.version, self.forward_speed_kinematics_lambdified, sym_vars=list(Q) + list(QDot), flags=self.get_flags())

        # -------------------------------------------------------------
        # 8) Misc: inverse_speed_kinematics, forward_dynamics, etc.
        # -------------------------------------------------------------
        # Inverse speed kinematics uses the damped pseudoinverse (PyTorch-based).
        self.inverse_speed_kinematics_lambdified = lambda q, v_des, damping=0.01: (
            damped_pinv_torch(
                torch.as_tensor(self.J_func(*q), dtype=torch.float32, device=device),
                damping,
            )
            @ torch.as_tensor(v_des, dtype=torch.float32, device=device)
        )

        # Forward dynamics lambda (PyTorch-based)
        self.forward_dynamics_lambdified = lambda state, torque: torch.cat(
            (
                torch.as_tensor(state[: self.nd], dtype=torch.float32, device=device).flatten(),
                torch.linalg.solve(
                    torch.as_tensor(
                        self.mass_matrix_func(*state[: self.nd]),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(torque, dtype=torch.float32, device=device).clone().detach()
                    - torch.as_tensor(
                        self.coriolis_forces_func(state[: self.nd], state[self.nd : 2 * self.nd]),
                        dtype=torch.float32,
                        device=device,
                    )
                    .clone()
                    .detach()
                    - torch.as_tensor(
                        self.gravitational_forces_func(*state[: self.nd]),
                        dtype=torch.float32,
                        device=device,
                    )
                    .clone()
                    .detach(),
                ).flatten(),
            )
        )

        # Inverse dynamics lambda (PyTorch-based)
        self.inverse_dynamics_lambdified = lambda state, desired_acc: (
            torch.as_tensor(
                self.mass_matrix_func(*state[: self.nd]),
                dtype=torch.float32,
                device=device,
            )
            @ torch.as_tensor(desired_acc, dtype=torch.float32, device=device)
            + torch.as_tensor(
                self.coriolis_forces_func(state[: self.nd], state[self.nd : 2 * self.nd]),
                dtype=torch.float32,
                device=device,
            )
            + torch.as_tensor(
                self.gravitational_forces_func(*state[: self.nd]),
                dtype=torch.float32,
                device=device,
            )
        )

        self.logger.debug("Lambdified functions initialized.")

    # --------------------------------------------------------------------
    # New Method: draw_model
    # This method draws the robot configuration using Matplotlib.
    # --------------------------------------------------------------------

    def collect_trajectory_info(
        self,
        q,
        q_dot,
        x_des,
        step,
        dt,
        initialize=False,
        ddq=None,
        torque=None,
        jerk=None,
        condJ_val=None,
    ):
        """
        Collects the trajectory and info data.
        If initialize=True, re-initializes self.trajectory and self.last_info.

        Parameters:
        q, q_dot : current joint angles and velocities (numpy arrays)
        x_des    : target in end-effector space (2D or 3D)
        ddq      : optional joint accelerations
        torque   : optional torque array
        jerk     : optional joint jerk array
        condJ_val: optional condition number of the Jacobian
        """
        if initialize:
            self.trajectory = {
                "time": [],
                "end_effector": [],
                "joint_angles": [],
                "joint_velocities": [],
                "joint_accelerations": [],
                "joint_jerks": [],
                "torques": [],
                "pos_error": [],
                "reward": [],
                "condJ": [],
                "target": [],
            }
            self.last_info = {}

        # 1) Record time.
        self.trajectory["time"].append(step * dt)

        # 2) Record end-effector position.
        ee_pos = self.fk_func(*q).flatten()
        self.trajectory["end_effector"].append(ee_pos)

        # 3) Record joint angles and velocities.
        self.trajectory["joint_angles"].append(q.copy())
        self.trajectory["joint_velocities"].append(q_dot.copy())

        # 4) Record accelerations, jerks, and torques (or use zeros if not provided).
        self.trajectory["joint_accelerations"].append(ddq.copy() if ddq is not None else np.zeros_like(q_dot))
        self.trajectory["joint_jerks"].append(jerk.copy() if jerk is not None else np.zeros_like(q_dot))
        self.trajectory["torques"].append(torque.copy() if torque is not None else np.zeros_like(q_dot))

        # 5) Record position error and reward.
        pos_err = np.linalg.norm(x_des - ee_pos)
        self.trajectory["pos_error"].append(pos_err)
        self.trajectory["reward"].append(0)

        # 6) Record Jacobian condition number and target.
        # self.trajectory["condJ"].append(condJ_val if condJ_val is not None else 0)
        # self.trajectory["target"].append(x_des)

        # 7) Update last_info for display in draw_model.
        if ddq is not None:
            self.last_info["current_state"] = np.concatenate([q, q_dot, ddq])
        else:
            self.last_info["current_state"] = np.concatenate([q, q_dot, np.zeros_like(q_dot)])
        self.last_info["target"] = x_des
        self.last_info["end_effector"] = ee_pos
        self.last_info["pos_error"] = pos_err
        self.last_info["condJ"] = condJ_val if condJ_val is not None else 0
        self.last_info["torque"] = torque if torque is not None else np.zeros_like(q_dot)
        self.last_info["joint_jerk_cmd"] = jerk if jerk is not None else np.zeros_like(q_dot)

    def play_trajectory(self, playback_duration=1.0, decimation=1):
        """
        Incrementally plots the recorded trajectory and joint signals without using
        the animation module. The figure is updated in a loop, and once all data
        has been plotted, the function waits for an additional delay before automatically closing the figure.

        Returns:
            bool: True if trajectory data was available.
        """
        # Decimate recorded data.
        time_arr = np.array(self.trajectory["time"])[::decimation]
        if len(time_arr) == 0:
            print("No trajectory data recorded.")
            return False

        ee_traj = np.array(self.trajectory["end_effector"])[::decimation]
        joint_angles = np.array(self.trajectory["joint_angles"])[::decimation]
        joint_velocities = np.array(self.trajectory["joint_velocities"])[::decimation]
        joint_accelerations = np.array(self.trajectory["joint_accelerations"])[::decimation]
        torques = np.array(self.trajectory["torques"])[::decimation]

        # Determine target.
        if hasattr(self, "x_target"):
            target = self.x_target
        elif "target" in self.last_info:
            target = self.last_info["target"]
        else:
            target = None

        num_joints = joint_angles.shape[1]
        fig, axs = plt.subplots(5, 1, figsize=(12, 15))
        plt.tight_layout()

        num_frames = len(time_arr)
        # Calculate interval per frame (in seconds)
        frame_interval = playback_duration / num_frames if num_frames > 0 else 0.05
        print(f"[Trajectory] Incrementally plotting {num_frames} frames over {playback_duration} seconds.")

        for frame in range(num_frames):
            idx = frame + 1
            # Update each subplot.
            axs[0].clear()
            axs[0].plot(ee_traj[:idx, 0], ee_traj[:idx, 1], "b.-", label="EE Trajectory")
            if target is not None:
                axs[0].plot(target[0], target[1], "rx", markersize=10, label="Target")
            axs[0].set_title("End-Effector Trajectory")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].clear()
            for j in range(num_joints):
                axs[1].plot(time_arr[:idx], joint_angles[:idx, j], label=f"θ{j+1}")
            axs[1].set_title("Joint Angles")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Angle (rad)")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].clear()
            for j in range(num_joints):
                axs[2].plot(time_arr[:idx], joint_velocities[:idx, j], label=f"θ{j+1}_dot")
            axs[2].set_title("Joint Velocities")
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Velocity (rad/s)")
            axs[2].legend()
            axs[2].grid(True)

            axs[3].clear()
            for j in range(num_joints):
                axs[3].plot(time_arr[:idx], joint_accelerations[:idx, j], label=f"θ{j+1}_ddot")
            axs[3].set_title("Joint Accelerations")
            axs[3].set_xlabel("Time (s)")
            axs[3].set_ylabel("Acceleration (rad/s²)")
            axs[3].legend()
            axs[3].grid(True)

            axs[4].clear()
            for j in range(num_joints):
                axs[4].plot(time_arr[:idx], torques[:idx, j], label=f"τ{j+1}")
            axs[4].set_title("Joint Torques")
            axs[4].set_xlabel("Time (s)")
            axs[4].set_ylabel("Torque (N·m)")
            axs[4].legend()
            axs[4].grid(True)

            # Force update and pause for the frame interval.
            plt.draw()
            plt.pause(frame_interval)

        # Extra delay after plotting all frames.
        extra_delay = 3  # seconds extra; adjust as needed
        print(f"[Trajectory] Finished plotting. Waiting an extra {extra_delay} seconds before auto-close.")
        plt.pause(extra_delay)
        plt.close(fig)
        return True

    def draw_model(
        self,
        q,
        in_deg=True,
        info=None,
        scale=0.8,
        use_axis_limits=True,
        alpha=1.0,
        text=True,
        play_trajectory_flag=False,
        playback_duration=1.0,
        decimation=1,
    ):
        """
        Draws the robot configuration using a reusable figure.
        Optionally displays variable information if provided via the `info` parameter.
        If a target is provided, draws a green ring with a small red dot at the target.

        New Args:
            play_trajectory_flag (bool): If True, calls play_trajectory after drawing,
                waits for the trajectory to finish plus an extra delay, then closes the figure.
            playback_duration (float): Duration used for playback if play_trajectory_flag is True.
            decimation (int): Decimation factor for trajectory data.
        """
        info = info or {}

        # 1) Perform symbolic substitution.
        consts = self.model_parameters(q=q, in_deg=in_deg)
        joints = np.array([[float(sp.N(el, 10, chop=True)) for el in row] for row in substitute(self.jc, consts)])
        muscle_a = np.array([[float(sp.N(el, 10, chop=True)) for el in row] for row in substitute(self.ap, consts)])
        muscle_b = np.array([[float(sp.N(el, 10, chop=True)) for el in row] for row in substitute(self.bp, consts)])
        end_effector = np.array([float(sp.N(el, 10, chop=True)) for el in substitute(self.ee, consts)]).flatten()
        CoM = np.array([[float(sp.N(el, 10, chop=True)) for el in row] for row in substitute(self.bc, consts)])

        # 2) Setup figure.
        if not (hasattr(self, "_fig") and self._fig is not None and hasattr(self, "_ax") and self._ax is not None and plt.fignum_exists(self._fig.number)):
            self._fig, self._ax = plt.subplots(1, 1, figsize=(5, 5))
            plt.show(block=False)
        else:
            self._ax.clear()
        ax = self._ax

        # Drawing parameters.
        linewidth = 4 * scale
        gd_markersize = 14 * scale
        jc_markersize = 12 * scale
        ef_markersize = 15 * scale
        fontsize = 12 * scale

        # 3) Plot robot geometry.
        ax.plot(joints[:, 0], joints[:, 1], "r-", linewidth=linewidth, alpha=alpha)
        ax.plot(joints[:, 0], joints[:, 1], "bo", markersize=gd_markersize, alpha=alpha)
        for i in range(muscle_a.shape[0]):
            ax.plot(
                [muscle_a[i, 0], muscle_b[i, 0]],
                [muscle_a[i, 1], muscle_b[i, 1]],
                "b",
                alpha=alpha,
            )
            if text:
                ax.text(
                    muscle_a[i, 0],
                    muscle_a[i, 1],
                    f"$a_{i+1}$",
                    fontsize=fontsize,
                    alpha=alpha,
                )
                ax.text(
                    muscle_b[i, 0],
                    muscle_b[i, 1],
                    f"$b_{i+1}$",
                    fontsize=fontsize,
                    alpha=alpha,
                )
                ax.text(
                    (muscle_a[i, 0] + muscle_b[i, 0]) / 2,
                    (muscle_a[i, 1] + muscle_b[i, 1]) / 2,
                    f"$l_{i+1}$",
                    fontsize=fontsize,
                    alpha=alpha,
                )
        ax.plot(CoM[:, 0], CoM[:, 1], "oy", markersize=jc_markersize, alpha=alpha)
        for i in range(CoM.shape[0]):
            ax.text(CoM[i, 0], CoM[i, 1], f"$Lc_{i+1}$", fontsize=fontsize, alpha=alpha)
        ax.plot(
            end_effector[0],
            end_effector[1],
            "<b",
            markersize=ef_markersize,
            alpha=alpha,
        )

        ax.set_xlabel("$x \\; (m)$")
        ax.set_ylabel("$y \\; (m)$")
        ax.set_title("Model Pose")
        ax.axis("equal")

        if use_axis_limits:
            L_max = self.constants[self.L[0]] + self.constants[self.L[1]]
            if self.nd >= 3:
                L_max += self.constants[self.L[2]]
            ax.set_xlim([-L_max, L_max])
            ax.set_ylim([-L_max / 2, 1.5 * L_max])

        # 4) Helper to convert Torch tensors.
        def to_numpy(val):
            return val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val

        # 5) Get state info.
        full_state = info.get("current_state", np.concatenate([q, np.zeros(2 * self.nd)]))
        joint_angles_str = np.round(to_numpy(full_state[: self.nd]), 2)
        joint_velocities_str = np.round(to_numpy(full_state[self.nd : 2 * self.nd]), 2)
        if full_state.size >= 3 * self.nd:
            joint_accelerations_str = np.round(to_numpy(full_state[2 * self.nd : 3 * self.nd]), 2)
        else:
            joint_accelerations_str = np.zeros(self.nd)
        joint_jerk_cmd = np.round(to_numpy(info.get("joint_jerk_cmd", np.zeros(self.nd))), 2)
        torque_str = np.round(to_numpy(info.get("torque", np.zeros(self.nd))), 2)
        pos_error_str = np.round(to_numpy(info.get("pos_error", 0)), 4)
        condJ_str = np.round(to_numpy(info.get("condJ", 0)), 2)
        raw_action_str = np.round(to_numpy(info.get("raw_action", np.zeros(8))), 2)

        # 6) Compute muscle values.
        q_rad = np.deg2rad(q)
        try:
            muscle_lengths = self.muscle_lengths_func(*q_rad)
        except:
            muscle_lengths = "N/A"
        try:
            if not hasattr(self, "muscle_velocities_func"):
                self.muscle_velocities_func = lambdify(self.Q(), self.lmd, modules="numpy")
            muscle_velocities = self.muscle_velocities_func(*q_rad)
        except:
            muscle_velocities = "N/A"
        try:
            if not hasattr(self, "muscle_accelerations_func"):
                self.muscle_accelerations_func = lambdify(self.Q(), self.lmdd, modules="numpy")
            muscle_accelerations = self.muscle_accelerations_func(*q_rad)
        except:
            muscle_accelerations = "N/A"

        muscle_lengths_str = np.round(to_numpy(muscle_lengths), 2) if not isinstance(muscle_lengths, str) else "N/A"
        muscle_velocities_str = np.round(to_numpy(muscle_velocities), 2) if not isinstance(muscle_velocities, str) else "N/A"
        muscle_accelerations_str = np.round(to_numpy(muscle_accelerations), 2) if not isinstance(muscle_accelerations, str) else "N/A"

        # 7) Get target info.
        target_val = info.get("target", None)
        target_str = np.round(to_numpy(target_val), 2) if target_val is not None else "N/A"
        ee_val = info.get("end_effector", None)
        ee_str = np.round(to_numpy(ee_val), 2) if ee_val is not None else "N/A"

        # 8) Build text overlay.
        text_str = f"Target: {target_str}\n" f"End Effector: {ee_str}\n" f"Joint Angles: {joint_angles_str}\n" f"Joint Velocities: {joint_velocities_str}\n" f"Joint Accelerations: {joint_accelerations_str}\n" f"Cmd Joint Jerk: {joint_jerk_cmd}\n" f"Torque: {torque_str}\n" f"Pos Error: {pos_error_str}\n" f"Cond(J): {condJ_str}\n" f"Raw Action: {raw_action_str}\n" f"Muscle Lengths: {muscle_lengths_str}\n" f"Muscle Velocities: {muscle_velocities_str}\n" f"Muscle Accelerations: {muscle_accelerations_str}"
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 9) Draw a target marker if provided.
        if target_val is not None:
            t = np.array(target_val)
            if t.size >= 2:
                outer_circle_radius = 0.04  # Adjust as desired.
                outer_circle = patches.Circle(
                    (t[0], t[1]),
                    radius=outer_circle_radius,
                    facecolor="none",
                    edgecolor="green",
                    linewidth=2,
                )
                ax.add_patch(outer_circle)
                inner_circle_radius = 0.01  # Inner dot.
                inner_circle = patches.Circle(
                    (t[0], t[1]),
                    radius=inner_circle_radius,
                    facecolor="red",
                    edgecolor="none",
                )
                ax.add_patch(inner_circle)

        plt.draw()
        plt.pause(0.001)

        # 10) If requested, play trajectory and then wait extra time before closing.
        if play_trajectory_flag:
            print("[draw_model] play_trajectory_flag is True. Calling play_trajectory...")
            ret_val = self.play_trajectory(playback_duration=playback_duration, decimation=decimation)
            if ret_val:
                print("[draw_model] play_trajectory finished. Waiting 3 seconds before auto-close of draw_model figure.")
                plt.pause(3)
                if self._fig is not None and plt.fignum_exists(self._fig.number):
                    plt.close(self._fig)
                    self._fig = None
                    self._ax = None
                    print("[draw_model] Closed the draw_model figure automatically.")
                else:
                    print("[draw_model] draw_model figure was already closed.")
            else:
                print("[draw_model] play_trajectory returned False (no data?), skipping auto-close.")


    def draw_model(
        self,
        q,
        in_deg=True,
        info=None,
        scale=0.8,
        use_axis_limits=True,
        alpha=1.0,
        text=True,
        play_trajectory_flag=False,
        playback_duration=1.0,
        decimation=1,
    ):
        """
        Draws the robot configuration in a Matplotlib figure with a debug overlay.

        Changes:
        - We now compute pos_error if not provided, using (target - end_effector).
        - We ensure 'Target' and 'End Effector' show numeric values if possible.
        - We show 'Final Joint Torque' if provided in info["joint_torque"].
        - We show 'Muscle Torques' from info["tau"] or similar.
        - Muscle lengths/velocities/accelerations in mm, mm/s, mm/s^2.
        """
        import sympy as sp  # ensure sympy is in scope if not already
        info = info or {}

        # -----------------------------
        # 1) Symbolic substitution
        # -----------------------------
        consts = self.model_parameters(q=q, in_deg=in_deg)
        # For the arm geometry
        joints = np.array([[float(sp.N(el, 10, chop=True)) 
                            for el in row] 
                        for row in substitute(self.jc, consts)])
        muscle_a = np.array([[float(sp.N(el, 10, chop=True)) 
                            for el in row] 
                            for row in substitute(self.ap, consts)])
        muscle_b = np.array([[float(sp.N(el, 10, chop=True)) 
                            for el in row] 
                            for row in substitute(self.bp, consts)])
        ee_sub = substitute(self.ee, consts)
        end_effector = np.array([float(sp.N(el, 10, chop=True)) 
                                for el in ee_sub]).flatten()
        CoM = np.array([[float(sp.N(el, 10, chop=True)) 
                        for el in row] 
                        for row in substitute(self.bc, consts)])

        # -----------------------------
        # 2) Setup figure/axes
        # -----------------------------
        if not (hasattr(self, "_fig") and self._fig is not None and
                hasattr(self, "_ax") and self._ax is not None and
                plt.fignum_exists(self._fig.number)):
            self._fig, self._ax = plt.subplots(1, 1, figsize=(5, 5))
            plt.show(block=False)
        else:
            self._ax.clear()
        ax = self._ax

        # -----------------------------
        # 3) Draw the arm geometry
        # -----------------------------
        linewidth = 4 * scale
        gd_markersize = 14 * scale
        jc_markersize = 12 * scale
        ef_markersize = 15 * scale

        # The arm
        ax.plot(joints[:, 0], joints[:, 1], "r-", linewidth=linewidth, alpha=alpha)
        ax.plot(joints[:, 0], joints[:, 1], "bo", markersize=gd_markersize, alpha=alpha)

        # Muscles
        for i in range(muscle_a.shape[0]):
            ax.plot([muscle_a[i, 0], muscle_b[i, 0]],
                    [muscle_a[i, 1], muscle_b[i, 1]], "b", alpha=alpha)

        # Centers of mass
        ax.plot(CoM[:, 0], CoM[:, 1], "oy", markersize=jc_markersize, alpha=alpha)

        # End effector
        ax.plot(end_effector[0], end_effector[1], "<b", markersize=ef_markersize, alpha=alpha)

        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_title("Model Pose")
        ax.axis("equal")

        if use_axis_limits:
            L_max = float(self.constants[self.L[0]]) + float(self.constants[self.L[1]])
            if self.nd == 3:
                L_max += float(self.constants[self.L[2]])
            ax.set_xlim([-L_max, L_max])
            ax.set_ylim([-L_max / 2, 1.5 * L_max])

        # -----------------------------
        # 4) Gather debug info
        # -----------------------------
        # current_state is something like [q0, q1, dq0, dq1, ddq0, ddq1]
        current_state = info.get("current_state", np.zeros(6))
        raw_action = info.get("raw_action", "N/A")

        # muscle torque array from muscle_space_controller
        muscle_torques = info.get("tau", None)  # sometimes stored as 'muscle_forces'
        if muscle_torques is not None:
            muscle_torques_str = np.round(muscle_torques, 3).tolist()
        else:
            muscle_torques_str = "N/A"

        # final joint torque if provided
        final_joint_torque = info.get("joint_torque", None)
        if final_joint_torque is not None:
            final_joint_torque_str = np.round(final_joint_torque, 3).tolist()
        else:
            final_joint_torque_str = "N/A"

        # muscle lengths, velocities, accelerations (in mm)
        # we look for them in info; if missing, we compute from q
        # multiply by 1000 => mm
        muscle_lengths_val = info.get("muscle_lengths", None)
        muscle_vels_val = info.get("muscle_velocities", None)
        muscle_accels_val = info.get("muscle_accelerations", None)

        if muscle_lengths_val is None:
            # compute from current q
            ml = np.array(self.muscle_lengths_func(*q), dtype=float)*1000
        else:
            ml = np.array(muscle_lengths_val, dtype=float)*1000

        if muscle_vels_val is None:
            # we can attempt to compute from (q, dq)
            mlv = np.array(self.muscle_velocities_func(q, current_state[2:4]), dtype=float)*1000
        else:
            mlv = np.array(muscle_vels_val, dtype=float)*1000

        if muscle_accels_val is None:
            # if we have ddq, we can compute
            mlvv = np.array(self.muscle_acceleration_func(q, current_state[2:4], current_state[4:6]),
                            dtype=float)*1000
        else:
            mlvv = np.array(muscle_accels_val, dtype=float)*1000

        ml_str = np.round(ml, 2).tolist()
        mlv_str = np.round(mlv, 2).tolist()
        mlvv_str = np.round(mlvv, 2).tolist()

        # target and end effector from info
        # fallback to the actual end_effector above if none given
        target_val = info.get("target", None)
        ee_val = info.get("end_effector", end_effector)  # from environment or fallback to the above

        # If the environment never sets "target", or it's None, we'll show "N/A"
        if target_val is not None and len(target_val) >= 2:
            target_str = np.round(target_val, 3).tolist()
        else:
            target_str = "N/A"

        # If environment sets "end_effector", use that; else from geometry
        if isinstance(ee_val, (list, np.ndarray)) and len(ee_val) >= 2:
            ee_str = np.round(ee_val, 3).tolist()
        else:
            ee_str = "N/A"

        # pos_error: either read from info or compute from (target, ee)
        pos_error = info.get("pos_error", None)
        if pos_error is None:
            # compute from target_val, ee_val if possible
            if isinstance(target_val, (list, np.ndarray)) and len(target_val) >= 2 and \
            isinstance(ee_val, (list, np.ndarray)) and len(ee_val) >= 2:
                pos_error = float(np.linalg.norm(np.array(target_val[:2]) - np.array(ee_val[:2])))
            else:
                pos_error = 0.0
        pos_error_rounded = np.round(pos_error, 4)

        # -----------------------------
        # 5) Format overlay text
        # -----------------------------
        q1, q2 = np.round(current_state[:2], 3).tolist()
        dq1, dq2 = np.round(current_state[2:4], 3).tolist()
        ddq1, ddq2 = np.round(current_state[4:6], 3).tolist()

        # raw action
        if isinstance(raw_action, (list, np.ndarray)):
            raw_action_str = np.round(raw_action, 3).tolist()
        else:
            raw_action_str = raw_action  # "N/A" or similar

        overlay_str = (
            f"Target: {target_str}\n"
            f"End Effector: {ee_str}\n"
            f"Joint Angles: [{q1}, {q2}]\n"
            f"Joint Velocities: [{dq1}, {dq2}]\n"
            f"Joint Accelerations: [{ddq1}, {ddq2}]\n"
            f"Muscle Torques: {muscle_torques_str}\n"
            f"Final Joint Torque: {final_joint_torque_str}\n"
            f"Pos Error: {pos_error_rounded}\n"
            f"Raw Action: {raw_action_str}\n"
            f"Muscle Lengths (mm): {ml_str}\n"
            f"Muscle Vel (mm/s): {mlv_str}\n"
            f"Muscle Acc (mm/s^2): {mlvv_str}\n"
        )

        # draw text box
        ax.text(
            0.05,
            0.95,
            overlay_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # -----------------------------
        # 6) Draw target if we have it
        # -----------------------------
        if isinstance(target_val, (list, np.ndarray)) and len(target_val) >= 2:
            circ_outer = plt.Circle(
                (target_val[0], target_val[1]),
                radius=0.03,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax.add_patch(circ_outer)
            circ_inner = plt.Circle(
                (target_val[0], target_val[1]),
                radius=0.01,
                color="red",
            )
            ax.add_patch(circ_inner)

        # update
        plt.draw()
        plt.pause(0.001)

        # -----------------------------
        # 7) If requested, play trajectory
        # -----------------------------
        if play_trajectory_flag:
            ret_val = self.play_trajectory(playback_duration=playback_duration, decimation=decimation)
            if ret_val:
                plt.pause(2)  # small delay
                if self._fig is not None and plt.fignum_exists(self._fig.number):
                    plt.close(self._fig)
                    self._fig = None
                    self._ax = None




    # --------------------------------------------------------------------
    # Joint Limit Enforcement Methods (as part of ArmModel)
    # --------------------------------------------------------------------

    def enforce_joint_position_limits(self, q):
        """
        Clamp joint positions q to be within [self.joint_low, self.joint_high].
        q: numpy array of joint positions.
        Returns:
            numpy array of clamped joint positions.
        """
        return np.clip(q, self.joint_low, self.joint_high)

    def enforce_joint_velocity_limits(self, q_dot):
        """
        Clamp joint velocities q_dot to be within [-self.joint_vel_limits, self.joint_vel_limits].
        q_dot: numpy array of joint velocities.
        Returns:
            numpy array of clamped joint velocities.
        """
        return np.clip(q_dot, -self.joint_vel_limits, self.joint_vel_limits)

    def enforce_joint_acceleration_limits(self, q_ddot):
        """
        Clamp joint accelerations q_ddot to be within [-self.joint_acc_limits, self.joint_acc_limits].
        q_ddot: numpy array of joint accelerations.
        Returns:
            numpy array of clamped joint accelerations.
        """
        return np.clip(q_ddot, -self.joint_acc_limits, self.joint_acc_limits)

    def task_space_PID_controller(
        self,
        q,
        q_dot,
        x_des,
        dot_x_des=None,
        ddot_x_des=None,
        Kp=None,
        Kd=None,
        Ki=None,
        dt=0.001,
        return_torque=False,
    ):
        nd = self.nd

        # Convert q and q_dot to 1D tensors.
        q_torch = torch.as_tensor(q, dtype=torch.float32, device=device).flatten()
        q_dot_torch = torch.as_tensor(q_dot, dtype=torch.float32, device=device).flatten()

        # Convert desired end-effector quantities to column vectors.
        x_des_torch = torch.as_tensor(x_des, dtype=torch.float32, device=device).view(-1, 1)
        if dot_x_des is not None:
            dot_x_des_torch = torch.as_tensor(dot_x_des, dtype=torch.float32, device=device).view(-1, 1)
        else:
            dot_x_des_torch = torch.zeros(x_des_torch.shape, dtype=torch.float32, device=device)
        if ddot_x_des is not None:
            ddot_x_des_torch = torch.as_tensor(ddot_x_des, dtype=torch.float32, device=device).view(-1, 1)
        else:
            ddot_x_des_torch = torch.zeros(x_des_torch.shape, dtype=torch.float32, device=device)

        # Set gains (default values if not provided)
        if Kp is None:
            Kp = np.eye(nd) * 50.0
        if Kd is None:
            Kd = np.eye(nd) * 10.0
        if Ki is None:
            Ki = np.eye(nd) * 5.0  # default integral gain

        Kp_torch = torch.as_tensor(Kp, dtype=torch.float32, device=device)
        Kd_torch = torch.as_tensor(Kd, dtype=torch.float32, device=device)
        Ki_torch = torch.as_tensor(Ki, dtype=torch.float32, device=device)

        # Compute current end-effector position and velocity.
        x_current = torch.as_tensor(self.fk_func(*q), dtype=torch.float32, device=device).view(-1, 1)
        x_dot_current = torch.as_tensor(
            self.forward_speed_kinematics_lambdified(q, q_dot),
            dtype=torch.float32,
            device=device,
        ).view(-1, 1)

        # Compute task-space errors.
        pos_error = x_des_torch - x_current
        vel_error = dot_x_des_torch - x_dot_current

        # Initialize or update the integral error term (store in self.integral_error).
        # It is assumed that self.integral_error is a column vector of shape (nd, 1)
        if not hasattr(self, "integral_error"):
            self.integral_error = torch.zeros_like(pos_error)
        self.integral_error = self.integral_error + pos_error * dt  # accumulate error over time

        # Desired end-effector acceleration command (PID)
        ddot_x_cmd = ddot_x_des_torch + Kp_torch @ pos_error + Kd_torch @ vel_error + Ki_torch @ self.integral_error

        if not return_torque:
            return ddot_x_cmd.detach().cpu().numpy().flatten()
        else:
            # Get Jacobian and its time derivative.
            J = torch.as_tensor(self.J_func(*q), dtype=torch.float32, device=device)
            J_dot = torch.as_tensor(self.J_dot_func(q, q_dot), dtype=torch.float32, device=device)
            # Compute joint acceleration command:
            ddq_cmd = torch.as_tensor(self.damped_pinv(J), dtype=torch.float32, device=device) @ (ddot_x_cmd - J_dot @ q_dot_torch.view(-1, 1))
            ddq_cmd = ddq_cmd.view(-1)  # make it 1D

            M_vals = torch.as_tensor(self.mass_matrix_func(*q), dtype=torch.float32, device=device)
            C_vals = torch.as_tensor(self.coriolis_forces_func(q, q_dot), dtype=torch.float32, device=device).view(-1)
            G_vals = torch.as_tensor(self.gravitational_forces_func(*q), dtype=torch.float32, device=device).view(-1)

            # Compute torque command.
            tau_cmd = (M_vals @ ddq_cmd.view(-1, 1) + C_vals.view(-1, 1) + G_vals.view(-1, 1)).view(-1)
            return tau_cmd.detach().cpu().numpy()

    def joint_space_controller(
        self,
        q,
        q_dot,
        q_des,
        dot_q_des=None,
        ddot_q_des=None,
        Kp=None,
        Kd=None,
        Ki=None,
        controller_type="PID",
        dt=0.01,
        return_torque=True,
    ):

        # self,
        # q,
        # q_dot,
        # q_des,  # desired joint positions
        # dot_q_des=None,  # desired joint velocities
        # ddot_q_des=None,  # desired joint accelerations (feedforward)
        # Kp=None,
        # Kd=None,
        # Ki=None,  # Only used in PID mode.
        # controller_type="PID",  # must be either 'PD' or 'PID'
        # dt=0.01,  # time step (for integration)
        # return_torque=True,  # if True, compute torque via computed-torque approach

        nd = self.nd  # number of joints
        # Convert inputs to Torch tensors.
        q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
        q_dot_t = torch.as_tensor(q_dot, dtype=torch.float32, device=device)
        q_des_t = torch.as_tensor(q_des, dtype=torch.float32, device=device)

        # Set default desired velocities and accelerations.
        if dot_q_des is not None:
            dot_q_des_t = torch.as_tensor(dot_q_des, dtype=torch.float32, device=device)
        else:
            dot_q_des_t = torch.zeros(nd, dtype=torch.float32, device=device)

        if ddot_q_des is not None:
            ddot_q_des_t = torch.as_tensor(ddot_q_des, dtype=torch.float32, device=device)
        else:
            ddot_q_des_t = torch.zeros(nd, dtype=torch.float32, device=device)

        # Set default gains if not provided.
        if Kp is None:
            Kp = np.eye(nd) * 50.0
        Kp_t = torch.as_tensor(Kp, dtype=torch.float32, device=device)

        if Kd is None:
            Kd = np.eye(nd) * 10.0
        Kd_t = torch.as_tensor(Kd, dtype=torch.float32, device=device)

        if Ki is None:
            Ki = np.eye(nd) * 5.0
        Ki_t = torch.as_tensor(Ki, dtype=torch.float32, device=device)

        # Compute errors.
        pos_error = q_des_t - q_t
        vel_error = dot_q_des_t - q_dot_t

        # Start with the feedforward acceleration.
        ddot_q_cmd = ddot_q_des_t.clone()

        # Add PD feedback.
        ddot_q_cmd += (Kp_t @ pos_error) + (Kd_t @ vel_error)

        # If in PID mode, update and add the integral term.
        if controller_type == "PID":
            if not hasattr(self, "integral_error"):
                self.integral_error = torch.zeros(nd, dtype=torch.float32, device=device)
            self.integral_error += pos_error * dt
            ddot_q_cmd += Ki_t @ self.integral_error
        else:
            # For PD mode, ensure the integrator is reset.
            if hasattr(self, "integral_error"):
                self.integral_error = torch.zeros(nd, dtype=torch.float32, device=device)

        # If only the acceleration command is desired, return it.
        if not return_torque:
            return ddot_q_cmd.detach().cpu().numpy()
        else:
            # Compute the torque command using the computed-torque formulation:
            # tau = M(q) * ddot_q_cmd + C(q,q_dot) + G(q)
            M_vals = np.array(self.mass_matrix_func(*q), dtype=float)
            C_vals = np.array(self.coriolis_forces_func(q, q_dot), dtype=float).ravel()
            G_vals = np.array(self.gravitational_forces_func(*q), dtype=float).ravel()

            M_t = torch.as_tensor(M_vals, dtype=torch.float32, device=device)
            C_t = torch.as_tensor(C_vals, dtype=torch.float32, device=device)
            G_t = torch.as_tensor(G_vals, dtype=torch.float32, device=device)

            tau_cmd_t = M_t @ ddot_q_cmd + C_t + G_t
            return tau_cmd_t.detach().cpu().numpy()

    # ---------------- Muscle-Space Methods ----------------


    def update_activationO(self, u_cmd, dt):
        """
        Update muscle activation using a first-order dynamics model.
        u_cmd: np.array of desired activations (values in [0, 1]) for each muscle.
        dt: time step.
        """
        alpha = 50.0  # activation time constant (tunable)
        # Simple first-order update:  dot{a} = alpha*(u_cmd - a)
        self.activation += alpha * (u_cmd - self.activation) * dt
        self.activation = np.clip(u_cmd, 0.0, 1.0)
        # You can print or log the updated activation if desired.
        self.logger.debug("Updated activation: " + str(self.activation))

    def update_activation(self, u_cmd, dt):
        """
        Perform multiple smaller substeps so muscle activation
        can converge quickly if needed.
        """
        alpha = 5.0        # muscle time constant
        substeps = 5       # or 10, 20, etc.
        sub_dt = dt / substeps

        for _ in range(substeps):
            # "dot(a) = alpha*(u_cmd - a)" in each small sub-step
            self.activation += alpha * (u_cmd - self.activation) * sub_dt
            self.activation = np.clip(self.activation, 0.0, 1.0)

        self.logger.debug("Updated activation: " + str(self.activation))


    def construct_muscle_space_inequality(self, NR, fm_par, Fmax):
        """
        Construct the feasible muscle-space inequality: Z * f_m0 <= B.
        Each muscle i's maximum force is scaled by its current activation level a[i],
        so that the net feasible muscle force lies within [-a[i]*Fmax[i], a[i]*Fmax[i]].
        """
        # Use self.activation (assumed to be a numpy array with one entry per muscle)
        a = self.activation  # shape (m,)
        Fmax_arr = np.asarray(Fmax).flatten()
        # Scale the maximum force by the activation
        Fmax_act = a * Fmax_arr  # elementwise multiplication

        # Lower bound constraint:
        #   fm_par + NR * f_m0 >= -Fmax_act   <=>   -NR * f_m0 <= Fmax_act + fm_par
        Z0 = -NR
        B0 = Fmax_act.reshape(-1, 1) + fm_par

        # Upper bound constraint:
        #   fm_par + NR * f_m0 <= Fmax_act    <=>    NR * f_m0 <= Fmax_act - fm_par
        Z1 = NR
        B1 = Fmax_act.reshape(-1, 1) - fm_par

        Z = np.concatenate((Z0, Z1), axis=0)
        B = np.concatenate((B0, B1), axis=0)
        return Z, B

    def muscle_space_controller(self, q, q_dot, q_des, dot_q_des=None, ddot_q_des=None,
                                  q_ddot_est=None, Kp_m=None, Kd_m=None, Ki_m=None, Kdd_m=None,
                                  controller_type="PID", dt=0.01, return_torque=True,
                                  use_optimization=False, u_cmd=None):
        """
        A muscle-space controller that computes joint torque via desired muscle accelerations.
        This version accepts an optional activation command (u_cmd) from an RL agent.
        If provided, it updates the internal muscle activation before computing the control.

        Parameters:
          q: current joint angles (array-like, size nd)
          q_dot: current joint velocities (array-like, size nd)
          q_des: desired joint angles (array-like, size nd)
          dot_q_des: desired joint velocities (optional)
          ddot_q_des: desired joint accelerations (feedforward, optional)
          q_ddot_est: estimated current joint accelerations (optional)
          Kp_m, Kd_m, Ki_m, Kdd_m: muscle-space gains (optional)
          controller_type: "PD" or "PID"
          dt: time step
          return_torque: if True, return the computed joint torque command; else return joint acceleration command.
          use_optimization: if True, use optimization to enforce muscle force constraints.
          u_cmd: (optional) RL activation command (array-like, one value per muscle, in [0,1]).

        Returns:
          np.array: joint torque command (if return_torque=True) or joint acceleration command.
        """
        nd = self.nd
        # Enforce joint limits on positions and velocities:
        q = self.enforce_joint_position_limits(np.array(q).flatten()[:nd])
        q_dot = self.enforce_joint_velocity_limits(np.array(q_dot).flatten()[:nd])

        # --- Update muscle activation if an RL command is provided ---
        if u_cmd is not None:
            self.update_activation(u_cmd, dt)
        # Now self.activation will be used in the feasible-force constraints

        # --- Step 1: Compute current muscle states from q, q_dot ---
        lm = np.array(self.muscle_lengths_func(*q), dtype=float).flatten()
        lmd = np.array(self.muscle_velocities_func(q.tolist(), q_dot.tolist()), dtype=float).flatten()

        # Build pose dictionary for substitution:
        pose = self.model_parameters(q=q, u=q_dot)
        q_vals = [pose[sym] for sym in self.q]
        q_dot_vals = [pose[sym] for sym in self.dq]

        R_num = np.array(self.R_func(*q_vals), dtype=float)
        RDotQDot_num = np.array(self.RDotQDot_func(q_vals, q_dot_vals), dtype=float).flatten()

        # --- Step 2: Compute current muscle acceleration ---
        if q_ddot_est is not None:
            q_ddot_est = self.enforce_joint_acceleration_limits(np.array(q_ddot_est, dtype=float))
            lmdd = R_num @ q_ddot_est + RDotQDot_num
        else:
            lmdd = RDotQDot_num

        # --- Step 3: Determine desired muscle states ---
        q_des = self.enforce_joint_position_limits(np.array(q_des).flatten()[:nd])
        if dot_q_des is not None:
            dot_q_des = self.enforce_joint_velocity_limits(np.array(dot_q_des).flatten()[:nd])
        lm_des = np.array(self.muscle_lengths_func(*q_des), dtype=float).flatten()

        if dot_q_des is not None:
         
            lmd_des = np.array(self.muscle_velocities_func(q_des.tolist(), dot_q_des.tolist()), dtype=float).flatten()
            
        else:
            lmd_des = np.zeros_like(lm_des)

        if ddot_q_des is not None:
            ddot_q_des = self.enforce_joint_acceleration_limits(np.array(ddot_q_des, dtype=float))
            pose_des = self.model_parameters(q=q_des, u=dot_q_des if dot_q_des is not None else np.zeros(nd))
            q_des_vals = [pose_des[sym] for sym in self.q]
            dot_q_des_vals = [pose_des[sym] for sym in self.dq]
            R_des_num = np.array(self.R_func(*q_des_vals), dtype=float)
            RDotQDot_des_num = np.array(self.RDotQDot_func(q_des_vals, dot_q_des_vals), dtype=float).flatten()
            lmdd_des = R_des_num @ ddot_q_des + RDotQDot_des_num
        else:
            lmdd_des = np.zeros_like(lm_des)

        # --- Step 4: Muscle-space control law ---
        m_val = lm.size  # number of muscles
        if Kp_m is None:
            Kp_m = np.eye(m_val) * 50.0
        if Kd_m is None:
            Kd_m = np.eye(m_val) * 10.0
        if Ki_m is None:
            Ki_m = np.eye(m_val) * 5.0
        if Kdd_m is None:
            Kdd_m = np.zeros((m_val, m_val))

        e_l = lm_des - lm      # muscle length error.
        e_ld = lmd_des - lmd    # muscle velocity error.
        e_ldd = lmdd_des - lmdd  # muscle acceleration error.

        lmdd_cmd = lmdd_des + (Kp_m @ e_l) + (Kd_m @ e_ld) + (Kdd_m @ e_ldd)
        #("e_l", e_l)
        #("e_ld", e_ld)
        #("e_ldd", e_ldd)
        #("lmdd_cmd", lmdd_cmd)

        if controller_type == "PID":
            if not hasattr(self, "integral_error_m"):
                self.integral_error_m = np.zeros(m_val)
            self.integral_error_m += e_l * dt
            lmdd_cmd += Ki_m @ self.integral_error_m
        else:
            if hasattr(self, "integral_error_m"):
                self.integral_error_m = np.zeros(m_val)

        # --- Step 5: Map desired muscle acceleration to an initial joint acceleration guess ---
        ddot_q_cmd_initial = damped_pinv_np(R_num, damping=0.01) @ (lmdd_cmd - RDotQDot_num)
        #("ddot_q_cmd_initial (before limits)", ddot_q_cmd_initial)

        ddot_q_cmd_initial = self.enforce_joint_acceleration_limits(ddot_q_cmd_initial)
        #("R_num", R_num)
        #("RDotQDot_num", RDotQDot_num)
        #("ddot_q_cmd_initial (after limits)", ddot_q_cmd_initial)

        # --- Step 6: Compute inverse dynamics torque from ddot_q_cmd_initial ---
        M_vals = np.array(self.mass_matrix_func(*q), dtype=float)
        C_vals = np.array(self.coriolis_forces_func(q, q_dot), dtype=float).ravel()
        G_vals = np.array(self.gravitational_forces_func(*q), dtype=float).ravel()
        tau_inv = M_vals @ ddot_q_cmd_initial + C_vals + G_vals
        #("tau_inv", tau_inv)

        # --- Step 7: Compute muscle-based torque and force ---
        from sympy import Matrix
        tau_muscle, fm_total = self.calculate_muscle_force(Matrix(lmdd_cmd), pose, use_optimization=use_optimization)
        #("tau_muscle", tau_muscle)
        #("fm_total", fm_total)

        # --- Step 8: Error correction ---
        tau_error = np.array(tau_muscle).flatten() - tau_inv
        ddot_corr = np.linalg.pinv(M_vals) @ tau_error
        ddot_q_cmd = ddot_q_cmd_initial + ddot_corr
        ddot_q_cmd = self.enforce_joint_acceleration_limits(ddot_q_cmd)
        #("tau_error", tau_error)
        #("ddot_corr", ddot_corr)
        #("ddot_q_cmd (after final limits)", ddot_q_cmd)

        # --- Step 9: Final torque command ---
        tau_final = M_vals @ ddot_q_cmd + C_vals + G_vals
        #("tau_final", tau_final)
        if not return_torque:
            return ddot_q_cmd
        return tau_final

    def get_initial_guess(self, fm_par, Fmax, NR):
        """
        Computes a good initial guess x0 for the null-space optimization.

        Parameters
        ----------
        fm_par : np.array, shape (m, 1)
            The particular muscle force solution.
        Fmax : np.array or list, shape (m,)
            The maximum allowable force (assumed symmetric: [-Fmax, Fmax]).
        NR : np.array, shape (m, m)
            The null-space projector.

        Returns
        -------
        x0 : np.array, shape (m,)
            An initial guess for the null-space adjustment.
        """
        # Ensure Fmax is a flattened numpy array.
        Fmax = np.asarray(Fmax).flatten()
        m = fm_par.shape[0]

        # Strategy 1: Warm start (if previous solution exists)
        if hasattr(self, "prev_x"):
            x0_warm = self.prev_x.copy().flatten()  # make sure to copy and flatten
        else:
            x0_warm = np.zeros(m)

        # Strategy 2: Heuristic Correction.
        # For each muscle force, if fm_par exceeds the bound, nudge a fraction (alpha)
        # toward the bound.
        alpha = 0.5  # a tunable fraction
        x0_heuristic = np.zeros(m)
        for i in range(m):
            if fm_par[i, 0] > Fmax[i]:
                x0_heuristic[i] = alpha * (Fmax[i] - fm_par[i, 0])
            elif fm_par[i, 0] < -Fmax[i]:
                x0_heuristic[i] = alpha * (-Fmax[i] - fm_par[i, 0])
            else:
                x0_heuristic[i] = 0.0

        # Strategy 3: Analytical Approximation.
        # We would like f_total = fm_par + NR*x to be within [-Fmax, Fmax].
        # Let desired_f = clip(fm_par, -Fmax, Fmax) and define
        #     d = desired_f - fm_par.
        # Then we solve NR*x ≈ d in the least-squares sense.
        desired_f = np.clip(fm_par.flatten(), -Fmax, Fmax)
        d = desired_f - fm_par.flatten()
        x0_analytical, _, _, _ = np.linalg.lstsq(NR, d, rcond=None)

        # Combine the candidates – for example, by averaging.
        x0_candidate = (x0_warm + x0_heuristic + x0_analytical) / 3.0

        return x0_candidate

    def calculate_muscle_force(self, lmdd, pose, use_optimization=False):
        """
        Compute muscle-space force from a desired muscle acceleration lmdd.
        (Code before remains the same …)
        """

        # Convert pose to numeric joint angles and velocities.
        q_vals = [pose[sym] for sym in self.q]
        q_dot_vals = [pose[sym] for sym in self.dq]
        #("q_vals", q_vals)
        #("q_dot_vals", q_dot_vals)

        # Evaluate numeric mass matrix, internal forces, R, and RDotQDot.
        M = np.array(self.mass_matrix_func(*q_vals), dtype=float)
        #("M", M)
        f = np.array(self.f.subs(pose).evalf(), dtype=float).reshape(-1, 1)
        #("f", f)
        R = np.array(self.R_func(*q_vals), dtype=float)
        #("R", R)
        RT = R.T
        #("RT", RT)
        RDotQDot = np.array(self.RDotQDot_func(q_vals, q_dot_vals), dtype=float).reshape(-1, 1)
        #("RDotQDot", RDotQDot)

        # Build muscle-space operators
        MInv = np.linalg.inv(M)
        LmInv = R @ MInv @ RT  # matrix in muscle space
        Lm = np.linalg.pinv(LmInv)
        RBarT = np.linalg.pinv(RT)
        NR = np.eye(RBarT.shape[0]) - RBarT @ RT  # null-space projector

        lmdd_val = np.array(lmdd, dtype=float).reshape(-1, 1)
        fm_par = -Lm @ (lmdd_val - RDotQDot) - RBarT @ f
        # For a model that supports both positive and negative forces,
        # you might not rectify fm_par here. (If desired, you can also clip fm_par later.)
        #("fm_par", fm_par)

        m = fm_par.shape[0]
        fm_0 = np.zeros((m, 1))

        if use_optimization:
            # Construct inequality constraints.
            # For each muscle, we want:
            #    -Fmax[i] <= fm_par[i] + (NR*x)[i] <= Fmax[i]
            # which is equivalent to:
            #    fm_par[i] + (NR*x)[i] <= Fmax[i]
            #    -fm_par[i] - (NR*x)[i] <= Fmax[i]
            Fmax_arr = np.asarray(self.Fmax).flatten()
            # Upper constraint: (NR*x) <= Fmax - fm_par
            Z_upper = NR
            B_upper = Fmax_arr.reshape(-1, 1) - fm_par
            # Lower constraint: -(NR*x) <= Fmax + fm_par  <=>  NR*x >= -Fmax - fm_par
            Z_lower = -NR
            B_lower = Fmax_arr.reshape(-1, 1) + fm_par

            # Combine these constraints
            Z = np.concatenate((Z_upper, Z_lower), axis=0)
            B = np.concatenate((B_upper, B_lower), axis=0)

            # Define the inequality function for the optimizer.
            def ineq(x):
                # Returns vector: B - Z @ x >= 0 elementwise.
                return (B - Z @ x.reshape(-1, 1)).flatten()

            # Also include a pair constraint if you have pairs of muscles that act oppositely.
            # For example, suppose muscles i and j are an antagonistic pair:
            #   -Fmax_pair <= (fm_par[i] + fm_par[j]) + ((NR*x)[i] + (NR*x)[j]) <= Fmax_pair
            # You can define such constraints here as needed. (This example is left for you to tailor.)
            def pair_constraint(x):
                # As an example, if muscles 0 and 1 are a pair:
                pair_sum = fm_par[0, 0] + fm_par[1, 0] + (NR[0, :] + NR[1, :]) @ x
                Fmax_pair = self.Fmax_pair  # a defined value for the pair limit
                return np.array([Fmax_pair - abs(pair_sum)])  # must be non-negative

            # Compute a good starting guess using our strategies:
            x0 = self.get_initial_guess(fm_par, Fmax_arr, NR)
            #("x0", x0)
            x_max = getattr(self, 'x_max', 1e6)

            bounds = [(-self.x_max, self.x_max)] * m
            constraints = [{"type": "ineq", "fun": ineq}]
            # Optionally, add pair constraints if defined:
            # constraints.append({"type": "ineq", "fun": pair_constraint})

            from scipy.optimize import minimize

            sol = minimize(lambda x: np.sum(x**2), x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"disp": False})
            #("Optimization result", sol)
            if not sol.success:
                # Fallback: saturate the forces to the feasible limit instead of erroring out.
                #("Optimization failed; saturating forces to feasible limits.")
                # For example, set x0 = 0 (i.e. use fm_par) or use the heuristic x0.
                fm_0 = x0.reshape(-1, 1)
            else:
                fm_0 = sol.x.reshape(-1, 1)
            # Save the solution for warm-starting next time.
            self.prev_x = sol.x.copy() if sol.success else x0.copy()

        fm_perp = NR @ fm_0
        fm_total = fm_par + fm_perp

        tau = -RT @ fm_par  # convert muscle forces to joint torques
        #tau = RT @ fm_par  # convert muscle forces to joint torques

        return tau, fm_total

    def task_space_controller_by_muscles(self,
                                        q,
                                        q_dot,
                                        x_des,
                                        dot_x_des=None,
                                        ddot_x_des=None,
                                        Kp_task=None,
                                        Kd_task=None,
                                        Ki_task=None,
                                        dt=0.01,
                                        return_torque=True,
                                        use_optimization=False,
                                        u_cmd=None):
        """
        Task-space controller by muscles using ONLY a task-space PID.
        
        This controller does the following:
        
        1. Enforces joint limits and (optionally) updates muscle activation.
        
        2. Computes the current end-effector (task-space) position and velocity.
            Using task-space errors with default gains (or user-specified ones),
            it computes a desired end-effector acceleration command:
            
            xddot_cmd = ddot_x_des + Kp_task*(x_des - x_current) + Kd_task*(dot_x_des - x_dot_current)
                        + Ki_task * (integral_error)
        
        3. It maps the desired task acceleration into a desired joint acceleration
            using the pseudoinverse of the task Jacobian:
            
                qddot_des = pinv(J) * (xddot_cmd - J_dot*q_dot)
        
        4. It then uses the muscle kinematics to compute the desired muscle acceleration:
            
                lmdd_des = R * qddot_des + RDotQDot
        
        5. Finally, it calls calculate_muscle_force to obtain the muscle-based joint torque:
            
                tau = - R.T * fm_par
                
        Since only task-space PID is used, no extra PID loop is applied in muscle space.
        
        Parameters
        ----------
        q : array-like
            Current joint angles (length nd).
        q_dot : array-like
            Current joint velocities (length nd).
        x_des : array-like
            Desired end-effector position (length d).
        dot_x_des : array-like, optional
            Desired end-effector velocity (length d). Defaults to zero.
        ddot_x_des : array-like, optional
            Feed-forward desired end-effector acceleration (length d). Defaults to zero.
        Kp_task : ndarray, optional
            Task-space proportional gain (default: 50*I_d).
        Kd_task : ndarray, optional
            Task-space derivative gain (default: 10*I_d).
        Ki_task : ndarray, optional
            Task-space integral gain (default: 5*I_d).
        dt : float, optional
            Time step (default: 0.01).
        return_torque : bool, optional
            If True, returns the joint torque command; otherwise returns the joint acceleration command.
        use_optimization : bool, optional
            If True, optimization (SLSQP) is used to enforce feasible muscle forces.
        u_cmd : array-like, optional
            Optional activation command (updates muscle activation if provided).
        
        Returns
        -------
        tau_final : ndarray
            The computed joint torque command if return_torque is True; otherwise,
            the desired joint acceleration command.
        """
        # ---------- Step 0: Preliminary (enforce limits and update activation) ----------
        nd = self.nd
        q = self.enforce_joint_position_limits(np.array(q).flatten()[:nd])
        q_dot = self.enforce_joint_velocity_limits(np.array(q_dot).flatten()[:nd])
        if u_cmd is not None:
            self.update_activation(u_cmd, dt)
        
        # ---------- Step 1: Task-Space Errors and Desired Acceleration ----------
        # Compute current end-effector position (using forward kinematics).
        x_current = np.array(self.fk_func(*q), dtype=float).flatten()  # shape (d,)
        # Use default desired velocities/accelerations if not provided.
        if dot_x_des is None:
            dot_x_des = np.zeros_like(x_des)
        if ddot_x_des is None:
            ddot_x_des = np.zeros_like(x_des)
        
        d = len(x_des)
        # Default task-space gains.
        if Kp_task is None:
            Kp_task = np.eye(d) * 50.0
        if Kd_task is None:
            Kd_task = np.eye(d) * 10.0
        if Ki_task is None:
            Ki_task = np.eye(d) * 5.0
        
        # Initialize or update task integral error.
        if not hasattr(self, "task_integral_error"):
            self.task_integral_error = np.zeros((d, 1))
        pos_error = np.array(x_des, dtype=float).reshape(-1, 1) - x_current.reshape(-1, 1)
        
        # For the current end-effector velocity, we assume that forward_speed_kinematics_lambdified is available.
        x_dot_current = np.array(self.forward_speed_kinematics_lambdified(q, q_dot), dtype=float).reshape(-1, 1)
        vel_error = np.array(dot_x_des, dtype=float).reshape(-1, 1) - x_dot_current
        
        # Accumulate the integral error.
        self.task_integral_error += pos_error * dt
        
        # Compute the desired task acceleration command.
        # (This is the sole PID loop in the controller.)
        xddot_cmd = np.array(ddot_x_des, dtype=float).reshape(-1, 1) \
                    + Kp_task @ pos_error \
                    + Kd_task @ vel_error \
                    + Ki_task @ self.task_integral_error
        
        # ---------- Step 2: Map Task Acceleration to Joint Acceleration ----------
        # Obtain task Jacobian J and its time derivative J_dot.
        J_num = np.array(self.J_func(*q), dtype=float)       # shape: (d, nd)
        J_dot_num = np.array(self.J_dot_func(q, q_dot), dtype=float)  # shape: (d, nd)
        # Compute desired joint acceleration via damped pseudoinverse:
        #   qddot_des = pinv(J) * (xddot_cmd - J_dot * q_dot)
        qddot_des = damped_pinv_np(J_num, damping=0.01) @ (xddot_cmd - J_dot_num @ q.reshape(-1, 1))
        
        # ---------- Step 3: Map Joint Acceleration to Desired Muscle Acceleration ----------
        # Get the current pose: substitute constants, coordinates, speeds
        pose = self.model_parameters(q=q, u=q_dot)
        # Get muscle Jacobian R and its derivative product RDotQDot.
        # Here, self.R_func and self.RDotQDot_func are assumed to be lambdified functions.
        R_num = np.array(self.R_func(*[pose[sym] for sym in self.q]), dtype=float)
        RDotQDot_num = np.array(self.RDotQDot_func([pose[sym] for sym in self.q],
                                                    [pose[sym] for sym in self.dq]), dtype=float).flatten()
        # Desired muscle length acceleration:
        lmdd_des = R_num @ qddot_des + RDotQDot_num.reshape(-1, 1)
        
        # For consistency with our symbolic muscle solver, convert lmdd_des into a sympy Matrix.
        lmdd_des_sym = sp.Matrix(lmdd_des.tolist())
        
        # ---------- Step 4: Compute Muscle-Based Torque Using the Muscle-Space Model ----------
        # Call the muscle space method to compute the required muscle force and joint torque.
        from sympy import Matrix
        tau_muscle, fm_total = self.calculate_muscle_force(Matrix(lmdd_des_sym), pose, use_optimization=use_optimization)
        
        # The muscle-space method already returns joint torque via: tau = - R^T * fm_par.
        tau_final = np.array(tau_muscle, dtype=float).flatten()
        
        # Optionally, you might add an inverse-dynamics correction here,
        # but since we rely solely on the task-space PID,
        # we simply return the muscle-based torque.
        if not return_torque:
            # Alternatively, return the desired joint acceleration command.
            return qddot_des.flatten()
        return tau_final

# tests

def test_muscle_space_controller(nd=2):
    """
    Demonstrates the muscle-space control approach using the updated
    muscle_space_controller signature, which expects:
        muscle_space_controller(q, q_dot, q_des, ...)

    This test simulates closed-loop dynamics over a set simulation time,
    collects trajectory data, renders the robot configuration at each step,
    and finally plays back the trajectory.
    """
    dt = 0.01  # time step [s]
    simulation_time = 2.0  # total simulation duration [s]
    num_steps = int(simulation_time / dt)

    # 1) Create the ArmModel with gravity enabled.
    model = ArmModel(nd=nd, use_gravity=1)

    # 2) Set maximum muscle forces (example values)
    model.Fmax = np.array([100000] * (9 if nd == 3 else 6))
    model.x_max = np.max(model.Fmax)

    # 3) Define the initial state and a desired joint configuration.
    if nd == 2:
        q = np.deg2rad([0.0, 0.0])
        q_dot = np.zeros(2)
        q_des = np.deg2rad([60.0, 0.0])
    else:
        q = np.deg2rad([0.0, 15.0, 30.0])
        q_dot = np.zeros(3)
        q_des = np.deg2rad([20.0, 30.0, 40.0])

    # Combine q and q_dot into state if needed.
    state = np.concatenate([q, q_dot])

    # 4) Initialize trajectory and info data.
    model.trajectory = {
        "time": [],
        "end_effector": [],
        "joint_angles": [],
        "joint_velocities": [],
        "joint_accelerations": [],
        "joint_jerks": [],
        "torques": [],
        "pos_error": [],
        "reward": [],
    }
    model.last_info = {}

    plt.ion()  # Enable interactive mode for live drawing

    # 5) Simulation loop
    for step in range(num_steps):
        # Get torque command from muscle_space_controller.
        # This function now expects (q, q_dot, q_des, ...)
        tau = model.muscle_space_controller(q=q, q_dot=q_dot, q_des=q_des, use_optimization=True)  # or False, based on your test

        # Compute forward dynamics: solve M*ddq = tau - (C+G)
        M_vals = np.array(model.mass_matrix_func(*q), dtype=float)
        C_vals = np.array(model.coriolis_forces_func(q, q_dot), dtype=float).ravel()
        G_vals = np.array(model.gravitational_forces_func(*q), dtype=float).ravel()
        ddq = np.linalg.solve(M_vals, tau - (C_vals + G_vals))

        # Enforce joint acceleration, velocity, and position limits.
        ddq = model.enforce_joint_acceleration_limits(ddq)
        q_dot = model.enforce_joint_velocity_limits(q_dot + ddq * dt)
        q = model.enforce_joint_position_limits(q + q_dot * dt)

        # Collect trajectory and info data.
        current_ee = model.fk_func(*q).flatten()
        model.collect_trajectory_info(q, q_dot, current_ee, step, dt)

        # Draw the model and print step info.
        model.draw_model(np.rad2deg(q), in_deg=True, info=model.last_info)
        print(f"Step {step:03d} | q(rad)={q} | torque={tau}")
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    # 6) Play back the recorded trajectory.
    model.play_trajectory(playback_duration=2.0, decimation=1)

def test_task_space_controller_by_muscles(nd=2):
    """
    Demonstrates the task-space control approach using only a task-space PID.
    
    In this test, the controller computes a desired end-effector acceleration via a
    task-space PID law and then maps that into a joint acceleration and finally into a
    desired muscle acceleration. The muscle-space model then returns the joint torque
    command via: tau = -R^T * f_m.
    
    The simulation runs in closed-loop over a fixed time, collects trajectory data,
    renders the robot configuration at each step, and finally plays back the trajectory.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    dt = 0.01              # time step [s]
    simulation_time = 2.0  # total simulation duration [s]
    num_steps = int(simulation_time / dt)

    # 1) Create the ArmModel (with gravity enabled).
    model = ArmModel(nd=nd, use_gravity=1)

    # 2) Set maximum muscle forces (example values) and x_max.
    # (For 2R, muscle count is 6; for 3R, muscle count is 9.)
    model.Fmax = np.array([100000] * (9 if nd == 3 else 6))
    model.x_max = np.max(model.Fmax)

    # 3) Define the initial state.
    if nd == 2:
        # For a 2-DoF arm: initial joint angles (in radians) and velocities (zeros).
        q = np.deg2rad([0.0, 0.0])
        q_dot = np.zeros(2)
        # For the task target, we pick a desired joint configuration and compute its end-effector pose.
        q_des = np.deg2rad([60.0, 0.0])
    else:
        # For a 3-DoF arm.
        q = np.deg2rad([0.0, 15.0, 30.0])
        q_dot = np.zeros(3)
        q_des = np.deg2rad([20.0, 30.0, 40.0])

    # Instead of passing q_des to the controller, we compute the desired
    # task target x_des using the forward kinematics:
    x_des = model.fk_func(*q_des)  # should return a vector (e.g., 2D position)

    # 4) Initialize trajectory and info data.
    model.trajectory = {
        "time": [],
        "end_effector": [],
        "joint_angles": [],
        "joint_velocities": [],
        "joint_accelerations": [],
        "joint_jerks": [],
        "torques": [],
        "pos_error": [],
        "reward": [],
    }
    model.last_info = {}

    plt.ion()  # Enable interactive mode for live drawing

    # 5) Simulation loop
    for step in range(num_steps):
        # Call the new task-space controller method.
        # It expects: (q, q_dot, x_des, dot_x_des, ddot_x_des, dt, etc.)
        # We provide x_des and let dot_x_des and ddot_x_des default to zeros.
        tau = model.task_space_controller_by_muscles(
            q=q,
            q_dot=q_dot,
            x_des=x_des,
            dot_x_des=None,
            ddot_x_des=None,
            dt=dt,
            return_torque=True,
            use_optimization=True,
            u_cmd=None  # Or provide an activation command if desired.
        )

        # Compute forward dynamics: solve M * ddq = tau - (C + G)
        M_vals = np.array(model.mass_matrix_func(*q), dtype=float)
        C_vals = np.array(model.coriolis_forces_func(q, q_dot), dtype=float).ravel()
        G_vals = np.array(model.gravitational_forces_func(*q), dtype=float).ravel()
        ddq = np.linalg.solve(M_vals, tau - (C_vals + G_vals))

        # Enforce joint acceleration, velocity, and position limits.
        ddq = model.enforce_joint_acceleration_limits(ddq)
        q_dot = model.enforce_joint_velocity_limits(q_dot + ddq * dt)
        q = model.enforce_joint_position_limits(q + q_dot * dt)

        # Collect trajectory and info data.
        current_ee = model.fk_func(*q).flatten()
        model.collect_trajectory_info(q, q_dot, current_ee, step, dt)

        # Render the model for visualization.
        model.draw_model(np.rad2deg(q), in_deg=True, info=model.last_info)
        print(f"Step {step:03d} | q = {q} | tau = {tau}")
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    # 6) Play back the recorded trajectory.
    model.play_trajectory(playback_duration=2.0, decimation=1)


def test_task_space_controller_PID_rendering(nd=2):
    model = ArmModel(nd=nd, use_gravity=0)
    dt = 0.01
    num_steps = 200

    if nd == 2:
        # initial joint configuration (feasible)
        q = np.deg2rad([0.0, 15.0])
        q_dot = np.zeros(2)
    else:
        raise ValueError("Only demonstrating for nd=2 here")

    # Increase gains for faster convergence and add integral gain.
    Kp = np.eye(2) * 50
    Kd = np.eye(2) * 5
    Ki = np.eye(2) * 5  # Integral gain

    # Generate a desired target that is feasible.
    q_des_feasible = model.enforce_joint_position_limits(q + np.deg2rad([45.0, 70.0]))
    x_des = model.fk_func(*q_des_feasible).flatten()

    # Enforce workspace limits on x_des.
    xmin, xmax, ymin, ymax = model.allowed_workspace_bounds()
    x_des[0] = np.clip(x_des[0], xmin, xmax)
    x_des[1] = np.clip(x_des[1], ymin, ymax)

    dot_x_des = np.zeros(2)
    ddot_x_des = np.zeros(2)

    # Initialize trajectory and info data
    model.trajectory = {
        "time": [],
        "end_effector": [],
        "joint_angles": [],
        "joint_velocities": [],
        "joint_accelerations": [],
        "joint_jerks": [],
        "torques": [],
        "pos_error": [],
        "reward": [],
    }
    model.last_info = {}

    plt.ion()
    for step in range(num_steps):
        # 1) Get torque command from the PID controller.
        # Note: Use task_space_PID_controller which updates an internal integral error.
        tau_cmd = model.task_space_PID_controller(
            q,
            q_dot,
            x_des=x_des,
            dot_x_des=dot_x_des,
            ddot_x_des=ddot_x_des,
            Kp=Kp,
            Kd=Kd,
            Ki=Ki,
            dt=dt,
            return_torque=True,
        )
        tau_vec = np.array(tau_cmd, dtype=float).ravel()

        # 2) Solve forward dynamics to obtain ddq
        M_full = np.array(model.mass_matrix_func(*q), dtype=float)
        C_full = np.array(model.coriolis_forces_func(q, q_dot), dtype=float).ravel()
        G_full = np.array(model.gravitational_forces_func(*q), dtype=float).ravel()
        #("G_full", G_full)
        M_t = torch.as_tensor(M_full, dtype=torch.float32, device=device)
        tau_t = torch.as_tensor(tau_vec, dtype=torch.float32, device=device)
        CG_t = torch.as_tensor(C_full + G_full, dtype=torch.float32, device=device)

        ddq_t = torch.linalg.solve(M_t, tau_t - CG_t)
        ddq = ddq_t.cpu().numpy()

        # 3) Enforce joint acceleration, velocity, and position limits
        ddq = model.enforce_joint_acceleration_limits(ddq)
        q_dot = model.enforce_joint_velocity_limits(q_dot + ddq * dt)
        q = model.enforce_joint_position_limits(q + q_dot * dt)

        # 4) Collect trajectory and info data:
        model.trajectory["time"].append(step * dt)
        current_ee = model.fk_func(*q).flatten()
        model.trajectory["end_effector"].append(current_ee)
        model.trajectory["joint_angles"].append(q.copy())
        model.trajectory["joint_velocities"].append(q_dot.copy())
        model.trajectory["joint_accelerations"].append(ddq.copy())
        model.trajectory["joint_jerks"].append(np.zeros_like(ddq))
        model.trajectory["torques"].append(tau_vec.copy())
        model.trajectory["pos_error"].append(np.linalg.norm(x_des - current_ee))
        model.trajectory["reward"].append(0)  # Update with actual reward if needed

        # 5) Populate 'last_info' with expected fields
        model.last_info["current_state"] = np.concatenate([q, q_dot, ddq])
        model.last_info["torque"] = tau_vec
        model.last_info["joint_jerk_cmd"] = np.zeros_like(q)
        model.last_info["pos_error"] = np.linalg.norm(x_des - current_ee)
        model.last_info["target"] = x_des
        model.last_info["end_effector"] = current_ee

        # 6) Draw the model; note we pass q in degrees if in_deg=True.
        model.draw_model(np.rad2deg(q), in_deg=True, info=model.last_info)
        print(f"Step {step:03d} | q(rad)={q} | torque={tau_vec}")

        plt.pause(0.001)

    plt.ioff()
    plt.show()

    # 7) Play the recorded trajectory.
    model.play_trajectory(playback_duration=2.0, decimation=5)


def test_joint_and_torque_space_controller():
    """
    Test the joint_space_controller (without acceleration error feedback) in both
    joint-space (direct acceleration command) and torque (computed torque) modes for
    each controller type (PD and PID) on 2R and 3R robots.

    For each controller type, the test is run in two modes:
      - Joint Test: the controller returns a commanded joint acceleration.
      - Torque Test: the controller returns a computed torque command.

    The function simulates the closed-loop dynamics, collects trajectory data,
    renders the model configuration at each step, and finally uses the updated
    draw_model(...) method (with play_trajectory_flag=True) to animate the
    recorded end-effector trajectory and auto-close both figures.
    """
    # Supported controller modes.
    scenarios = ["PD", "PID"]
    dt = 0.01  # time step [s]
    simulation_time = 2.0  # total simulation duration [s]
    num_steps = int(simulation_time / dt)

    for nd in [2, 3]:
        print(f"\nTesting {nd}R robot with joint_space_controller...")
        # Create a new ArmModel instance.
        model = ArmModel(nd=nd, version="V0")

        # Initialize trajectory and info data.
        model.trajectory = {
            "time": [],
            "end_effector": [],
            "joint_angles": [],
            "joint_velocities": [],
            "joint_accelerations": [],
            "joint_jerks": [],
            "torques": [],
            "pos_error": [],
            "reward": [],
        }
        model.last_info = {}
        plt.ion()  # Enable interactive mode for live drawing

        # Compute desired joint target and corresponding end-effector target.
        q_des = model.enforce_joint_position_limits(model.reference_pose)
        x_des = model.fk_func(*q_des).flatten()
        xmin, xmax, ymin, ymax = model.allowed_workspace_bounds()
        x_des[0] = np.clip(x_des[0], xmin, xmax)
        x_des[1] = np.clip(x_des[1], ymin, ymax)

        for ctrl_type in scenarios:
            # ===== Joint Test: Direct acceleration command =====
            print(f"  Running scenario: {ctrl_type} (Joint Test)")
            state = model.state0.copy()  # state = [q, q_dot]
            model.collect_trajectory_info(
                state[:nd],
                state[nd : 2 * nd],
                model.forward_kinematics(state[:nd]),
                0,
                dt,
                initialize=True,
            )
            if ctrl_type == "PID":
                model.integral_error = torch.zeros(nd, dtype=torch.float32, device=device)

            for step in range(num_steps):
                ddot_q_cmd = model.joint_space_controller(
                    q=state[:nd],
                    q_dot=state[nd : 2 * nd],
                    q_des=model.reference_pose,
                    dot_q_des=None,
                    ddot_q_des=None,
                    Kp=None,
                    Kd=None,
                    Ki=None,
                    controller_type=ctrl_type,
                    dt=dt,
                    return_torque=False,
                )
                # Preliminary Euler update
                ddq = model.enforce_joint_acceleration_limits(ddot_q_cmd)
                q_dot_new = model.enforce_joint_velocity_limits(state[nd : 2 * nd] + ddq * dt)
                q_new = model.enforce_joint_position_limits(state[:nd] + q_dot_new * dt)
                state[:nd] = q_new
                state[nd : 2 * nd] = q_dot_new

                # 4) Collect trajectory and info data.
                model.trajectory["time"].append(step * dt)
                current_ee = model.fk_func(*state[:nd]).flatten()
                model.trajectory["end_effector"].append(current_ee)
                model.trajectory["joint_angles"].append(state[:nd].copy())
                model.trajectory["joint_velocities"].append(state[nd : 2 * nd].copy())
                model.trajectory["joint_accelerations"].append(ddq.copy())
                model.trajectory["joint_jerks"].append(np.zeros_like(ddq))
                # In joint test, no torque command is applied.
                model.trajectory["torques"].append(np.zeros(nd))
                model.trajectory["pos_error"].append(np.linalg.norm(x_des - current_ee))
                model.trajectory["reward"].append(0)

                # 5) Populate last_info for display.
                model.last_info["current_state"] = np.concatenate([state[:nd], state[nd : 2 * nd], ddq])
                model.last_info["torque"] = np.zeros(nd)
                model.last_info["joint_jerk_cmd"] = np.zeros_like(state[:nd])
                model.last_info["pos_error"] = np.linalg.norm(x_des - current_ee)
                model.last_info["target"] = x_des
                model.last_info["end_effector"] = current_ee

                # 6) Draw the model and print step info.
                model.draw_model(np.rad2deg(state[:nd]), in_deg=True, info=model.last_info)
                print(f"Step {step:03d} | q(rad)={state[:nd]} | torque=0")
                plt.pause(0.001)

            # Instead of calling model.play_trajectory(...) directly,
            # we do a final call to draw_model(...) with play_trajectory_flag=True.
            # This will auto-call play_trajectory(...) inside draw_model after drawing.
            print(f"    Rendering joint test trajectory for {nd}R using {ctrl_type} controller...")
            model.draw_model(
                np.rad2deg(state[:nd]),
                in_deg=True,
                info=model.last_info,
                play_trajectory_flag=True,  # triggers the auto-call to play_trajectory
                playback_duration=2.0,  # how long to run the animation
                decimation=1,  # decimation for trajectory
            )

            # ===== Torque Test: Computed torque command =====
            print(f"  Running scenario: {ctrl_type} (Torque Test)")
            state = model.state0.copy()  # Reset state for torque test
            model.collect_trajectory_info(
                state[:nd],
                state[nd : 2 * nd],
                model.forward_kinematics(state[:nd]),
                0,
                dt,
                initialize=True,
            )
            if ctrl_type == "PID":
                model.integral_error = torch.zeros(nd, dtype=torch.float32, device=device)

            for step in range(num_steps):
                tau_cmd = model.joint_space_controller(
                    q=state[:nd],
                    q_dot=state[nd : 2 * nd],
                    q_des=model.reference_pose,
                    dot_q_des=None,
                    ddot_q_des=None,
                    Kp=None,
                    Kd=None,
                    Ki=None,
                    controller_type=ctrl_type,
                    dt=dt,
                    return_torque=True,
                )
                tau_vec = np.array(tau_cmd, dtype=float).ravel()

                # Retrieve dynamics
                M_vals = np.array(model.mass_matrix_func(*state[:nd]), dtype=float)
                C_vals = np.array(
                    model.coriolis_forces_func(state[:nd], state[nd : 2 * nd]),
                    dtype=float,
                ).ravel()
                G_vals = np.array(model.gravitational_forces_func(*state[:nd]), dtype=float).ravel()
                print("G_vals", G_vals)
                ddot_q = np.linalg.solve(M_vals, tau_vec - (C_vals + G_vals))

                # Enforce joint limits and update state
                ddq = model.enforce_joint_acceleration_limits(ddot_q)
                q_dot_new = model.enforce_joint_velocity_limits(state[nd : 2 * nd] + ddq * dt)
                q_new = model.enforce_joint_position_limits(state[:nd] + q_dot_new * dt)
                state[:nd] = q_new
                state[nd : 2 * nd] = q_dot_new

                # Collect trajectory info
                model.trajectory["time"].append(step * dt)
                current_ee = model.fk_func(*state[:nd]).flatten()
                model.trajectory["end_effector"].append(current_ee)
                model.trajectory["joint_angles"].append(state[:nd].copy())
                model.trajectory["joint_velocities"].append(state[nd : 2 * nd].copy())
                model.trajectory["joint_accelerations"].append(ddq.copy())
                model.trajectory["joint_jerks"].append(np.zeros_like(ddq))
                model.trajectory["torques"].append(tau_vec.copy())
                model.trajectory["pos_error"].append(np.linalg.norm(x_des - current_ee))
                model.trajectory["reward"].append(0)

                # Populate last_info.
                model.last_info["current_state"] = np.concatenate([state[:nd], state[nd : 2 * nd], ddq])
                model.last_info["torque"] = tau_vec
                model.last_info["joint_jerk_cmd"] = np.zeros_like(state[:nd])
                model.last_info["pos_error"] = np.linalg.norm(x_des - current_ee)
                model.last_info["target"] = x_des
                model.last_info["end_effector"] = current_ee

                # Draw the model and print step information.
                model.draw_model(np.rad2deg(state[:nd]), in_deg=True, info=model.last_info)
                print(f"Step {step:03d} | q(rad)={state[:nd]} | torque={tau_vec}")
                plt.pause(0.001)

            # Again, we do a final call to draw_model(...) with the local flag.
            print(f"    Rendering torque test trajectory for {nd}R using {ctrl_type} controller...")
            model.draw_model(
                np.rad2deg(state[:nd]),
                in_deg=True,
                info=model.last_info,
                play_trajectory_flag=True,  # triggers the auto-call to play_trajectory
                playback_duration=2.0,
                decimation=1,
            )

        plt.ioff()
        plt.show()



def test_construct_muscle_geometry_comparisonO(nd=2):
    """
    Create two ArmModel instances: one using the default (sqrt-based) geometry
    and another using an alternate arcsin/arccos approach. Then compare their
    computed muscle lengths, velocities, accelerations, and R-related matrices
    at several sample joint configurations.

    Args:
        nd (int): number of DoFs (2 or 3). Defaults to 2 in this example.
    """
    print("\n[TEST] Construct Muscle Geometry Comparison")

    # 1) Create first model (default sqrt-based approach).
    print(" -- Creating default (sqrt-based) geometry model --")
    model_default = ArmModel(nd=nd, version="CompareDefault")
    # Force it to build geometry with the default representation.
    model_default.construct_muscle_geometry(use_alt_representation=False)

    # 2) Create second model (alternate arcsin/arccos approach).
    print(" -- Creating alternate (arcsin/arccos) geometry model --")
    model_alt = ArmModel(nd=nd, version="CompareAlt")
    model_alt.construct_muscle_geometry(use_alt_representation=True)

    # 3) Define a few sample joint configurations to test:
    if nd == 3:
        test_configs = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.2, -0.3],
        ]
    else:
        test_configs = [
            [0.0, 0.0],
            [0.4, 0.9],
        ]

    # We'll define some small fixed q_dot, q_ddot so that muscle velocities
    # and accelerations are not trivial. Adjust as you like.
    q_dot_arr = 0.1 * np.ones(nd)
    q_ddot_arr = 0.2 * np.ones(nd)

    # 4) Evaluate muscle geometry for each config in both models, compare results.
    for idx, q_vec in enumerate(test_configs):
        q_arr = np.array(q_vec, dtype=float)

        # Default model: lengths, velocities, accelerations, R, RDot, RDotQDot
        lengths_def = model_default.muscle_lengths_func(*q_arr)
        vels_def = model_default.muscle_velocities_func(q_arr, q_dot_arr)
        acc_def = model_default.muscle_acceleration_func(q_arr, q_dot_arr, q_ddot_arr)
        R_def = model_default.R_func(*q_arr)
        RDot_def = model_default.RDot_func(q_arr, q_dot_arr)
        RDotQDot_def = model_default.RDotQDot_func(q_arr, q_dot_arr)

        # Alternate model: lengths, velocities, accelerations, R, RDot, RDotQDot
        lengths_alt = model_alt.muscle_lengths_func(*q_arr)
        vels_alt = model_alt.muscle_velocities_func(q_arr, q_dot_arr)
        acc_alt = model_alt.muscle_acceleration_func(q_arr, q_dot_arr, q_ddot_arr)
        R_alt = model_alt.R_func(*q_arr)
        RDot_alt = model_alt.RDot_func(q_arr, q_dot_arr)
        RDotQDot_alt = model_alt.RDotQDot_func(q_arr, q_dot_arr)

        # Convert to numpy for printing
        lengths_def = np.array(lengths_def, dtype=float).flatten()
        lengths_alt = np.array(lengths_alt, dtype=float).flatten()
        vels_def = np.array(vels_def, dtype=float).flatten()
        vels_alt = np.array(vels_alt, dtype=float).flatten()
        acc_def = np.array(acc_def, dtype=float).flatten()
        acc_alt = np.array(acc_alt, dtype=float).flatten()

        # Print them out
        print(f"\nTest #{idx+1}: q = {q_arr} (radians)")
        print(f"  -- Default (sqrt-based) --")
        print(f"     lengths:       {np.round(lengths_def, 5)}")
        print(f"     velocities:    {np.round(vels_def, 5)}  (q_dot={q_dot_arr})")
        print(f"     accelerations: {np.round(acc_def, 5)}  (q_ddot={q_ddot_arr})")
        print(f"     R:\n{np.round(R_def, 5)}")
        print(f"     RDot:\n{np.round(RDot_def, 5)}")
        print(f"     RDotQDot: {np.round(RDotQDot_def, 5)}")

        print(f"  -- Alternate (arcsin/arccos) --")
        print(f"     lengths:       {np.round(lengths_alt, 5)}")
        print(f"     velocities:    {np.round(vels_alt, 5)}")
        print(f"     accelerations: {np.round(acc_alt, 5)}")
        print(f"     R:\n{np.round(R_alt, 5)}")
        print(f"     RDot:\n{np.round(RDot_alt, 5)}")
        print(f"     RDotQDot: {np.round(RDotQDot_alt, 5)}")

        # Optionally compute differences in lengths:
        diff_len = lengths_alt - lengths_def
        print(f"  Difference in lengths (alt - default): {np.round(diff_len, 5)}")

    print("[TEST] Done comparing muscle geometry.\n")


def test_construct_muscle_geometry_comparison(nd=2):
    """
    Create two ArmModel instances: one using the default (sqrt‐based) geometry and
    another using the parabolic approximation for muscle lengths (via the unified 
    construct_muscle_geometry function with use_parabolic=True). Then compare their
    computed muscle lengths, velocities, accelerations, and R‐related matrices at
    several sample joint configurations.
    
    Args:
        nd (int): number of DoFs (2 or 3). Defaults to 2.
    """
    import numpy as np
    print("\n[TEST] Construct Muscle Geometry Comparison (Default vs Parabolic Approximation)")

    # 1) Create the default model.
    print(" -- Creating default (sqrt‐based) geometry model --")
    model_default = ArmModel(nd=nd, version="CompareDefault")
    model_default.construct_muscle_geometry(use_parabolic=False, use_alt_representation=False)

    # 2) Create the parabolic model.
    print(" -- Creating parabolic approximation geometry model --")
    model_parabolic = ArmModel(nd=nd, version="CompareParabolic")
    model_parabolic.construct_muscle_geometry(use_parabolic=True, h=0.09)

    # 3) Define sample joint configurations.
    if nd == 3:
        test_configs = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.2, -0.3],
        ]
    else:
        test_configs = [
            [0.0, 0.0],
            [0.4, 0.9],
        ]

    q_dot_arr = 0.1 * np.ones(nd)
    q_ddot_arr = 0.2 * np.ones(nd)

    # 4) Evaluate geometry for each configuration.
    for idx, q_vec in enumerate(test_configs):
        q_arr = np.array(q_vec, dtype=float)

        # Default geometry evaluation.
        lengths_def = np.array(model_default.muscle_lengths_func(*q_arr)).astype(float).flatten()
        vels_def = np.array(model_default.muscle_velocities_func(q_arr, q_dot_arr)).astype(float).flatten()
        acc_def = np.array(model_default.muscle_acceleration_func(q_arr, q_dot_arr, q_ddot_arr)).astype(float).flatten()
        R_def = np.array(model_default.R_func(*q_arr)).astype(float)
        RDot_def = np.array(model_default.RDot_func(q_arr, q_dot_arr)).astype(float)
        RDotQDot_def = np.array(model_default.RDotQDot_func(q_arr, q_dot_arr)).astype(float)

        # Parabolic geometry evaluation.
        lengths_par = np.array(model_parabolic.muscle_lengths_func(*q_arr)).astype(float).flatten()
        vels_par = np.array(model_parabolic.muscle_velocities_func(q_arr, q_dot_arr)).astype(float).flatten()
        acc_par = np.array(model_parabolic.muscle_acceleration_func(q_arr, q_dot_arr, q_ddot_arr)).astype(float).flatten()
        R_par = np.array(model_parabolic.R_func(*q_arr)).astype(float)
        RDot_par = np.array(model_parabolic.RDot_func(q_arr, q_dot_arr)).astype(float)
        RDotQDot_par = np.array(model_parabolic.RDotQDot_func(q_arr, q_dot_arr)).astype(float)

        print(f"\nTest #{idx+1}: q = {q_arr} (radians)")
        print("  -- Default (sqrt‐based) --")
        print(f"     lengths:       {np.round(lengths_def, 5)}")
        print(f"     velocities:    {np.round(vels_def, 5)}  (q_dot={q_dot_arr})")
        print(f"     accelerations: {np.round(acc_def, 5)}  (q_ddot={q_ddot_arr})")
        print(f"     R:\n{np.round(R_def, 5)}")
        print(f"     RDot:\n{np.round(RDot_def, 5)}")
        print(f"     RDotQDot: {np.round(RDotQDot_def, 5)}")

        print("  -- Parabolic Approximation --")
        print(f"     lengths:       {np.round(lengths_par, 5)}")
        print(f"     velocities:    {np.round(vels_par, 5)}")
        print(f"     accelerations: {np.round(acc_par, 5)}")
        print(f"     R:\n{np.round(R_par, 5)}")
        print(f"     RDot:\n{np.round(RDot_par, 5)}")
        print(f"     RDotQDot: {np.round(RDotQDot_par, 5)}")

        diff_len = lengths_par - lengths_def
        print(f"  Difference in lengths (parabolic - default): {np.round(diff_len, 5)}")

    print("[TEST] Done comparing muscle geometry.\n")






def main():
    GLOBAL_LOGGER("Starting test ")
    test_task_space_controller_by_muscles(nd=2)
    GLOBAL_LOGGER("Test complete.")



if __name__ == "__main__":
    main()
