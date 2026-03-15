"""
Sine Class (Sine function)
Modified from LogSig implementation
Adapted for sine function: f(x) = sin(x)

Key changes:
- Activation function changed from sigmoid to sine
- Seven regions to handle monotonicity and convexity changes
- Properly handles decreasing regions and extrema crossings

Author: Zhuoyang Zhou
Date: 9/30/2025
"""

# !/usr/bin/python3
import numpy as np
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar


class Sine(object):
    """
    Sine Class for reachability analysis
    Activation function: f(x) = sin(x)

    Handles 7 regions based on monotonicity and convexity:
    Region 1: -π < lb < ub < -π/2    (decreasing, concave)
    Region 2: -π < lb < -π/2 < ub < 0 (crossing minimum, concave)
    Region 3: -π/2 < lb < ub < 0      (increasing, concave)
    Region 4: -π/2 < lb < 0 < ub < π/2 (crossing zero, changing convexity)
    Region 5: 0 < lb < ub < π/2       (increasing, convex)
    Region 6: 0 < lb < π/2 < ub < π   (crossing maximum, convex)
    Region 7: π/2 < lb < ub < π       (decreasing, convex)
    """

    @staticmethod
    def evaluate(x):
        """Evaluate sine function"""
        return Sine.f(x)

    @staticmethod
    def f(x):
        """Sine activation function: f(x) = sin(x)"""
        return np.sin(x)

    @staticmethod
    def df(x):
        """Derivative of sine function: f'(x) = cos(x)"""
        return np.cos(x)

    @staticmethod
    def reachApprox_star(I, opt=True, lp_solver='gurobi', RF=0.0):
        """
        Compute reachable set approximation for sine activation using Star sets

        Parameters:
        -----------
        I : Star
            Input Star set
        opt : bool
            Whether to use optimal approximation (reserved for future use)
        lp_solver : str
            Linear programming solver to use ('gurobi', 'linprog', etc.)
        RF : float
            Relaxation factor for range computation

        Returns:
        --------
        Star
            Output Star set after sine activation
        """

        assert isinstance(I, Star), 'error: input set is not a Star set'

        N = I.dim

        # Get input ranges for each dimension
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        # Compute sine function values and derivatives at bounds
        yl, yu = Sine.f(l), Sine.f(u)
        dyl, dyu = Sine.df(l), Sine.df(u)

        # Identify neurons where l != u (non-constant inputs)
        map0 = np.where(l != u)[0]
        m = len(map0)

        # Create new generator matrix for output variables
        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1

        # Properly preserve the original basis matrix structure
        new_V = np.hstack([np.zeros([N, 1]), np.zeros([N, I.nVars]), V0])

        # Handle constant neurons (l == u)
        map1 = np.where(l == u)[0]
        if len(map1):
            new_V[map1, 0] = yl[map1]
            new_V[map1, 1:] = 0

        nv = I.nVars + m

        # ========================================================================
        # Region 1: -π < lb < ub < -π/2 (decreasing, concave)
        # Monotonicity: DECREASING (so f(u) < f(l))
        # Convexity: CONCAVE (f'' > 0)
        # Constraints: Upper = secant, Lower = tangents
        # ========================================================================
        map1 = np.where((l[map0] >= -np.pi) & (u[map0] <= -np.pi/2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bound: tangents at l and u
            # y >= cos(l)*(x - l) + sin(l)
            C11 = np.hstack([dyl_diag @ V1, -V2])
            d11 = -dyl_ * (c1 - l_) - yl_

            # y >= cos(u)*(x - u) + sin(u)
            C12 = np.hstack([dyu_diag @ V1, -V2])
            d12 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant line
            # y <= (sin(u) - sin(l)) / (u - l) * (x - l) + sin(l)
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C13 = np.hstack([-g_diag @ V1, V2])
            d13 = g * (c1 - l_) + yl_

            # Additional lower bound: tangent at midpoint
            xo = 0.5 * (u_ + l_)
            dyo = Sine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C14 = np.hstack([dyo_diag @ V1, -V2])
            d14 = -dyo * (c1 - xo) - Sine.f(xo)

            C1 = np.vstack((C11, C12, C13, C14))
            d1 = np.hstack((d11, d12, d13, d14))
        else:
            C1 = np.empty((0, nv))
            d1 = np.empty((0))

        # ========================================================================
        # Region 2: -π < lb < -π/2, -π/2 < ub < 0 (crossing minimum at -π/2, concave)
        # Crosses minimum value sin(-π/2) = -1
        # Convexity: CONCAVE (f'' > 0)
        # Strategy: Use tangent method similar to crossing zero, but with minimum
        # ========================================================================
        map1 = np.where((l[map0] >= -np.pi) & (l[map0] < -np.pi/2) & (u[map0] > -np.pi/2) & (u[map0] <= 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bound: Must include minimum value -1
            # Use a simple lower bound: y >= -1
            C21 = np.hstack([np.zeros((len(map_), V1.shape[1])), -V2])
            d21 = np.ones(len(map_))

            # Additional lower bound: tangents at endpoints
            C22 = np.hstack([dyl_diag @ V1, -V2])
            d22 = -dyl_ * (c1 - l_) - yl_

            C23 = np.hstack([dyu_diag @ V1, -V2])
            d23 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant line
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C24 = np.hstack([-g_diag @ V1, V2])
            d24 = g * (c1 - l_) + yl_

            C2 = np.vstack((C21, C22, C23, C24))
            d2 = np.hstack((d21, d22, d23, d24))
        else:
            C2 = np.empty((0, nv))
            d2 = np.empty((0))

        # ========================================================================
        # Region 3: -π/2 < lb < ub < 0 (increasing, concave)
        # Monotonicity: INCREASING
        # Convexity: CONCAVE (f'' > 0)
        # Constraints: Upper = secant, Lower = tangents
        # ========================================================================
        map1 = np.where((l[map0] >= -np.pi/2) & (u[map0] <= 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bounds: tangents
            C31 = np.hstack([dyl_diag @ V1, -V2])
            d31 = -dyl_ * (c1 - l_) - yl_

            C32 = np.hstack([dyu_diag @ V1, -V2])
            d32 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C33 = np.hstack([-g_diag @ V1, V2])
            d33 = g * (c1 - l_) + yl_

            # Additional lower bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Sine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C34 = np.hstack([dyo_diag @ V1, -V2])
            d34 = -dyo * (c1 - xo) - Sine.f(xo)

            C3 = np.vstack((C31, C32, C33, C34))
            d3 = np.hstack((d31, d32, d33, d34))
        else:
            C3 = np.empty((0, nv))
            d3 = np.empty((0))

        # ========================================================================
        # Region 4: -π/2 < lb < 0 < ub < π/2 (crossing zero, changing convexity)
        # Same as before - using quadrilateral method with y=x
        # ========================================================================
        map1 = np.where((l[map0] >= -np.pi/2) & (l[map0] < 0) & (u[map0] > 0) & (u[map0] <= np.pi/2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bounds: tangents
            C41 = np.hstack([dyl_diag @ V1, -V2])
            d41 = -dyl_ * (c1 - l_) - yl_

            C42 = np.hstack([dyu_diag @ V1, -V2])
            d42 = -dyu_ * (c1 - u_) - yu_

            # Calculate intersections with y=x
            epsilon = 1e-10
            denom_l = np.maximum(1 - dyl_, epsilon)
            denom_u = np.maximum(1 - dyu_, epsilon)

            xl_int = (yl_ - l_ * dyl_) / denom_l
            xu_int = (yu_ - u_ * dyu_) / denom_u

            # Upper bounds: lines forming quadrilateral
            slope3 = (yu_ - xl_int) / (u_ - xl_int)
            slope3_diag = np.diag(slope3.flatten())
            C43 = np.hstack([-slope3_diag @ V1, V2])
            d43 = slope3 * (c1 - xl_int) + xl_int

            slope4 = (xu_int - yl_) / (xu_int - l_)
            slope4_diag = np.diag(slope4.flatten())
            C44 = np.hstack([-slope4_diag @ V1, V2])
            d44 = slope4 * (c1 - l_) + yl_

            C4 = np.vstack((C41, C42, C43, C44))
            d4 = np.hstack((d41, d42, d43, d44))
        else:
            C4 = np.empty((0, nv))
            d4 = np.empty((0))

        # ========================================================================
        # Region 5: 0 < lb < ub < π/2 (increasing, convex)
        # Monotonicity: INCREASING
        # Convexity: CONVEX (f'' < 0)
        # Constraints: Upper = tangents, Lower = secant
        # ========================================================================
        map1 = np.where((l[map0] >= 0) & (u[map0] <= np.pi/2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bounds: tangents
            C51 = np.hstack([-dyl_diag @ V1, V2])
            d51 = dyl_ * (c1 - l_) + yl_

            C52 = np.hstack([-dyu_diag @ V1, V2])
            d52 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C53 = np.hstack([g_diag @ V1, -V2])
            d53 = -g * (c1 - l_) - yl_

            # Additional upper bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Sine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C54 = np.hstack([-dyo_diag @ V1, V2])
            d54 = dyo * (c1 - xo) + Sine.f(xo)

            C5 = np.vstack((C51, C52, C53, C54))
            d5 = np.hstack((d51, d52, d53, d54))
        else:
            C5 = np.empty((0, nv))
            d5 = np.empty((0))

        # ========================================================================
        # Region 6: 0 < lb < π/2, π/2 < ub < π (crossing maximum at π/2, convex)
        # Crosses maximum value sin(π/2) = 1
        # Convexity: CONVEX (f'' < 0)
        # Strategy: Upper bound must include maximum value 1
        # ========================================================================
        map1 = np.where((l[map0] >= 0) & (l[map0] < np.pi/2) & (u[map0] > np.pi/2) & (u[map0] <= np.pi))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bound: Must not exceed maximum value 1
            # y <= 1
            C61 = np.hstack([np.zeros((len(map_), V1.shape[1])), V2])
            d61 = np.ones(len(map_))

            # Additional upper bounds: tangents at endpoints
            C62 = np.hstack([-dyl_diag @ V1, V2])
            d62 = dyl_ * (c1 - l_) + yl_

            C63 = np.hstack([-dyu_diag @ V1, V2])
            d63 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant line
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C64 = np.hstack([g_diag @ V1, -V2])
            d64 = -g * (c1 - l_) - yl_

            C6 = np.vstack((C61, C62, C63, C64))
            d6 = np.hstack((d61, d62, d63, d64))
        else:
            C6 = np.empty((0, nv))
            d6 = np.empty((0))

        # ========================================================================
        # Region 7: π/2 < lb < ub < π (decreasing, convex)
        # Monotonicity: DECREASING
        # Convexity: CONVEX (f'' < 0)
        # Constraints: Upper = tangents, Lower = secant
        # ========================================================================
        map1 = np.where((l[map0] >= np.pi/2) & (u[map0] <= np.pi))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bounds: tangents
            C71 = np.hstack([-dyl_diag @ V1, V2])
            d71 = dyl_ * (c1 - l_) + yl_

            C72 = np.hstack([-dyu_diag @ V1, V2])
            d72 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C73 = np.hstack([g_diag @ V1, -V2])
            d73 = -g * (c1 - l_) - yl_

            # Additional upper bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Sine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C74 = np.hstack([-dyo_diag @ V1, V2])
            d74 = dyo * (c1 - xo) + Sine.f(xo)

            C7 = np.vstack((C71, C72, C73, C74))
            d7 = np.hstack((d71, d72, d73, d74))
        else:
            C7 = np.empty((0, nv))
            d7 = np.empty((0))

        # ========================================================================
        # Combine all constraints
        # ========================================================================
        n = I.C.shape[0]
        if len(I.d):
            C0 = np.hstack([I.C, np.zeros([n, m])])
            d0 = I.d
        else:
            C0 = np.empty([0, I.nVars + m])
            d0 = np.empty([0])

        new_C = np.vstack((C0, C1, C2, C3, C4, C5, C6, C7))
        new_d = np.hstack((d0, d1, d2, d3, d4, d5, d6, d7))

        # new_pred_lb = np.hstack((I.pred_lb, yl[map0]))
        # new_pred_ub = np.hstack((I.pred_ub, yu[map0]))

        # ---- Robust β-predicate bounds for all regions (single pass) ----
        idx = map0  # Only set bounds for the newly added m β dimensions
        yL = np.minimum(yl[idx], yu[idx])  # Get min/max by endpoint
        yU = np.maximum(yl[idx], yu[idx])

        # If the interval passes through the global minimum/maximum point, clip it to -1/+1
        cross_min = (l[idx] < -np.pi / 2) & (u[idx] > -np.pi / 2)  # Passing sin minimum -1
        cross_max = (l[idx] < np.pi / 2) & (u[idx] > np.pi / 2)  # Passing sin maximum +1
        if np.any(cross_min):
            yL[cross_min] = -1.0
        if np.any(cross_max):
            yU[cross_max] = 1.0

        new_pred_lb = np.hstack((I.pred_lb, yL))
        new_pred_ub = np.hstack((I.pred_ub, yU))

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    @staticmethod
    def reach(I, opt=False, lp_solver='gurobi', RF=0.0):
        """
        Main entry point for reachability analysis with sine activation

        Parameters:
        -----------
        I : Star or ImageStar
            Input set
        opt : bool
            Whether to use optimal approximation (reserved)
        lp_solver : str
            Linear programming solver
        RF : float
            Relaxation factor

        Returns:
        --------
        Star or ImageStar
            Output reachable set
        """
        if isinstance(I, Star):
            return Sine.reachApprox_star(I, opt=opt, lp_solver=lp_solver, RF=RF)
        elif isinstance(I, ImageStar):
            shape = I.shape()
            S = Sine.reachApprox_star(I.toStar(), opt=opt, lp_solver=lp_solver, RF=RF)
            return S.toImageStar(image_shape=shape, copy_=False)
        else:
            raise Exception('error: unknown input set type')