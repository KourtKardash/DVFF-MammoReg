import cv2
import numpy as np


class CurvilinearCoordinatesBuilder:
    def __init__(self,
                 i,
                 orientation: str,
                 mask_path: str,
                 mammogram_path: str
        ) -> None:
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)
        self.orientation = orientation
        self.height, self.width = self.mask.shape
        self.start_parameter = i
        if self.orientation == "L":
            self.mask = np.flip(self.mask, axis=1)
            self.mammogram = np.flip(self.mammogram, axis=1)

        target_size = (896, 1152)
        if (self.width, self.height) != target_size:
            self.mask = cv2.resize(
                self.mask, target_size, interpolation=cv2.INTER_LINEAR
            )
            self.mammogram = cv2.resize(
                self.mammogram, target_size, interpolation=cv2.INTER_LINEAR
            )
            self.height, self.width = target_size[1], target_size[0]

        self.upper_poly_x, self.upper_poly_y, self.lower_poly_x, self.lower_poly_y = self._get_approximate_curves()
        self.upper_poly_dx, self.upper_poly_dy = self.upper_poly_x.deriv(), self.upper_poly_y.deriv()
        self.lower_poly_dx, self.lower_poly_dy = self.lower_poly_x.deriv(), self.lower_poly_y.deriv()

        self.upper_poly_d2x, self.upper_poly_d2y = self.upper_poly_x.deriv(2), self.upper_poly_y.deriv(2)
        self.lower_poly_d2x, self.lower_poly_d2y = self.lower_poly_x.deriv(2), self.lower_poly_y.deriv(2)


    def _get_approximate_curves(self):
        upper_points, lower_points = self._get_contour_points()
        upper_distances = self._normalize_distances(upper_points)
        lower_distances = self._normalize_distances(lower_points)
        upper_poly_x = self._fit_poly(upper_distances, upper_points[:, 0])
        upper_poly_y = self._fit_poly(upper_distances, upper_points[:, 1])

        lower_poly_x = self._fit_poly(lower_distances, lower_points[:, 0])
        lower_poly_y = self._fit_poly(lower_distances, lower_points[:, 1])
        return upper_poly_x, upper_poly_y, lower_poly_x, lower_poly_y


    def _get_contour_points(self):
        non_zero_columns = np.argmax(self.mask > 0, axis=1)
        non_zero_rows = np.arange(self.mask.shape[0])

        valid_rows = non_zero_columns > 0
        points = np.column_stack((non_zero_columns[valid_rows], non_zero_rows[valid_rows]))

        delta = points[-1][1] - points[0][1]
        self.start_system = points[int(round(self.start_parameter * delta))]
        upper_points = points[points[:, 1] <= self.start_system[1]][::-1]
        lower_points = points[points[:, 1] >= self.start_system[1]]

        upper_points_shifted = upper_points - self.start_system
        lower_points_shifted = lower_points - self.start_system

        return upper_points_shifted, lower_points_shifted
    

    @staticmethod
    def _normalize_distances(points):
        distances = np.linalg.norm(points, axis=1)
        return distances / distances.max() if distances.max() != 0 else distances
    

    @staticmethod
    def _fit_poly(distances, values, degree=4):
        A = np.column_stack([distances ** i for i in range(1, degree + 1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        poly = np.poly1d(np.concatenate([coeffs[::-1], [0]]))
        return poly


    @staticmethod
    def calculate_angle(poly_dx, poly_dy, s):
        dx = poly_dx(s)
        dy = poly_dy(s)

        theta = np.arctan2(dy, dx)
        return theta


    def auxiliary_curve(self, s, t):
        theta0 = self.calculate_angle(self.upper_poly_dx, self.upper_poly_dy, s)
        theta1 = self.calculate_angle(self.lower_poly_dx, self.lower_poly_dy, s)

        x0 = self.upper_poly_x(s)
        y0 = self.upper_poly_y(s)
        x1 = self.lower_poly_x(s)
        y1 = self.lower_poly_y(s)
        
        x_hat = (1-t) * (x0 * np.cos(theta0) + y0 * np.sin(theta0)) + t * (x1 * np.cos(theta1) + y1 * np.sin(theta1))
        y_hat = (1-t) * (y0 * np.cos(theta0) - x0 * np.sin(theta0)) + t * (y1 * np.cos(theta1) - x1 * np.sin(theta1))

        return x_hat, y_hat, theta0, theta1


    def curvilinear_to_cartesian(self, s, t):
        x_hat, y_hat, theta0, theta1 = self.auxiliary_curve(s, t)
        theta_t = (theta1 - theta0) * t + theta0
        x = x_hat * np.cos(theta_t) - y_hat * np.sin(theta_t)
        y = x_hat * np.sin(theta_t) + y_hat * np.cos(theta_t)
        return x, y
    

    @staticmethod
    def dthetai_ds(poly_dx, poly_dy, poly_d2x, poly_d2y, s):
        dx = poly_dx(s)
        dy = poly_dy(s)

        dxdx = poly_d2x(s)
        dydy = poly_d2y(s)
        return (dx*dydy - dxdx*dy)/(dx*dx + dy*dy)      


    @staticmethod
    def theta(theta0, theta1, t):
        return (theta1 - theta0) * t + theta0


    def get_J(self, s, t):
        x_aux, y_aux, theta0, theta1 = self.auxiliary_curve(s, t)
        dtheta0_ds = self.dthetai_ds(self.upper_poly_dx, self.upper_poly_dy, 
                                    self.upper_poly_d2x, self.upper_poly_d2y, s)
        dtheta1_ds = self.dthetai_ds(self.lower_poly_dx, self.lower_poly_dy,
                                    self.lower_poly_d2x, self.lower_poly_d2y, s)
        theta = self.theta(theta0, theta1, t)

        dA = self.upper_poly_dx(s) * np.cos(theta0) + self.upper_poly_dy(s) * np.sin(theta0) + \
             dtheta0_ds * (self.upper_poly_y(s) * np.cos(theta0) - self.upper_poly_x(s) * np.sin(theta0))
        dB = self.lower_poly_dx(s) * np.cos(theta1) + self.lower_poly_dy(s) * np.sin(theta1) + \
             dtheta1_ds * (self.lower_poly_y(s) * np.cos(theta1) - self.lower_poly_x(s) * np.sin(theta1))
        dC = self.upper_poly_dy(s) * np.cos(theta0) - self.upper_poly_dx(s) * np.sin(theta0) - \
             dtheta0_ds * (self.upper_poly_y(s) * np.sin(theta0) + self.upper_poly_x(s) * np.cos(theta0))
        dD = self.lower_poly_dy(s) * np.cos(theta1) - self.lower_poly_dx(s) * np.sin(theta1) - \
             dtheta1_ds * (self.lower_poly_y(s) * np.sin(theta1) + self.lower_poly_x(s) * np.cos(theta1))

        aux_dx_ds = (1 - t) * dA + t * dB
        aux_dx_dt = -(self.upper_poly_x(s) * np.cos(theta0) + self.upper_poly_y(s) * np.sin(theta0)) + \
                    (self.lower_poly_x(s) + np.cos(theta1) + self.lower_poly_y(s) * np.sin(theta1))
        aux_dy_ds = (1 - t) * dC + t * dD
        aux_dy_dt = -(self.upper_poly_y(s) * np.cos(theta0) - self.upper_poly_x(s) * np.sin(theta0)) + \
                    (self.lower_poly_y(s) * np.cos(theta1) - self.lower_poly_x(s) * np.sin(theta1))

        dx_ds = aux_dx_ds * np.cos(theta) - x_aux * np.sin(theta) * ((dtheta1_ds - dtheta0_ds) * t + dtheta0_ds) - \
                (aux_dy_ds * np.sin(theta) + y_aux * np.cos(theta) * ((dtheta1_ds - dtheta0_ds) * t + dtheta0_ds))
        dx_dt = aux_dx_dt * np.cos(theta) - x_aux * np.sin(theta) * theta1 - \
                (aux_dy_dt * np.sin(theta) + y_aux * np.cos(theta) * theta1)
        dy_ds = aux_dx_ds * np.sin(theta) + x_aux * np.cos(theta) * ((dtheta1_ds - dtheta0_ds) * t + dtheta0_ds) + \
                aux_dy_ds * np.cos(theta) - y_aux * np.sin(theta) * ((dtheta1_ds - dtheta0_ds) * t + dtheta0_ds)
        dy_dt = aux_dx_dt * np.sin(theta) + x_aux * np.cos(theta) * theta1 + \
                aux_dy_dt * np.cos(theta) - y_aux * np.sin(theta) * theta1
        #J = np.array([[dx_ds, dx_dt],
        #              [dy_ds, dy_dt]])
        N = s.shape[0]
        J = np.zeros((N, 2, 2))
        J[:, 0, 0] = dx_ds
        J[:, 0, 1] = dx_dt
        J[:, 1, 0] = dy_ds
        J[:, 1, 1] = dy_dt
        return J
    

    @staticmethod
    def solve_linear_systems_vectorized_simple(J_matrices, R_vectors):
        a, b = J_matrices[:, 0, 0], J_matrices[:, 0, 1] 
        c, d = J_matrices[:, 1, 0], J_matrices[:, 1, 1]
        b1, b2 = R_vectors[:, 0], R_vectors[:, 1]
        
        det = a * d - b * c
        
        #det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        
        x = (d * b1 - b * b2) / det
        y = (a * b2 - c * b1) / det
        
        return np.column_stack([x, y])


    def newton_method_inverse_mapping_vectorized(self, x_targets, y_targets, s0, t0, tol=1e-3, max_iter=100000):
        n_points = len(x_targets)
        p = np.column_stack([s0, t0])
        converged = np.full(n_points, False)
        iterations = np.zeros(n_points, dtype=int)
        active_mask = np.ones(n_points, dtype=bool)

        for iteration in range(1, max_iter + 1):
            if not np.any(active_mask):
                break
                
            active_indices = np.where(active_mask)[0]
            active_count = np.sum(active_mask)

            x_current_active, y_current_active = self.curvilinear_to_cartesian(
                p[active_indices, 0], p[active_indices, 1]
            )
            R_x = x_targets[active_indices] - x_current_active
            R_y = y_targets[active_indices] - y_current_active
            R_norm = np.sqrt(R_x**2 + R_y**2)

            converged_now = R_norm < tol
            if np.any(converged_now):
                converged_indices = active_indices[converged_now]
                converged[converged_indices] = True
                iterations[converged_indices] = iteration
                active_mask[converged_indices] = False
                if np.all(converged_now):
                    continue
                active_indices = active_indices[~converged_now]
                R_x = R_x[~converged_now]
                R_y = R_y[~converged_now]
            
            if not np.any(active_mask):
                break
                
            J_matrices = self.get_J(p[active_indices, 0], p[active_indices, 1])

            R_vectors = np.column_stack([R_x, R_y])

            delta_p = self.solve_linear_systems_vectorized_simple(J_matrices, R_vectors)

            p[active_indices] += delta_p
        
        iterations[~converged] = max_iter
        return p[:, 0], p[:, 1], iterations, converged