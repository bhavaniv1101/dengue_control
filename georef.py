from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraParams:
    """Parameters for transforming from image frame to camera frame."""

    f_mm: float = 12.5
    image_width_px: int = 2448
    image_height_px: int = 2048
    sensor_width_mm: float = 8.6
    sensor_height_mm: float = 8.6 * 2048 / 2448

    def principal_point_px(self) -> (float, float):
        """
        Returns location of principal point (c_x, c_y) in pixel units.
        """
        return self.image_width_px / 2, self.image_height_px / 2

    def fx_fy(self):
        """
        Returns fx and fy
        """
        fx = self.f_mm * self.image_width_px / self.sensor_width_mm
        fy = self.f_mm * self.image_height_px / self.sensor_height_mm

        return fx, fy

    def uv_tilde(self, u: int, v: int) -> (float, float):
        """
        Returns (u_tilde, v_tilde) for given (u, v).
        """
        c_x, c_y = self.principal_point_px()
        fx, fy = self.fx_fy()
        return (u - c_x) / fx, (v - c_y) / fy

    def uv_tilde_grid(self, num_u, num_v):
        """
        Returns a grid of u_tilde, v_tilde points of size num_u by num_v.
        """

        u_grid = np.linspace(0, self.image_width_px, num_u)
        v_grid = np.linspace(0, self.image_height_px, num_v)
        u_mesh, v_mesh = np.meshgrid(u_grid, v_grid)
        return self.uv_tilde(u_mesh, v_mesh)


@dataclass(frozen=True)
class EulerAngles:
    """Euler angles (alpha, beta, gamma) in radians, representing orientation of UAS in NED frame."""

    alpha: float  # Yaw
    beta: float  # Pitch
    gamma: float  # Roll


def rotation_matrix(euler_angles: EulerAngles):
    """
    Returns the rotation matrix for the angles alpha, beta, gamma.

    :param euler_angles: EulerAngles dataclass with alpha, beta, gamma
    """

    alpha = euler_angles.alpha
    beta = euler_angles.beta
    gamma = euler_angles.gamma

    sin_a = np.sin(alpha)
    sin_b = np.sin(beta)
    sin_g = np.sin(gamma)

    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)
    cos_g = np.cos(gamma)

    rot_matrix = np.array(
        [
            [cos_a * cos_b, sin_a * cos_b, -sin_b],
            [
                cos_a * sin_b * sin_g - sin_a * cos_g,
                sin_a * sin_b * sin_g + cos_a * cos_g,
                cos_b * sin_g,
            ],
            [
                cos_a * sin_b * cos_g + sin_a * sin_g,
                sin_a * sin_b * cos_g - cos_a * sin_g,
                cos_b * cos_g,
            ],
        ]
    )

    return rot_matrix


def h_i(i, g, u_tilde, v_tilde):
    """
    Returns the 'h' value for row i based on the camera parameters

    :param i: row number for h
    :param g: rotation matrix
    :param u_tilde: (u - cx) / fx --> camera parameters
    :param v_tilde: (v - cy) / fy --> camera parameters
    """

    h = g[i, 0] * u_tilde + g[i, 1] * v_tilde + g[i, 2]

    return h


def x_ned(z_ned, g, u_tilde, v_tilde):
    """
    Returns the value of the x coordinate of the target in the NED frame.
    """

    return z_ned * h_i(0, g, u_tilde, v_tilde) / h_i(2, g, u_tilde, v_tilde)


def y_ned(z_ned, g, u_tilde, v_tilde):
    """
    Returns the value of the y coordinate of the target in the NED frame.
    """

    return z_ned * h_i(1, g, u_tilde, v_tilde) / h_i(2, g, u_tilde, v_tilde)


def derive_g_ij_wrt_angle(i, j, angle: str, euler_angles: EulerAngles):
    """
    Returns the derivative of g[i,j] with respect to the given angle (alpha, beta, or gamma)
    """

    alpha = euler_angles.alpha
    beta = euler_angles.beta
    gamma = euler_angles.gamma

    g = rotation_matrix(euler_angles)

    if angle == "alpha":
        if j == 0:
            return -g[i, 1]
        if j == 1:
            return g[i, 0]
        if j == 2:
            return 0

    if angle == "beta":
        if i == 0:
            if j == 0:
                return -np.cos(alpha) * np.sin(beta)
            if j == 1:
                return -np.sin(alpha) * np.sin(beta)
            if j == 2:
                return -np.cos(beta)

        if i == 1:
            return g[0, j] * np.sin(gamma)

        if i == 2:
            return g[0, j] * np.cos(gamma)

    if angle == "gamma":
        if i == 0:
            return 0
        if i == 1:
            return g[2, j]
        if i == 2:
            return -g[1, j]

    raise ValueError("Invalid inputs")


def derive_h_i_wrt_angle(i, angle: str, euler_angles: EulerAngles, u_tilde, v_tilde):
    """
    Returns the derivative of h_i for given i
    """

    return (
        derive_g_ij_wrt_angle(i, 0, angle, euler_angles) * u_tilde
        + derive_g_ij_wrt_angle(i, 1, angle, euler_angles) * v_tilde
        + derive_g_ij_wrt_angle(i, 2, angle, euler_angles)
    )


def derive_x_ned(angle: str, z_ned, euler_angles: EulerAngles, u_tilde, v_tilde):
    """
    Returns the derivative of x_ned with respect to the given angle

    :param angle: alpha, beta, or gamma
    :param z_ned: z coordinate in NED rederence frame (altitude of point)
    :param euler_angles: EulerAngles dataclass with alpha, beta, gamma
    """

    g = rotation_matrix(euler_angles)

    h0 = h_i(0, g, u_tilde, v_tilde)
    h2 = h_i(2, g, u_tilde, v_tilde)

    d_h0 = derive_h_i_wrt_angle(0, angle, euler_angles, u_tilde, v_tilde)
    d_h2 = derive_h_i_wrt_angle(2, angle, euler_angles, u_tilde, v_tilde)

    return ((h2 * d_h0 - h0 * d_h2) / (h2**2)) * z_ned


def derive_y_ned(angle: str, z_ned, euler_angles: EulerAngles, u_tilde, v_tilde):
    """
    Returns the derivative of y_ned with respect to the given angle

    :param angle: alpha, beta, or gamma
    :param z_ned: z coordinate in NED rederence frame (altitude of point)
    :param euler_angles: EulerAngles dataclass with alpha, beta, gamma
    """

    g = rotation_matrix(euler_angles)

    h1 = h_i(1, g, u_tilde, v_tilde)
    h2 = h_i(2, g, u_tilde, v_tilde)

    d_h1 = derive_h_i_wrt_angle(1, angle, euler_angles, u_tilde, v_tilde)
    d_h2 = derive_h_i_wrt_angle(2, angle, euler_angles, u_tilde, v_tilde)

    return ((h2 * d_h1 - h1 * d_h2) / (h2**2)) * z_ned


def test_rotation_matrix():
    g = rotation_matrix(EulerAngles(0, 0, 0))
    assert np.allclose(g, np.eye(3))

    g = rotation_matrix(EulerAngles(np.pi / 2, 0, 0))
    assert np.allclose(g, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    g = rotation_matrix(EulerAngles(0, np.pi / 2, 0))
    assert np.allclose(g, [[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    g = rotation_matrix(EulerAngles(0, 0, np.pi / 2))
    assert np.allclose(g, [[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    g = rotation_matrix(EulerAngles(np.pi / 2, np.pi / 2, 0))
    assert np.allclose(g, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def test_derive_g_ij_wrt_angle():
    derivative = derive_g_ij_wrt_angle(0, 0, "alpha", EulerAngles(0, 0, 0))
    assert np.allclose(derivative, 0)

    derivative = derive_g_ij_wrt_angle(0, 0, "alpha", EulerAngles(np.pi / 2, 0, 0))
    assert np.allclose(derivative, -1)

    derivative = derive_g_ij_wrt_angle(0, 0, "alpha", EulerAngles(0, np.pi / 2, 0))
    assert np.allclose(derivative, 0)

    derivative = derive_g_ij_wrt_angle(1, 1, "beta", EulerAngles(0, 0, 0))
    assert np.allclose(derivative, 0)

    derivative = derive_g_ij_wrt_angle(1, 0, "beta", EulerAngles(np.pi / 2, 0, 0))
    assert np.allclose(derivative, 0)

    derivative = derive_g_ij_wrt_angle(2, 2, "beta", EulerAngles(0, np.pi / 2, 0))
    assert np.allclose(derivative, -1)

    derivative = derive_g_ij_wrt_angle(0, 1, "gamma", EulerAngles(0, np.pi / 2, 0))
    assert np.allclose(derivative, 0)

    derivative = derive_g_ij_wrt_angle(1, 2, "gamma", EulerAngles(np.pi / 2, 0, 0))
    assert np.allclose(derivative, 1)

    derivative = derive_g_ij_wrt_angle(2, 0, "gamma", EulerAngles(0, 0, np.pi / 2))
    assert np.allclose(derivative, 0)


def test_h_i():
    h = h_i(0, rotation_matrix(EulerAngles(0, 0, 0)), 1, 1)
    assert np.allclose(h, 1)

    h = h_i(1, rotation_matrix(EulerAngles(0, 0, 0)), 1, 1)
    assert np.allclose(h, 1)

    h = h_i(2, rotation_matrix(EulerAngles(0, 0, 0)), 1, 1)
    assert np.allclose(h, 1)


def test_derive_h_i_wrt_angle():
    derivative = derive_h_i_wrt_angle(0, "alpha", EulerAngles(0, 0, 0), 1, 1)
    assert np.allclose(derivative, 1)

    derivative = derive_h_i_wrt_angle(1, "alpha", EulerAngles(0, np.pi / 2, 0), 1, 1)
    assert np.allclose(derivative, -1)

    derivative = derive_h_i_wrt_angle(
        2, "alpha", EulerAngles(np.pi / 2, np.pi / 2, 0), 1, 1
    )
    assert np.allclose(derivative, -1)

    derivative = derive_h_i_wrt_angle(0, "beta", EulerAngles(0, 0, np.pi / 2), 1, 1)
    assert np.allclose(derivative, -1)

    derivative = derive_h_i_wrt_angle(1, "beta", EulerAngles(np.pi / 2, 0, 0), 1, 1)
    assert np.allclose(
        derivative,
        h_i(1, rotation_matrix(EulerAngles(np.pi / 2, 0, 0)), 1, 1) * np.sin(0),
    )

    derivative = derive_h_i_wrt_angle(
        2, "beta", EulerAngles(np.pi / 2, np.pi / 2, np.pi / 2), 1, 1
    )
    assert np.allclose(
        derivative,
        h_i(1, rotation_matrix(EulerAngles(np.pi / 2, np.pi / 2, np.pi / 2)), 1, 1)
        * np.cos(np.pi / 2),
    )

    derivative = derive_h_i_wrt_angle(0, "gamma", EulerAngles(0, 0, 0), 1, 1)
    assert np.allclose(derivative, 0)

    derivative = derive_h_i_wrt_angle(1, "gamma", EulerAngles(0, np.pi / 2, 0), 1, 1)
    assert np.allclose(
        derivative, h_i(2, rotation_matrix(EulerAngles(0, np.pi / 2, 0)), 1, 1)
    )

    derivative = derive_h_i_wrt_angle(
        2, "gamma", EulerAngles(np.pi / 2, np.pi / 2, 0), 1, 1
    )
    assert np.allclose(
        derivative, -h_i(1, rotation_matrix(EulerAngles(np.pi / 2, np.pi / 2, 0)), 1, 1)
    )


def test_derive_x_ned():
    derivative = derive_x_ned("alpha", 1, EulerAngles(0, 0, 0), 1, 1)
    assert np.allclose(derivative, 1)

    derivative = derive_x_ned("beta", 1, EulerAngles(0, np.pi / 2, 0), 1, 1)
    assert np.allclose(derivative, -2)

    derivative = derive_x_ned("gamma", 1, EulerAngles(np.pi / 2, 0, np.pi / 2), 1, 1)
    assert np.allclose(derivative, 1)


def test_derive_y_ned():
    derivative = derive_y_ned("alpha", 1, EulerAngles(0, 0, 0), 1, 1)
    assert np.allclose(derivative, -1)

    derivative = derive_y_ned("beta", 1, EulerAngles(0, np.pi / 2, 0), 1, 1)
    assert np.allclose(derivative, 1)

    derivative = derive_y_ned("gamma", 1, EulerAngles(np.pi / 2, 0, np.pi / 2), 1, 1)
    assert np.allclose(derivative, 2)


def main():
    """Run main function."""

    # Camera parameters
    camera_par = CameraParams()

    # Camera parameter matrix (K)
    c_x, c_y = camera_par.principal_point_px()
    f_x, f_y = camera_par.fx_fy()
    cam_par_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    # Target position in camera (C) frame, normalized by z_C
    pos_in_c = np.linalg.solve(cam_par_matrix, np.array([1095, 1099, 1]))

    # Assume gimbal (G) and camera (C) frames are identical.
    # Assume NED and UAS frames are identical.
    trans_uas_from_g = np.array([0.3, 0.0, 0.2])
    rot_uas_from_g = rotation_matrix(EulerAngles(-np.pi / 2, -np.pi / 3, 0.0))
    rot_enu_from_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    trans_enu_from_ned = np.array([31.72212, -6.55099, 42.44889])

    tmp_1 = rot_enu_from_ned.dot(rot_uas_from_g.dot(pos_in_c))
    tmp_2 = rot_enu_from_ned.dot(trans_uas_from_g) + trans_enu_from_ned
    z_c = -tmp_2[2] / tmp_1[2]

    pos_in_enu = z_c * tmp_1 + tmp_2
    print(pos_in_enu)

    # # Target location w.r.t. principal point (in pixel units)
    # u_tilde, v_tilde = camera_par.uv_tilde(1095, 1099)
    #
    # # Orientation of UAS in NED frame (from Table 1 in paper)
    # # Assume origins of UAS and NED frames coincide
    # euler_angles = EulerAngles(alpha=(-np.pi / 2), beta=(np.pi / 6), gamma=0)
    # # Target height (m) in NED frame
    # z_ned = -42.44889 + 0.2
    #
    # rot_mtx = rotation_matrix(euler_angles)
    # x_enu = y_ned(z_ned, rot_mtx, u_tilde, v_tilde) - 31.72212
    # y_enu = y_ned(z_ned, rot_mtx, u_tilde, v_tilde) + 6.55099
    # print(x_enu, y_enu)


if __name__ == "__main__":
    main()
