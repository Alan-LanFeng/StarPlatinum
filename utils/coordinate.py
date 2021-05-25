import numpy as np
from utils import util
from numba import jit


@jit(nopython=True)
def tf2ego_location(loca: np.ndarray, ego_loca: np.ndarray, ego_direc: float) -> np.ndarray:
    rotate_matrix = np.array([
        [np.cos(-ego_direc), -np.sin(-ego_direc)],
        [np.sin(-ego_direc), np.cos(-ego_direc)]], dtype=loca.dtype)
    loca_tf = loca - ego_loca
    loca_tf = np.dot(rotate_matrix, loca_tf.T).T
    return loca_tf

@jit(nopython=True)
def tf2glo_location(loca: np.ndarray, glo_loca: np.ndarray, glo_direc: float) -> np.ndarray:
    rotate_matrix = np.array([
        [np.cos(glo_direc), -np.sin(glo_direc)],
        [np.sin(glo_direc), np.cos(glo_direc)]], dtype=loca.dtype)
    loca_tf = np.ascontiguousarray(rotate_matrix) @ np.ascontiguousarray(loca.T)
    loca_tf = loca_tf.T + glo_loca
    return loca_tf


@jit(nopython=True)
def tf2ego_locationX(locas: np.ndarray, ego_loca: np.ndarray, ego_direc: float) -> list:
    return [tf2ego_location(loca, ego_loca, ego_direc) for loca in locas]

@jit(nopython=True)
def tf2ego_direction(direc: np.ndarray, ego_direc: float, rm='-pipi') -> np.ndarray:
    return util.wrap_angle_miuspi_pi(direc - ego_direc) if rm == '-pipi' else\
           util.wrap_angle_zero_2pi(direc - ego_direc)

@jit(nopython=True)
def tf2glo_direction(direc: np.ndarray, glo_direc: float, rm='-pipi') -> np.ndarray:
    return util.wrap_angle_miuspi_pi(direc + glo_direc) if rm == '-pipi' else\
           util.wrap_angle_zero_2pi(direc + glo_direc)

@jit(nopython=True)
def tf2ego_directionX(direcs: np.ndarray, ego_direc: float, rm='-pipi') -> list:
    return [tf2ego_direction(direc, ego_direc, rm) for direc in direcs]

@jit(nopython=True)
def cart2polar(euler: np.ndarray, rm='-pipi'):
    radius = np.sqrt(euler[:, 0] ** 2 + euler[:, 1] ** 2)
    angles = np.arctan2(euler[:, 1], euler[:, 0])
    angles = util.wrap_angle_miuspi_pi(angles) if rm == '-pipi' else\
             util.wrap_angle_zero_2pi(angles)
    return np.stack((radius, angles), axis=-1)

@jit(nopython=True)
def cart2polarX(eulers: np.ndarray, rm='-pipi'):
    return [cart2polar(euler, rm) for euler in eulers]

@jit(nopython=True)
def cart2image(pt2d: np.ndarray, image_shape: np.ndarray, res: float) -> np.ndarray:
    point = pt2d / res + image_shape[:2] / 2
    return np.rint(point)