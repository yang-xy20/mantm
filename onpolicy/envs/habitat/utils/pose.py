import numpy as np
import math

def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.:
        o -= 360.

    return x, y, o

def get_new_pose_from_dis(pose, dis, angle):
    dx = np.cos(np.deg2rad(angle))* dis
    dy = np.sin(np.deg2rad(angle))* dis
    do = angle
    x, y, o = pose
    
    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += do
    if o > 180.:
        o -= 360.
    return x, y, o


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords

def get_patch_coordinates(angles, goal, pano_radius):
    dist = pano_radius
    coords = []
    # 120 FOV -> crop to 110 get the following angles
    # angles = [-55, -45, -30, -15, 0, 15, 30, 45, 55]
    for ang in angles:
        ang = math.radians(ang)
        y = int(dist * np.cos(ang) + goal[0])
        x = int(dist * np.sin(ang) + goal[1])
        coords.append([x, y])
    return coords
