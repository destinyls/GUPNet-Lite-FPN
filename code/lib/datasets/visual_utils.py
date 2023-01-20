import cv2
import numpy as np

def draw_3d_box_on_image(image, gt_boxes, intrin, c=(0, 255, 0)):
    for gt_box in gt_boxes:
        xyz = gt_box["xyz"]
        hwl = gt_box["hwl"]
        yaw = gt_box["yaw"]
        dim = np.array(hwl).astype(float)
        location = np.array(xyz).astype(float)
        rotation_y = yaw
        box_3d = compute_box_3d_camera(dim, location, rotation_y)
        box_2d = project_to_image(box_3d, intrin[:3, :])
        image = draw_box_3d(image, box_2d, c=c)
    return image

def compute_box_3d_camera(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
    face_idx = [[0,1,5,4],
                [1,2,6,5],
                [2,3,7,6],
                [3,0,4,7]]
    for ind_f in [3, 2, 1, 0]:
        f = face_idx[ind_f]
        for j in [0, 1, 2, 3]:
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                    (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                    (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                    (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
    return image