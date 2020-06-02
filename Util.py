import cv2
import numpy as np
from math import sqrt

"""path_base = 'Resources/images'
img_header = cv2.imread(path_base+'/header.png')
img_header_BEV = cv2.imread(path_base+'/header_BEV.png')
img_footer_BEV = cv2.imread(path_base+'/footer_BEV.png')"""


def get_dim_video():
    h_video_resized = 540
    dim_ini_video = (1080, 1920)

    rK = dim_ini_video[1] / dim_ini_video[0]
    dim_final_bbox = (h_video_resized, int(h_video_resized * rK))
    dim_final_bev = (h_video_resized, int(h_video_resized / rK))
    return dim_ini_video, dim_final_bbox, dim_final_bev


def get_matrix_perspective():
    pts1 = np.float32([[1480, 0], [123, 847], [762, 1080], [1778, 22]])
    pts2 = np.float32([[1100, 80], [1100, 1000], [1300, 1000], [1300, 80]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return matrix


def get_perspective_points(point0, point1):
    matrix_pt = get_matrix_perspective() @ [point0, point1, 1]
    points_perspective = (int(matrix_pt[0] / matrix_pt[2]), int(matrix_pt[1] / matrix_pt[2]))
    return points_perspective


def put_border(src, l_border):
    value = [170, 170, 170]
    dst = cv2.copyMakeBorder(src, l_border, l_border, l_border, l_border, cv2.BORDER_CONSTANT, None, value)
    return dst


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def join_images(image1, image2, axis):
    """
    :param image1:
    :param image2:
    :param _axis: 1-> concatenate h -
                : 0-> concatenate v |
    :return: imageFinal: la imagen final, luego de las uniones

    """

    col_img1 = image1.shape[1]
    img1_row = image1.shape[0]
    img2_row = image2.shape[0]
    col_img2 = image2.shape[1]

    if axis == 0:
        if col_img1 > col_img2:
            image2 = resize_image(image2, img2_row, col_img2, col_img1, True)
        elif col_img2 > col_img1:
            image1 = resize_image(image1, img1_row, col_img1, col_img2, True)
    elif axis == 1: ##Falta completar
        if img1_row > img2_row:
            ratio = img2_row / col_img2
            new_dim = (img1_row, int(col_img2 * ratio))
            image2 = cv2.resize(image2, new_dim)
        elif col_img2 > col_img1:
            ratio = img1_row / col_img1
            new_dim = (img2_row, int(col_img1 * ratio))
            image1 = cv2.resize(image1, new_dim)

    img_final = np.concatenate((image1, image2), axis=axis)

    return img_final


def get_all_point_perspective(array_points):
    point_persp = []
    for old_points in array_points:
        point_persp.append(get_perspective_points(old_points[0], old_points[1]))

    return np.array(point_persp)


def get_group_distance(list_index, list_points, min_separation_dist):
    array_position = get_all_point_perspective(np.array(list_points))
    g_close, g_distant = separate_group(array_position, list_index, min_separation_dist)

    return g_close, g_distant


def separate_group(list_position, _df_point_inter, min_separation_dist):
    distant_points = []
    close_points = []

    num_people = list_position.shape[0]

    index_closes = []
    for i in range(0, num_people):
        num_close_points = 0
        for j in range(i + 1, num_people):
            dist = distance(list_position[i], list_position[j])
            if dist < min_separation_dist:
                pto_ori = (list_position[i][0], list_position[i][1], _df_point_inter[i])
                pto_dest = (list_position[j][0], list_position[j][1], _df_point_inter[j])
                close_points.append([pto_ori, pto_dest])

                if not (j in index_closes):
                    index_closes.append(j)
                num_close_points += 1
        if (num_close_points > 0) and ((i in index_closes) == False):
            index_closes.append(i)
        elif not (i in index_closes):
            pto_ori = (list_position[i][0], list_position[i][1], _df_point_inter[i])
            distant_points.append(pto_ori)

    return close_points, distant_points


def point_float2int(p_float):
    px, py = p_float
    p_int = (int(px), int(py))
    return p_int


def resize_image(img_src, img_min_row, img_min_col, len_img_final, col):
    new_dim = ()
    if col:
        ratio_col = img_min_row / img_min_col
        new_len_col = len_img_final
        new_len_row = int(new_len_col * ratio_col)
        new_dim = (new_len_col, new_len_row)

    img_resized = cv2.resize(img_src, new_dim)
    return img_resized
