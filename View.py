from Util import *
import DataProcessing as datap

path_base = 'Resources/images'
img_header = cv2.imread(path_base + '/header.png')
img_header_BEV = cv2.imread(path_base + '/header_BEV.png')
img_footer_BEV = cv2.imread(path_base + '/footer_BEV.png')


def get_image_camera(frame, data, g_close, g_distant):
    img_bbox = draw_bbox_video(frame, data, g_close, g_distant)
    _, dim_final_bbox, _ = get_dim_video()
    frame_resized = cv2.resize(img_bbox, (dim_final_bbox[1], dim_final_bbox[0]))
    return frame_resized


def draw_bbox_video(image, data, g_cerca, g_lejos):
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    ind_p, p_bbox_center, pair_point_bbox = datap.process_df2bbox(data)

    for pair_points in g_cerca:
        points_line = []
        for points in pair_points:
            id_people = points[2]

            ind_list = ind_p.index(id_people)
            cv2.rectangle(img=image,
                          pt1=point_float2int(pair_point_bbox[ind_list][0]),
                          pt2=point_float2int(pair_point_bbox[ind_list][1]),
                          color=color_red,
                          thickness=2)
            points_line.append(p_bbox_center[ind_list])
        cv2.line(image, points_line[0], points_line[1], color_red, thickness=2)

    for Po in g_lejos:
        id_people = Po[2]
        ind_list = ind_p.index(id_people)
        cv2.rectangle(img=image,
                      pt1=point_float2int(pair_point_bbox[ind_list][0]),
                      pt2=point_float2int(pair_point_bbox[ind_list][1]),
                      color=color_green,
                      thickness=2)
    return image


def get_image_bev(g_close, g_distant, min_separation_dist):
    _, _, dim_final_ebv = get_dim_video()
    rows_resized = dim_final_ebv[0]
    cols_resized = int(dim_final_ebv[1])

    image = draw_point_ebv(g_close, g_distant, min_separation_dist)

    img_bev_rec = image[150:, 860:1560]
    l_border = 20

    img_bev_res_temp = cv2.resize(img_bev_rec, (cols_resized - (2 * l_border), rows_resized - 120))
    img_bev_res_temp = join_images(img_header_BEV, img_bev_res_temp, axis=0)
    img_bev_res_temp = join_images(img_bev_res_temp, img_footer_BEV, axis=0)

    img_with_border = put_border(img_bev_res_temp, l_border)
    img_resize_bev = cv2.resize(img_with_border, (cols_resized, rows_resized))

    return img_resize_bev


def draw_point_ebv(g_close, g_distant, min_separation_dist):
    dim_ini_video, _, _ = get_dim_video()
    background = np.zeros((dim_ini_video[0] + 150, dim_ini_video[1], 3), dtype=np.uint8)

    color_red = (0, 0, 255)
    color_green = (0, 255, 0)

    for pair_distant_points in g_distant:
        distant_points = (pair_distant_points[0], pair_distant_points[1])
        cv2.line(background, distant_points, distant_points, color_green, thickness=22)
        # cv2.circle(background, distant_points, int(min_separation_dist / 2), color_green, thickness=2)

    n_close_points = 0
    for pair_close_points in g_close:
        n_close_points += 1
        close_points = []
        for i in range(0, 2):
            id_pers = str(pair_close_points[i][2])
            close_points.append((pair_close_points[i][0], pair_close_points[i][1]))
            cv2.line(background, close_points[i], close_points[i], color_red, thickness=22)
            # cv2.circle(background, close_points[i], int(min_separation_dist / 2), color_red, thickness=2)
        cv2.line(background, close_points[0], close_points[1], color_red, thickness=8)
    return background
