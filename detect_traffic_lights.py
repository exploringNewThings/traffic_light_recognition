import cv2
import os
import numpy as np
from typing import List
import imutils


class TemplateMatching:

    def __init__(self, template_image_list: List[str], scale_list: List[float]):
        self.template_image_list = []
        self.scale_list = scale_list
        for template in template_image_list:
            template_image = None
            try:
                if isinstance(template, str):
                    template_image = cv2.imread(template)
                elif isinstance(template, np.ndarray):
                    template_image = template
                else:
                    raise TypeError("Invalid template type")
            except TypeError:
                raise

            # Computing Edge Map
            template_image = cv2.Canny(template_image, 50, 200)

            self.template_image_list.append(template_image)

    def run(self, img):
        if len(img.shape) > 2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()

        found = None
        for scale in self.scale_list:
            try:
                resized = imutils.resize(gray_img, width=int(gray_img.shape[1] * scale))
            except Exception as e:
                # print(e)
                continue

            r = gray_img.shape[1] / float(resized.shape[1])

            edged = cv2.Canny(resized, 50, 200)

            for template in self.template_image_list:
                tH, tW = template.shape[:2]

                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

        if found:
            (maxVal, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

            return [startX, startY, endX, endY], maxVal
        else:
            return None, None


def get_top_hat(img, filter_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)
    top_hat = cv2.morphologyEx(img,
                               cv2.MORPH_TOPHAT,
                               kernel)
    return top_hat


def get_watershed(rgb_img, top_hat, threshold=100):
    ret, thresh = cv2.threshold(top_hat, threshold, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
    watershed = cv2.watershed(rgb_img, markers)

    return watershed


def compute_circularity(cnt, area):

    perimeter = cv2.arcLength(cnt, True)

    circularity = perimeter / (2 * np.sqrt(np.pi * area) + 1e-5)

    return circularity


def filter_blobs(stats, label_map, label_idx):
    component_mask = (label_map == label_idx).astype("uint8") * 255

    contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    cnt = contours[0]

    x_min, _ = cnt[cnt[:, :, 0].argmin()][0]
    x_max, _ = cnt[cnt[:, :, 0].argmax()][0]
    _, y_min = cnt[cnt[:, :, 1].argmin()][0]
    _, y_max = cnt[cnt[:, :, 1].argmax()][0]

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    if max(h, w) > 2 * min(h, w):
        return False, cnt

    circularity = compute_circularity(cnt, stats[label_idx, cv2.CC_STAT_AREA])
    delta = 0.45
    if circularity > (1 + delta):
        return False, cnt

    if circularity < 0.5:
        return False, cnt

    return True, cnt


def get_traffic_light_mask(rgb_image, template_matching_object, filter_size=(71, 71), threshold=0,
                           confidence_threshold=0.0):
    h_image, w_image, _ = rgb_image.shape
    viz_image = rgb_image.copy()

    luv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LUV)
    top_hat = get_top_hat(luv_image[:, :, 0], filter_size)

    ret, thresh = cv2.threshold(np.uint8(top_hat), threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    output = cv2.connectedComponentsWithStats(np.uint8(dist_transform), 8, cv2.CV_32S)
    (num_labels, labels, stats, centroids) = output

    final_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), np.uint8)
    confidence_list = []
    for i in range(1, num_labels):
        # extract the connected component statistics and centroid for
        # the current label
        status, cnt = filter_blobs(stats, labels, i)
        cnt = cnt[:, 0]
        if status is False:
            continue

        component_mask = (labels == i).astype("uint8") * 255

        left_x, left_y = cnt[cnt[:, 0].argmin()]
        right_x, right_y = cnt[cnt[:, 0].argmax()]
        top_x, top_y = cnt[cnt[:, 1].argmin()]
        bottom_x, bottom_y = cnt[cnt[:, 1].argmax()]

        height_blob, width_blob = bottom_y - top_y + 1, right_x - left_x + 1

        margin_w = 50
        height_factor = 8
        crop_top_left_x, crop_top_left_y = max(left_x - margin_w, 0), max(left_y - (height_factor//2) * height_blob, 0)
        crop_height, crop_width = height_factor * height_blob, width_blob + 2 * margin_w

        if (crop_top_left_x + crop_width) >= w_image:
            crop_width = w_image - 1 - crop_top_left_x

        if (crop_top_left_y + crop_height) >= h_image:
            crop_height = h_image - 1 - crop_top_left_y

        crop = rgb_image[crop_top_left_y: crop_top_left_y + crop_height,
                         crop_top_left_x: crop_top_left_x + crop_width]

        coords, confidence = template_matching_object.run(crop)
        matching_mask = np.zeros_like(component_mask)
        if coords and (confidence > confidence_threshold):
            coords[0] = coords[0] + crop_top_left_x
            coords[2] = coords[2] + crop_top_left_x
            coords[1] = coords[1] + crop_top_left_y
            coords[3] = coords[3] + crop_top_left_y

            matching_mask[coords[1]: coords[3] + 1, coords[0]: coords[2] + 1] = 1
            confidence_list.append(confidence)

            traffic_light, color_mean = recognize_traffic_light(rgb_img, cnt)

            viz_image = draw_bbox(viz_image, cnt, traffic_light, color_mean=None)
        component_mask = component_mask * matching_mask

        final_mask = final_mask + component_mask
    return final_mask, viz_image


def recognize_traffic_light(rgb_img, cnt, color_thresh=140):
    left_x, left_y = cnt[cnt[:, 0].argmin()]
    right_x, right_y = cnt[cnt[:, 0].argmax()]
    top_x, top_y = cnt[cnt[:, 1].argmin()]
    bottom_x, bottom_y = cnt[cnt[:, 1].argmax()]

    rgb_crop = rgb_img[top_y: bottom_y + 1, left_x: right_x + 1]

    r_mean = rgb_crop[:, :, 2].mean()
    g_mean = rgb_crop[:, :, 1].mean()
    b_mean = rgb_crop[:, :, 0].mean()

    if r_mean > color_thresh and g_mean > color_thresh:
        color_pred = "yellow"
    elif r_mean > color_thresh:
        color_pred = "red"
    elif g_mean > color_thresh:
        color_pred = "green"
    else:
        color_pred = "off"

    return color_pred, (r_mean, g_mean, b_mean)


def draw_bbox(rgb_img, cnt, traffic_light, color_mean=None):
    x_min, _ = cnt[cnt[:, 0].argmin()]
    x_max, _ = cnt[cnt[:, 0].argmax()]
    _, y_min = cnt[cnt[:, 1].argmin()]
    _, y_max = cnt[cnt[:, 1].argmax()]

    area = cv2.contourArea(np.expand_dims(cnt, 1))
    perimeter = cv2.arcLength(np.expand_dims(cnt, 1), True)

    circularity = round(perimeter / (2 * np.sqrt(np.pi * area)), 2)

    rgb_img = cv2.rectangle(rgb_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    # Writing text
    org = (x_min, y_max + 30)
    if color_mean is not None:
        color_mean = list(map(int, color_mean))
    else:
        color_mean = ''
    # rgb_img = write_text(rgb_img, f"{traffic_light}, {str(color_mean)}, {str(circularity)}", org,
    #                      font_scale=0.5)
    rgb_img = write_text(rgb_img, f"{traffic_light}", org,
                         font_scale=0.5)

    return rgb_img


def write_text(img, text, org,
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.35,
               color=(0, 0, 255), thickness=1):
    img = cv2.putText(img, text, org, font, font_scale,
                      color, thickness, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./input_dir")
    parser.add_argument("--output_dir", type=str, default="./output_dir")
    parser.add_argument("--template_dir", type=str, default="./template_dir")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = args.input_dir
    output_dir = args.output_dir
    template_dir = args.template_dir

    template_list = []
    for root, _, files in os.walk(template_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                template_list.append(os.path.join(root, file))

    scale_list = list(np.linspace(0.2, 2.0, 40)[::-1])
    template_matching_object = TemplateMatching(template_list, scale_list)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                print(f"Processing: {file}")
                img_file = os.path.join(root, file)
                rgb_img = cv2.imread(img_file)
                traffic_light_mask, viz_image = get_traffic_light_mask(rgb_img, template_matching_object)

                traffic_light_mask = traffic_light_mask//255
                masked_rgb_img = rgb_img * np.expand_dims(traffic_light_mask, axis=2)
                # cv2.imwrite(os.path.join(output_dir, f"traffic_light_mask_{file}"), masked_rgb_img)
                cv2.imwrite(os.path.join(output_dir, f"traffic_light_viz_{file}"), viz_image)

