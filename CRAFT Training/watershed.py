import cv2
import numpy as np
import math
from shapely.geometry import Polygon

def watershed(original_image, image, viz=False):
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    _, thresh = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("Thresh", thresh)
        cv2.waitKey()
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    if viz:
        cv2.imshow("sure_bg", opening)
        cv2.waitKey()
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg, connectivity=4)
    markers = labels.copy() + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()
    markers = cv2.watershed(original_image, markers)
    original_image[markers == -1] = [0, 0, 255]

    if viz:
        color_markers = np.uint8(markers + 1)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers1", color_markers)
        cv2.waitKey()

    if viz:
        cv2.imshow("image", image)
        cv2.waitKey()
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = Polygon(box)
        area = poly.area
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)


if __name__ == '__main__':
    input = cv2.imread('D:/Computer Vision Projects/CRAFT/inputs/test1.png', cv2.IMREAD_COLOR)
    image = cv2.imread('D:/Computer Vision Projects/CRAFT/outputs/test1_text_score_heatmap.png', cv2.IMREAD_COLOR)
    boxes = watershed(input, image, True)
    print(boxes)