import os
import cv2
import numpy as np
import imutils
from PIL import Image
import tesserocr
from tesserocr import PyTessBaseAPI

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns


def pre_process_image(img, save_in_file, morph_size=(8,8 )):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #bluring %
    # pre = cv2.GaussianBlur(pre, (3, 3), 0)
    # Otsu threshold
    # pre = cv2.threshold(pre, 150, 255, cv2.THRESH_BINARY )[1]
    pre = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # pre = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)[1]
    # pre = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # dilate the text to make it solid spot
    cpy = pre.copy()
    #new preporcess
    # bitwise = cv2.bitwise_not(cpy)
    # erosion = cv2.erode(bitwise, np.ones((1, 1), np.uint8), iterations=5)
    # cpy = cv2.dilate(erosion, np.ones((3, 3), np.uint8), iterations=5)
    #
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre

#
# ret, thresh1 = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
# bitwise = cv2.bitwise_not(thresh1)




def find_text_boxes(pre, min_text_height_limit=4, max_text_height_limit=100):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    return boxes


def find_table_in_boxes(boxes, cell_threshold=25, min_columns=8):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines


if __name__ == "__main__":
    in_file = os.path.join("crop", "2.png")
    pre_file = os.path.join("crop", "3.png")
    out_file = os.path.join("crop", "out.png")

    img = cv2.imread(os.path.join(in_file))

    pre_processed = pre_process_image(img, pre_file)
    text_boxes = find_text_boxes(pre_processed)
    cells = find_table_in_boxes(text_boxes)
    img = Image.open(in_file)
    out_text =[]
    import pandas as pd
    list1 = []
    list2  =[]
    list3  =[]
    list4 =[]
    list5 =[]
    with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK,
                       oem=tesserocr.OEM.LSTM_ONLY) as api:
        api.SetImage(img)
        for row in cells :
            # temp = []
            for cell in row:
                api.SetRectangle(cell[0],cell[1],cell[2],cell[3])
                out_text.append([api.GetUTF8Text(),cell[0],cell[1],cell[2],cell[3]])
                list1.append(api.GetUTF8Text())
                list2.append(cell[0])
                list3.append(cell[1])
                list4.append(cell[2])
                list5.append(cell[3])
    df = pd.DataFrame({
        'text' :list1,
        'x' : list2,
        'y' : list3,
        'w' : list4,
        'h' : list5

    })
    df.to_csv('test.csv')

            # out_text.append(temp)


out_text[:][2]

    # hor_lines, ver_lines = build_lines(cells)
    #
    # # Visualize the result
    # vis = img.copy()
    #
    # # for box in text_boxes:
    # #     (x, y, w, h) = box
    # #     cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)
    #
    # for line in hor_lines:
    #     [x1, y1, x2, y2] = line
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #
    # for line in ver_lines:
    #     [x1, y1, x2, y2] = line
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(out_file, vis)

print ('done')