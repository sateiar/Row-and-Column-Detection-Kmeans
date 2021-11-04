import tesserocr
from tesserocr import PyTessBaseAPI
from PIL import Image
import cv2
import os
import pandas as pd
from prepocessing import process_image_for_ocr

def pre_process_image(path, morph_size=(17,3)):
    save_in_file = os.path.join("crop", "preprocesed.png")
    img = cv2.imread(path)
    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy



    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre



def find_text_boxes(pre, min_text_height_limit=3, max_text_height_limit=100):

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




def find_table_in_boxes(boxes, cell_threshold=25, min_columns=9):
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





def linelocate (path):
    tableline = []
    img = process_image_for_ocr(path)
    cv2.imwrite('output/img.png', img)
    with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK) as api:
        api.SetImageFile('output/img.png')
        boxes = api.GetComponentImages( tesserocr.RIL.TEXTLINE,True )
        print ('Found {} textline image components.'.format(len(boxes)))
        for i, (im, box, _, _) in enumerate(boxes):
            # im is a PIL image object
            # box is a dict with x, y, w and h keys
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            tableline.append ([api.GetUTF8Text(),box['x'], box['y'], box['w'], box['h'],api.MeanTextConf()])
            # print(api.GetTextlines())

            # conf = api.MeanTextConf()
            # print( box ,api.GetUTF8Text())
        return tableline
# a = linelocate(r'D:\Alliance\OCR-invoince\crop\245863_3.png')
# print('done')

def boxlocate(path):
    pre_processed = pre_process_image(path)
    cells = find_text_boxes(pre_processed)
    # cells = find_table_in_boxes(text_boxes)
    out_text =[]
    list1 = []
    list2  =[]
    list3  =[]
    list4 =[]
    list5 =[]
    img = process_image_for_ocr(path)
    cv2.imwrite('output/img.png', img)
    with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK) as api:
        api.SetImageFile('output/img.png')
        for cell in cells :
                api.SetRectangle(cell[0],cell[1],cell[2],cell[3])
                out_text.append([api.GetUTF8Text(),cell[0],cell[1],cell[2],cell[3]])
                box = api.GetUTF8Text()
                if len(box) !=0 :
                    list1.append(str(box).splitlines()[0])
                    list2.append(cell[0])
                    list3.append(cell[1])
                    list4.append(cell[2])
                    list5.append(cell[3])
    df = pd.DataFrame({
        'text' :list1,
        'x' : list2,
        'y' : list3,
        'w' : list4,
        'h' : list5 })
    return df
#
# b=boxlocate(r'D:\Alliance\OCR-invoince\crop\245863_3.png')
# print('done')