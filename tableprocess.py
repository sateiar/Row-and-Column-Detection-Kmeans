import tesserocr
from tesserocr import PyTessBaseAPI


with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng') as api:
    api.SetImageFile(r'D:\Alliance\OCR-invoince\crop\2.png')
    boxes = api.GetComponentImages( tesserocr.RIL.TEXTLINE, True)
    print ('Found {} textline image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
        try :
            print  (u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                   "confidence: {1}, text: {2}").format(i, conf, ocrResult,box)
        except :
            print("cannot ")