import tesserocr
from tesserocr import PyTessBaseAPI , RIL
ocrResult = []
with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK , oem=tesserocr.OEM.LSTM_ONLY ) as api:
    api.SetImageFile(r'D:\Alliance\OCR-invoince\crop\2.png')
    boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE,True )
    print ('Found {} textline image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult.append (api.GetUTF8Text())
        conf = api.MeanTextConf()
        print( box ,api.GetUTF8Text())