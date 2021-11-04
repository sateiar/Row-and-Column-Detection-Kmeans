import pandas as pd
import glob
import os
import time
from ocr_process import ocr
img_dir = "output/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
# for path in files :
#     blocks = ocr(path)
#     for block in blocks :
#         data.append(block)
#     time.sleep(5)
#     print ('done')

from tesserocr import PyTessBaseAPI
import tesserocr
with PyTessBaseAPI(path='tessdata-master/tessdata-master/.', lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK) as api:
    for img in files:
        api.SetImageFile(img)
        b = api.GetUTF8Text()
        blocks = b.splitlines()
        for block in blocks:
            data.append(block)
        print('done')

df =pd.DataFrame({'sentence' : data})
df.to_csv('sentece.csv')