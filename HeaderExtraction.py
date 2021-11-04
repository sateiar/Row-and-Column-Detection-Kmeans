import json
import pandas as pd
from Locater import boxlocate,linelocate
#read dictionary
with open('dict.json') as json_file:
     table_dic = json.load(json_file)
# path = r'D:\Alliance\OCR-invoince\crop\2.png'
def headerextraction (path):
     boxes = boxlocate(path)
     # lines = linelocate(path)
     header_title = []
     header_text = []
     header_annotions = []
     for box in boxes.values:
          for phase in table_dic:
               for word in phase['words']:
                    if str(word) == str(box[0]).lower():
                         header_title.append(phase['title'])
                         header_text.append(box[0])
                         header_annotions.append([box[1],box[2],box[3],box[4]])
                         break
     df = pd.DataFrame({
          'title': header_title,
          'text' : header_text,
          'annotaion': header_annotions
     })
     return df

# path = r'D:\Alliance\OCR-invoince\crop\2.png'
# a = headerextraction(path)
#
#
# print("done")