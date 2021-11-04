from HeaderExtraction import headerextraction
from Locater import boxlocate,linelocate
path = r'D:\Alliance\OCR-invoince\crop\glenmc.png'
# header_df=headerextraction(path)


def lineextractor(path):

    header_df = headerextraction(path)
    lines = linelocate(path)
    boxes = boxlocate(path)
    final = []
    for line in lines :
        if 'PROGESIC 250MG' in str(line[0]):
            print('it')
        temp = {}
        linetest = []
        for box in boxes.values:
            #extract line
            if (str(box[0]) in str(line[0])) and (box[0] not in header_df['text'].to_list()) and (abs(box[2]-line[2]) < 20):
                # linetest.append(box[0])
                for header in header_df.values:
                    if abs(box[1]-header[2][0]) < 100 :
                       temp[str(header[0])] = str(box[0])
                       break
                       # str(header[0]) + str(box[0])

        if len(temp)>1:
           final.append(temp)
    return final

a = lineextractor(path)
print('done')