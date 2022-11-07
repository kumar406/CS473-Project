#importing modules
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'])

input = [{'img': 'Tensorflow/workspace/images/test/110.png',
  'labels': [{'label': {'id': 3, 'name': 'rel'},
    'box_norm': [0.43824703, 0.41720623, 0.5775799 , 0.5425698 ],
    'box': [ 773, 1593, 1018, 2072],
    'score': 0.99421114},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.        , 0.3701775 , 0.36916012, 0.6272964 ],
    'box': [   0, 1414,  651, 2396],
    'score': 0.9936568},
   {'label': {'id': 3, 'name': 'rel'},
    'box_norm': [0.03292408, 0.2181472 , 0.16734695, 0.3457169 ],
    'box': [  58,  833,  295, 1320],
    'score': 0.99266666},
   {'label': {'id': 3, 'name': 'rel'},
    'box_norm': [0.8016169 , 0.39198196, 0.9586301 , 0.5374013 ],
    'box': [1414, 1497, 1691, 2052],
    'score': 0.9909662},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.27778038, 0.002736  , 0.7902849 , 0.25589988],
    'box': [ 490,   10, 1394,  977],
    'score': 0.989134},
   {'label': {'id': 3, 'name': 'rel'},
    'box_norm': [0.03168968, 0.65816694, 0.17212266, 0.7855497 ],
    'box': [  55, 2514,  303, 3000],
    'score': 0.98225397},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.3983488, 0.7408758, 0.6080718, 1.       ], 
    'box': [ 702, 2830, 1072, 3820],
    'score': 0.97994417},
   {'label': {'id': 2, 'name': 'weak_entity'},
    'box_norm': [0.76961327, 0.6065664 , 0.984624  , 0.8773991 ], 
    'box': [1357, 2317, 1736, 3351],
    'score': 0.91416836},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.76956284, 0.60632616, 0.983237  , 0.8709249 ],
    'box': [1357, 2316, 1734, 3326],
    'score': 0.4910127},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.23741356, 0.        , 0.35724053, 0.25868225], 
    'box': [418,   0, 630, 988],
    'score': 0.39984587},
   {'label': {'id': 1, 'name': 'entity'},
    'box_norm': [0.26728418, 0.01075818, 0.38652036, 0.23253429], 
    'box': [471,  41, 681, 888],
    'score': 0.39837205}]}]

output = []
for x in input:
    fileName = x['img']
    img = Image.open(fileName)
    print(fileName)
    for y in x['labels']:
        tArray = [y['label']['name']]
        temp = [0, 0, 0, 0]
        temp[0] = y['box'][1] #xmin
        temp[1] = y['box'][0] #ymin
        temp[2] = y['box'][3] #xmax
        temp[3] = y['box'][2] #ymax
        cropped_img = img.crop(temp)
        cropped_img = cropped_img.convert('RGB')
        cropped_img.save('cropped_img.jpg')
        tArray += reader.readtext('cropped_img.jpg', detail = 0)
        #print(tArray)
        output.append(tArray)
        tArray = []
print(output)