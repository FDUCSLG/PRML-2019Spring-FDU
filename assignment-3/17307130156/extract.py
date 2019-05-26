# Extracting poems from every json file under path specified by directory.
import os
import json


OUT = './raw_data/out.json'
DIR = './raw_data'

included_extensions = ['.json']
files = [DIR + '/' + fn for fn in os.listdir(DIR) if any(fn.endswith(ext) for ext in included_extensions)]

outputs = []

for fn in files:
    print (fn)
    datas = json.loads(open(fn, 'r').read())
    for p in datas:
        if len(p['paragraphs']) == 4:
            outputs.append(p)

print (len(outputs))


with open(OUT, 'w') as f:
    f.write(json.dumps(outputs, indent=4, ensure_ascii=False))
    
