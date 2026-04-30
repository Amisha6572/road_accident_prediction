import json
nb = json.load(open('code.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:150].replace('\n', ' ')
    print(f"[{i}] {c['cell_type']}: {src}")
