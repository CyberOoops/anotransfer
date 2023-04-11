import json
with open('./sample/scripts/transfer_entirely/AIOPS_result.json','r') as f: 
    r = json.loads(f.read())
sum = 0
""" for i in r:
    sum += r[i]
average = sum/len(r)
print(average) """

for i in r:
    js = r[i]
    sum = 0
    for p in js:
        sum = sum + js[p]
    average = sum/len(js)
    print(len(js))
    print(average)