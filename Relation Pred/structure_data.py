import json

with open("raw.json",'r', encoding="utf8") as anno_file:
    new = []
    l = []
    for item in anno_file:
        try:
            # print(item)
            a = json.loads(item)

            for p in a['passages']:
                
                dicts = {}
                dicts['sentText'] = p['passageText']
                dicts['relationMentions'] = []
                
                flag = False
                for arr in p['facts']:
                    annosplit = arr['humanReadable'].split('> <')
                    if('DATE_OF_BIRTH' in annosplit[1] or 'RESIDENCE' in annosplit[1] or 'BIRTH' in annosplit[1] 
                        or 'NATIONALITY' in annosplit[1] or 'EMPLOYEE_OF' in annosplit[1] or 'EDUCATED_AT' in annosplit[1]):
                        print(annosplit[1])
                        l.append(annosplit[1])
                        flag = True
                        dicts['relationMentions'].append({'em1Text': annosplit[0][1:], 'em2Text': annosplit[2][:len(annosplit[2]) - 1], 'label': annosplit[1]})

                if(flag):

                    new.append(json.dumps(dicts))
        except Exception as e:
            print(e)

print(new, l)

from sklearn.model_selection import train_test_split
train, test = train_test_split(new, test_size = 0.2)

with open('train.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train))

with open('test.json', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test))