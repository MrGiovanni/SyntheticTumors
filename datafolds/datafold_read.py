import os
import json


def datafold_read(datalist, basedir, fold=0, key='training'):
    '''

    :param datalist: json file (filename only) with the list of data
    :param basedir: directory of json file
    :param fold: which fold to use (0..1 if in training set)
    :param key: usually 'training' , but can try 'validation' or 'testing' to get the list data without labels (used in challenges)
    :return:  our own 2 arrays (training, validation)
    '''

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr=[]
    val=[]
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val
