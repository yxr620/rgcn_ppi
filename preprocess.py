import dgl
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
import random
from shutil import copyfile


def process(info_file, link_file, GTEx_file, GO_file, result_link, result_GTEx, result_GO):
    extid = np.loadtxt(info_file, dtype='str', delimiter='\t', skiprows=1)[:, :2]
    links = np.loadtxt(link_file, dtype='str', delimiter=' ', skiprows=1)[:, :2]
    GTEx = np.loadtxt(GTEx_file, dtype=str, delimiter='\t', skiprows=3)[:, 1:]
    with open(GO_file, 'r') as read_file: 
        GO = json.load(read_file)

    ext_name = set(np.squeeze(extid[:, 1]))
    GTEx_name = set(np.squeeze(GTEx[:, 0]))
    GO_name = set(GO['gene2id'].keys())
    result_name = ext_name & GTEx_name & GO_name

    # result_name.difference_update(random.sample(result_name, 10000))
    # print(len(result_name))

    extid2name = {}
    name2extid = {}
    for extid, name in extid:
        extid2name[extid] = name
        name2extid[name] = extid
    
    links_list = []
    for link in links:
        name1 = extid2name[link[0]]
        name2 = extid2name[link[1]]
        if name1 in result_name and name2 in result_name:
            links_list.append([name1, name2])

    GTEx_list = []
    for line in GTEx:
        if line[0] in result_name:
            GTEx_list.append(line)

    GTEx_array = np.array(GTEx_list)
    GTEx_data = normalize(GTEx_array[:, 1:].astype(float))

    GO_list = []
    GO_feature = GO['embedding']
    for key in GO['gene2id'].keys():
        if key in result_name:
            GO_list.append([key] + GO_feature[GO['gene2id'][key]])
    GO_array = np.array(GO_list)
    GO_data = normalize(GO_array[:, 1:].astype(float))

    with open(result_link, 'w') as write_file:
        for link in links_list:
            write_file.write(f"{link[0]}\t{link[1]}\n")
    with open(result_GTEx, 'w') as write_file:
        for i, line in enumerate(GTEx_data):
            write_file.write(f"{GTEx_array[i][0]}\t")
            for j in range(len(line)):
                if j == len(line) - 1: write_file.write(f"{line[j]}\n")
                else: write_file.write(f"{line[j]}\t")

    with open(result_GO, 'w') as write_file:
        for i, line in enumerate(GO_data):
            write_file.write(f"{GO_array[i][0]}\t")
            for j in range(len(line)):
                if j == len(line) - 1: write_file.write(f"{line[j]}\n")
                else: write_file.write(f"{line[j]}\t")


    print(f"extid {extid}")
    print(f"links {links}")
    print(f"GTEx {GTEx}")
    return result_name
    # print(f"GO {GO}")

def process_train(source_file, result_links, result_name):
    useless_method = {
        'GenomeRNAi;Text Mining',
        'Text Mining',
        'computer analysis',
        'Synlethality;Text Mining',
        'textmining',
        'Text Mining;Synlethality',
        'Decipher;Text Mining',
        'computational analysis',
        'Text Mining;Daisy'}
    with open(source_file, 'r') as readfile:
        with open(result_links, 'w') as writefile:
            for line in readfile.readlines():
                tmp = line.split('\t')
                if tmp[-2] not in useless_method\
                    and tmp[0] in result_name and tmp[2] in result_name:
                    writefile.write(f"{tmp[0]}\t{tmp[2]}\n")

def copyTrain2Test(GTEx_src, GO_src, link_src, GTEx_dst, GO_dst, link_dst):
    copyfile(GTEx_src, GTEx_dst)
    copyfile(GO_src, GO_dst)
    copyfile(link_src, link_dst)

def splitTrain(train_file, test_file, test_ratio):
    train_data = np.loadtxt(train_file, dtype=str, delimiter='\t')
    np.random.shuffle(train_data)
    num_test = int(train_data.shape[0] * test_ratio)
    num_train = train_data.shape[0] - num_test
    test_data = train_data[:num_test]
    train_data = train_data[num_test:]
    with open(train_file, 'w') as write_file:
        for line in train_data:
            write_file.write(f"{line[0]}\t{line[1]}\n")
    with open(test_file, 'w') as write_file:
        for line in test_data:
            write_file.write(f"{line[0]}\t{line[1]}\n")

def normalize(unnorm):
    normed = np.log(unnorm + 1)
    normed = (normed - normed.mean(axis=0)) / normed.std(axis=0)
    # print(f"mean{normed.mean(axis=0)}")
    # print(f"std {normed.std(axis=0)}")
    # print(f"normed\n{normed}")
    # print(f"unnormed\n{unnorm}")
    print(f"2274 \n{normed[2274]}")
    print(type(normed))
    return np.nan_to_num(normed)


if __name__ == "__main__":
    raw_path = "./data/raw"
    ppi_info = "/9606.protein.info.v11.0.txt"
    ppi_links = "/9606.protein.links.v11.0.txt"
    GTEx = "/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    GO = "/GO/dump.txt"

    train_path = "./data/train"
    test_path = "./data/test"
    result_link = "/links.txt"
    result_GTEx = "/GTEx.txt"
    result_GO = "/GO.txt"
    train_source = "/SynLethDB/Human_SL.csv"
    result_train = "/train_data.txt"
    result_test = "/test_data.txt"

    # process(
    #     db_path + ppi_info, db_path + ppi_links, db_path + GTEx, db_path + GO,\
    #     train + result_link, train + result_GTEx, train + result_GO
    # )

    # result_name = set(np.squeeze(
    #     np.loadtxt(db_path + result_GO, dtype=str, delimiter='\t')[:, 0]
    # ))
    # process_train(db_path + train_source, db_path + result_train, result_name)

    copyTrain2Test(train_path + result_GTEx, train_path + result_GO, train_path + result_link,
        test_path + result_GTEx, test_path + result_GO, test_path + result_link)
    splitTrain(train_path + result_train, test_path + result_test, 0.1)