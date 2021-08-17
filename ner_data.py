# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:30:39 2021

@author: OK
"""

# sentence piece의 결과와 유사함.
import os
import argparse
import re
import json
from tqdm import tqdm
import kobert_transformers
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str , default = r'C:\Users\OK\Desktop\프로젝트&공모전\2021\KLUE\klue-ner-v1.1\klue-ner-v1.1_train.tsv')
parser.add_argument('--dev_data', type = str , default = r'C:\Users\OK\Desktop\프로젝트&공모전\2021\KLUE\klue-ner-v1.1\klue-ner-v1.1_dev.tsv')
parser.add_argument('--train_output_dir', type = str , default = r'C:\Users\OK\Desktop\프로젝트&공모전\2021\KLUE\klue-ner-v1.1\klue-ner-v1.1_train.json')
parser.add_argument('--val_output_dir', type = str , default = r'C:\Users\OK\Desktop\프로젝트&공모전\2021\KLUE\klue-ner-v1.1\klue-ner-v1.1_val.json')

# 파일 읽어와서 list로 저장하기
# label을 적절하게 변환하기

class preprocessing(object):
    def __init__(self,tokenizer,strip_char='▁'):
        self.tokenizer = tokenizer
        self.strip_char = strip_char
    
    def get_labels(self) -> list:
        return ['PS','LC','OG','DT','TI','QT']
    
    def get_labels(self) -> list:
        return ["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"]
    
    def organize(self, data : str) -> dict :
        # 데이터를 문서별로 변환
        data = data.split('\n\n')
        data[0] = '##'+data[0].split('##')[-1]
        #doc = data[0] ###
        Data = {}
        for doc in tqdm(data):
            if doc:
                d = [i.split('\t') for i in doc.split('\n')]
                try:
                    key, sentence = d[0]
                except:
                    return 
                # 빈칸을 제거한 chars와 labels    
                chars = []
                labels = []
                original_sentence = ""
                for char,label in d[1:]:
                    original_sentence+=char
                    if char == ' ':
                        continue
                    chars.append(char)
                    labels.append(label)
                tokenized_sent = self.tokenizer.tokenize(original_sentence)
        # 방안 1 : discriminative way
                #sent_s_by_space = original_sentence.split(' ')
                modi_labels = []
                char_idx = 0
                decoder_input = []
                for word in tokenized_sent:
                    token = word.replace(self.strip_char,"")
                    if not token:
                        continue
                    modi_labels.append(labels[char_idx])
                    char_idx+=len(token)
                    decoder_input.append(token)
        # 방안 2 : generative way
                Data[key]={"original_sentence":original_sentence, "sentence":sentence, "chars":chars, "labels":labels,"decoder_input":decoder_input, "modi_labels":modi_labels}
        return Data
        
    # def _make_original(self,sentence):
    #     for label in self.labels:
    #         sentence = re.sub(r":%s"%label,"",sentence)
    #     sentence = re.sub('<|>','',sentence)
    #     return sentence    

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = kobert_transformers.get_tokenizer()
    p = preprocessing(tokenizer)
    train_data = open(args.train_data,'r',encoding = 'utf-8')
    val_data = open(args.dev_data,'r',encoding = 'utf-8')
    
    for i,j in zip([args.train_output_dir, args.val_output_dir],[train_data,val_data]):
        d = j.read()
        d = p.organize(d)
        with open(i,'w') as json_file:
            json.dump(d,json_file)
            
#####################################
    
    
    tokenizer
    f = open(args.train_output_dir,'rb')
    D = json.load(f)
    len(D[list(D.keys())[5]]['modi_labels'])
    print(D[list(D.keys())[5]]['modi_labels'])
    
    for i in range(2):
        print(D[list(D.keys())[i]]['sentence'])
        for i,j in zip(D[list(D.keys())[i]]['decoder_input'],D[list(D.keys())[i]]['modi_labels']):
            print((i,j))
    
    D[list(D.keys())[5]]['modi_labels']
    print(D[list(D.keys())[5]]['labels'])
    print([i for i in tokenizer.tokenize(D[list(D.keys())[5]]['original_sentence'])])
    len([i for i in tokenizer.tokenize(D[list(D.keys())[5]]['original_sentence'])])
    len([i for i in tokenizer.tokenize(D[list(D.keys())[5]]['original_sentence'])])


    len(D[list(D.keys())[5]]['sentence'])
    
    print(D[list(D.keys())[5]]['sentence'])
    
    print(D[list(D.keys())[5]]['labels'])
    print(D[list(D.keys())[5]]['chars'])
    len(D[list(D.keys())[5]]['labels'])    
    len(D[list(D.keys())[5]]['chars'])    

# for i in range(10000):
#     if tokenizer.unk_token in tokenizer.tokenize(D[list(D.keys())[i]]['original_sentence']):
#         break

    # unk 관련
s=D[list(D.keys())[5]]['original_sentence']
s
tokenizer.encode(s)
tokenizer.unk_token_id
s.split(" ")
modi_labels = []
char_idx = 0
original_clean_labels = D[list(D.keys())[5]]['labels']
for word in s.split(" "):   
    l = len(word)
    tokenized_word = tokenizer.tokenize(word)
    print(tokenized_word)    
    contain_unk = True if tokenizer.unk_token in tokenized_word else False
    for i, token in enumerate(tokenized_word):
        token = token.replace("▁","")
        # token이 없다면
        if not token:
            continue
        modi_labels.append(original_clean_labels[char_idx])
        # unk가 없다면
        if not contain_unk:
            char_idx+=len(token)
        
    if contain_unk:
        char_idx+=len(token)
len(modi_labels)
char_idx

original_sentence = D[list(D.keys())[5]]['original_sentence']
labels = D[list(D.keys())[5]]['labels']

############################################################################
tokenized_sent = tokenizer.tokenize(original_sentence)
        # 방안 1 : discriminative way
                #sent_s_by_space = original_sentence.split(' ')
modi_labels = []
char_idx = 0
modi_labels = []
for word in tokenized_sent:
    #contain_unk = True if self.tokenizer.unk_token in tokenized_sent else False
    token = word.replace("▁","")
    
    if not token:
        continue
    print(token)
    modi_labels.append(labels[char_idx])
    char_idx+=len(token)
        
    
len(modi_labels)
len(tokenized_sent)
tokenized_sent
