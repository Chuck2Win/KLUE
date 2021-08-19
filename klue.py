# 파일 읽어와서 list로 저장하기
# label을 적절하게 변환하기

class preprocessing(object):
    def __init__(self, mecab_tokenizer, tokenizer):
        self.mecab_tokenizer = mecab_tokenizer
        self.tokenizer = tokenizer
        self.labels_bio = {}
        UNUSED = "<unused%d>"  
        
        label = ["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"]
        idx = [UNUSED%i for i in range(len(label))]
        for i,j in zip(label,idx):
            self.labels_bio[i] = j
        
    def get_labels(self) -> list:
        return ['PS','LC','OG','DT','TI','QT','O']
    
    def get_bio_labels(self) -> list:
        return ["B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT", "O"]
    
    def organize(self, data : str) -> list :
        # 데이터를 문서별로 변환
        data = data.split('\n\n')
        data[0] = '##'+data[0].split('##')[-1]
        #doc = data[0] ###
        Data = []
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
                # unigram tokenizer 적용
                tokenized_sent = ' '.join(self.tokenizer.tokenize(original_sentence))
                
                gt = []
                char_idx = 0
                for token in tokenized_sent.split():
                    token = re.sub("▁",'',token)
                    gt.append(labels[char_idx])
                    char_idx+=len(token)
                target = []
                for i,j in zip(tokenized_sent.split(),gt):
                    target.append(i)
                    if j != 'O':
                        target.append(self.labels_bio[j])
                target = ' '.join(target)
                #Data.append((tokenized_sent, target))
                #Data.append((original_sentence,sentence,tokenized_sent,target,gt,chars,labels))
                Data.append((tokenized_sent,target))
        # 방안 2 : generative way
        return Data             

