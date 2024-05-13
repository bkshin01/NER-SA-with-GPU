import torch
import tensorflow as tf
import numpy as np
import torch.cuda.amp as amp
import os
from transformers import RobertaForSequenceClassification
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm_notebook
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 1. GPU, 모델 경로 세팅
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NER_MODLE_PATH = "NER_model"
SA_MODLE_PATH = "SA_model/klue_large_fold3_s.pth"


# 2. NER 모델 세팅
NER_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer_for_ner = AutoTokenizer.from_pretrained(NER_MODEL_NAME)

tag2id = {'B-ORG': 0, 'I-ORG': 1, 'O': 2}
unique_tags={'B-ORG', 'I-ORG', 'O'}
id2tag={0: 'B-ORG', 1: 'I-ORG', 2: 'O'}
pad_token_id = tokenizer_for_ner.pad_token_id # 0
cls_token_id = tokenizer_for_ner.cls_token_id # 101
sep_token_id = tokenizer_for_ner.sep_token_id # 102
pad_token_label_id = tag2id['O']    # tag2id['O']
cls_token_label_id = tag2id['O']
sep_token_label_id = tag2id['O']

model = AutoModelForTokenClassification.from_pretrained(NER_MODLE_PATH, num_labels=len(unique_tags))
model.to(device)

# NER 관련 함수 정의
def ner_tokenizer(sent, max_seq_length):
    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable
        pre_syllable = syllable

        input_ids[i] = (tokenizer_for_ner.convert_tokens_to_ids(syllable))
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids
    input_ids[len(sent)+1] = sep_token_id
    attention_mask = [1] + attention_mask
    attention_mask[len(sent)+1] = 1
    return {"input_ids":input_ids,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids}

def ner_inference(text) :
    words = []
    tags= []
    model.eval()
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text)+2)
    input_ids = torch.tensor(tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    for i, tag in enumerate(pred_tags):
      words.append(tokenizer_for_ner.convert_ids_to_tokens(tokenized_sent['input_ids'][i]))
      tags.append(tag)
    return words, tags

def ner_ORG_export(test_tag, test_text):
    b_list = [i for i, x in enumerate(test_tag) if x == "B-ORG"]

    ii_list = []
    for n in range(len(b_list)):
        if n + 1 < len(b_list):
            sample = test_tag[(b_list[n]):(b_list[n+1])]
        else:
            sample = test_tag[(b_list[n]):]
        sample.reverse()

        if "I-ORG" in sample:
            ii_list.append(len(sample) - sample.index("I-ORG"))
        else:
            ii_list.append(0)

    i_list = []
    for i, j in zip(b_list, ii_list):
        i_list.append(i + j)

    ticker_range = list(zip(b_list, i_list))
    ticker_list = []

    for r in ticker_range:
        ticker_list.append(test_text[r[0]-1:r[1]-1])

    return ticker_list


# 3. SA 환경 설정
tf.random.set_seed(1234)
np.random.seed(1234)

class args:
    # ---- factor ---- #
    debug=False
    amp = True
    gpu = '0'

    batch_size=32
    max_len = 128

    start_lr = 1e-5 #1e-3,5e-5
    min_lr=1e-6
    # ---- Dataset ---- #

    # ---- Else ---- #
    num_workers=8
    seed=2021
    scheduler = None #'get_linear_schedule_with_warmup'

# SA tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
tokenizer_for_sa = AutoTokenizer.from_pretrained("klue/roberta-large", cache_dir='bert_ckpt', do_lower_case=False)

# 데이터셋 클래스 정의
class TestDataset(Dataset):
    def __init__(self, df):
        self.df_data = df
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.loc[index, 'text']
        encoded_dict = tokenizer_for_sa(
          text = sentence,
          add_special_tokens = True,
          max_length = args.max_len,
          pad_to_max_length = True,
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, token_type_id , att_mask)
        return sample
    def __len__(self):
        return len(self.df_data)

# 예측 함수 정의
def do_predict(net, valid_loader):
    val_loss = 0
    pred_lst = []
    logit=[]
    net.eval()
    for batch_id, (input_id,token_type_id,attention_mask) in enumerate(valid_loader):
        input_id = input_id.long().to(device)
        token_type_id = token_type_id.long().to(device)
        attention_mask = attention_mask.long().to(device)

        with torch.no_grad():
            if args.amp:
                with amp.autocast():
                    # output
                    output = net(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask)[0]

            else:
                output = net(outputs = model(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask))

            pred_lst.extend(output.argmax(dim=1).tolist())
            logit.extend(output.tolist())
    return pred_lst,logit

def run_predict(model_path,test_dataloader):

    print('set testloader')
    ## net ----------------------------------------
    scaler = amp.GradScaler()
    net = RobertaForSequenceClassification.from_pretrained('klue/roberta-large', num_labels = 3)
    net.to(device)

    if len(args.gpu)>1:
        net = nn.DataParallel(net)

    f = torch.load(model_path)
    net.load_state_dict(f, strict=True)  # True
    print('load saved models')
    # ------------------------
    # validation
    preds, logit = do_predict(net, test_dataloader) #outputs
    print('complete predict')

    return preds, np.array(logit)
