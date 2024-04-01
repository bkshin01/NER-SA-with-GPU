import torch
import pandas as pd

from model_loader import args, ner_inference, ner_ORG_export, TestDataset, run_predict, SA_MODLE_PATH


def ner_go(sentence_list):
    index_list=[]
    text_list= []
    outputs_list=[]
    tags_list =[]

    for i in range(len(sentence_list)):
        for text in sentence_list[i]:
            words, tags = ner_inference(text)
            index_list.append(i)

            text_list.append(text)
            tags_list.append(tags)
            outputs_list.append(words)

    df=pd.DataFrame(zip(index_list,text_list,outputs_list,tags_list))
    df.columns=['news_index','text','output','tag']

    # export 함수 적용
    df["기업명"] = df.apply(lambda x: ner_ORG_export(x.tag, x.text), axis=1)
    df.기업명 = df.기업명.apply(lambda x: ','.join(dict.fromkeys(x)))

    df=df.drop('news_index', axis=1)
    df=df.drop('output', axis=1)
    df=df.drop('tag', axis=1)

    return df

def sa_go(df):
    test_data = TestDataset(df)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 예측 실행
    preds1, logit1 = run_predict(SA_MODLE_PATH,test_dataloader)

    # 결과 정리
    df['감성분류']=preds1
    df.감성분류=df.감성분류.apply(lambda x: '중립' if x == 0 else '긍정' if x == 1 else '부정')

    return df
