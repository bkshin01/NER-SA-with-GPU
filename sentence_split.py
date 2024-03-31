import kss

def spliter(df):
    sentence_list=[]
    s_text = kss.split_sentences(df.본문[0])
    sentence_list.append(s_text)
    
    return sentence_list