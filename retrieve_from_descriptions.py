from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import pandas as pd
from tqdm import tqdm
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_recall_MSRVTT_BM25(description_path, ks=[1, 5, 10]):
    documents = []
    df = pd.read_csv(description_path)
    description_columns = [col for col in df.columns if 'desc' in col]
    for index, row in df.iterrows():
        content = ' '.join([row[col] for col in description_columns])
        documents.append(Document(page_content=content, metadata={'id': row['video_id']}))

    retriever = BM25Retriever.from_documents(documents, k=max(ks))

    eval_df = pd.read_csv('/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv')
    successes = [0 for _ in ks]
    for index, row in tqdm(eval_df.iterrows()):
        results = retriever.get_relevant_documents(row['sentence'])
        ids = [result.metadata['id'] for result in results]
        for i, k in enumerate(ks):
            if row['video_id'] in ids[:k]:
                successes[i] += 1
    return [success / len(eval_df) for success in successes]

def get_recall_MSRVTT_DPR(description_path, ks=[1, 5, 10]):
    ### WARNING running this costs moneyyyy!
    documents = []
    df = pd.read_csv(description_path)
    description_columns = [col for col in df.columns if 'desc' in col]
    for index, row in df.iterrows():
        content = ' '.join([row[col] for col in description_columns])
        documents.append(Document(page_content=content, metadata={'id': row['video_id']}))

    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    retriever = db.as_retriever(search_kwargs={'k': max(ks)})
    eval_df = pd.read_csv('/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv')
    successes = [0 for _ in ks]
    for index, row in tqdm(eval_df.iterrows()):
        results = retriever.get_relevant_documents(row['sentence'], k=max(ks))
        assert len(results) == max(ks)
        ids = [result.metadata['id'] for result in results]
        for i, k in enumerate(ks):
            if row['video_id'] in ids[:k]:
                successes[i] += 1
    return [success / len(eval_df) for success in successes]

if __name__ == '__main__':
    # print(get_Recall_k_MSRVTT_DPR('descriptions/msr-vtt_descriptions_3strat_0temp_concise.csv'))
    for file in os.listdir("descriptions/"):
        if file.endswith(".csv") and "hockey" not in file and "socc" not in file:
            print(file)
            print("DPR Recall@1,5,10:")
            print(get_recall_MSRVTT_DPR('descriptions/'+file, ks=[1, 5, 10]))
            # print("BM25 Recall@1,5,10:")
            # print(get_recall_MSRVTT_BM25('descriptions/'+file, ks=[1, 5, 10]))
