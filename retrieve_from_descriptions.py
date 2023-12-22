from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import pandas as pd
from tqdm import tqdm
from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings



def get_Recall_k_MSRVTT_BM25(description_path, k=5):
    documents = []
    df = pd.read_csv(description_path)
    for index, row in df.iterrows():
        documents.append(Document(page_content=row['description'], metadata={'id': row['video_id']}))

    retriever = BM25Retriever.from_documents(documents, k=k)

    eval_df = pd.read_csv('/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv')
    successes = 0
    for index, row in tqdm(eval_df.iterrows()):
        results = retriever.get_relevant_documents(row['sentence'])
        assert len(results) == k
        ids = [result.metadata['id'] for result in results]
        if row['video_id'] in ids:
            successes += 1
    return successes / len(eval_df)

def get_Recall_k_MSRVTT_DPR(description_path, k=5):
    ### WARNING running this costs moneyyyy!
    documents = []
    df = pd.read_csv(description_path)
    for index, row in df.iterrows():
        documents.append(Document(page_content=row['description'], metadata={'id': row['video_id']}))

    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    retriever = db.as_retriever()
    eval_df = pd.read_csv('/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv')
    successes = 0
    for index, row in tqdm(eval_df.iterrows()):
        results = retriever.get_relevant_documents(row['sentence'], k=k)
        assert len(results) == k
        ids = [result.metadata['id'] for result in results]
        if row['video_id'] in ids:
            successes += 1
    return successes / len(eval_df)

if __name__ == '__main__':
    # print(get_Recall_k_MSRVTT_DPR('descriptions/msr-vtt_descriptions_3strat_0temp_concise.csv'))
    print(get_Recall_k_MSRVTT_BM25('descriptions/msr-vtt_descriptions_3strat_0temp.csv'))
    print(get_Recall_k_MSRVTT_BM25('descriptions/msr-vtt_descriptions_5strat_0temp_concise.csv'))
    print(get_Recall_k_MSRVTT_BM25('descriptions/msr-vtt_descriptions_5strat_0temp.csv'))
    print(get_Recall_k_MSRVTT_BM25('descriptions/msr-vtt_descriptions_L1180.csv'))
