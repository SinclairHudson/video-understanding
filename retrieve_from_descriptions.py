from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import pandas as pd
from tqdm import tqdm


documents = []

df = pd.read_csv('msr-vtt_descriptions.csv')
for index, row in df.iterrows():
    documents.append(Document(page_content=row['description'], metadata={'id': row['video_id']}))

retriever = BM25Retriever.from_documents(documents)

# results = retriever.get_relevant_documents("red car", k=10)
# for result in results:
    # print(result.metadata['id'])

df = pd.read_csv('/media/sinclair/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv')
recall_4 = 0
for index, row in tqdm(df.iterrows()):
    results = retriever.get_relevant_documents(row['sentence'], k=10)
    ids = [result.metadata['id'] for result in results]
    if row['video_id'] in ids:
        recall_4 += 1
