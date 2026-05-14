import sys
import asyncio
import textract
import requests
from fastapi import FastAPI
import uvicorn
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
    LLMContextPrecisionWithReference,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

question = input("Enter a Question: ")

def rag(filename):
    print("inside rag")
    llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    context_precision_metric = LLMContextPrecisionWithReference(llm=ragas_llm)

    text_bytes = textract.process(filename)
    text_str = text_bytes.decode('utf-8')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    content = text_splitter.split_text(text_str)

    vectorstore = FAISS.from_texts(content, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    template = """You are a precise assistant. Answer the question using ONLY the context below.
    Be concise and directly address what is asked. Do not add extra information.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    print(response)

    '''queries = [
        'Which mountain is the highest peak in Antarctica and where is it located?',
        'What is the significance of Lake Vostok and how has scientific understanding of it changed?',
        'What are the extreme climate conditions of Antarctica, and what temperature records does it hold?',
        'Who were the key figures in the race to the South Pole in the early 20th century?',
        'How much of the world\'s freshwater is stored in Antarctica, and what would happen if it melted?',
        'What are the primary regulations and human activities permitted under the Antarctic Treaty System?',
    ]

    ground_truths = [
        'The highest peak on the continent is Vinson Massif, which reaches an elevation of 4,892 metres (16,050 ft). It is situated within the Ellsworth Mountains.',
        'Lake Vostok is the largest subglacial lake globally, located deep beneath Russia\'s Vostok Station. Scientists now estimate that its water is replaced every 13,000 years through a slow cycle of melting and freezing ice caps.',
        'Antarctica is characterized as the coldest, driest, and windiest continent on Earth. It holds the world record for the lowest measured temperature at −89.2 °C (−128.6 °F), though coastal regions can reach over 10 °C (50 °F) in summer.',
        'In 1909, Douglas Mawson, Edgeworth David, and Alistair Mackay became the first to reach the magnetic South Pole. In 1911, Roald Amundsen led the first expedition to reach the geographic South Pole.',
        'Antarctica contains approximately 70% of the world\'s freshwater reserves, frozen in its ice sheets. If all this ice melted, global sea levels would rise by almost 60 metres (200 ft).',
        'The 1959 Antarctic Treaty designates the continent as a peaceful zone, prohibiting military activity, mining, nuclear explosions, and nuclear waste disposal. Human activity is limited to scientific research, fishing, and tourism.',
    ]

    results = []
    contexts_list = []
    for query in queries:
        docs = retriever.invoke(query)
        context_content = [doc.page_content for doc in docs]
        contexts_list.append(context_content)
        res = rag_chain.invoke(query)
        results.append(res)

    data = {
        "question": queries,
        "answer": results,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision_metric,   
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    ]


    for metric in metrics:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings  

    # dataset = Dataset.from_dict(data)
    # eval_results = evaluate(dataset=dataset, metrics=metrics)  
    # print(eval_results)'''

if __name__ == "__main__":
     file_name = sys.argv[1]
     rag(file_name)
    