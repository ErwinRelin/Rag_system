from interface import RAGHandler
import textract
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentRAGHandler(RAGHandler):
    def load(self, filepath):
        text_bytes = textract.process(filepath)
        text = text_bytes.decode('utf-8')
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )

    def retrieve(self, question):
        docs = self.retriever.invoke(question)
        return "\n".join(doc.page_content for doc in docs)

    def answer(self, question):
        template = """You are a precise assistant. Answer ONLY using the context below.
        If the answer is not present, say "I don't know based on the provided document."

        Context:
        {context}

        Question: {question}

        Answer:"""

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)

