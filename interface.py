from abc import ABC, abstractmethod

class RAGHandler(ABC):
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    @abstractmethod
    def load(self, filepath): 
        pass

    @abstractmethod
    def retrieve(self, question): 
        pass

    @abstractmethod
    def answer(self, question): 
        pass