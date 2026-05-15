from interface import RAGHandler
from documentrag import DocumentRAGHandler
from tabularrag import TabularRAGHandler

class RAGHandlerFactory:
    TABULAR_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    @staticmethod
    def get_handler(filepath, llm, embeddings) -> RAGHandler:
        ext = '.' + filepath.rsplit('.', 1)[-1].lower()
        if ext in RAGHandlerFactory.TABULAR_EXTENSIONS:
            return TabularRAGHandler(llm, embeddings)
        elif ext in RAGHandlerFactory.DOCUMENT_EXTENSIONS:
            return DocumentRAGHandler(llm, embeddings)
        else:
            raise ValueError(f"Unsupported file type: {ext}")