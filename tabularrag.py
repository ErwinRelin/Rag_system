from interface import RAGHandler
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TabularRAGHandler(RAGHandler):
    def load(self, filepath):
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported tabular format: {filepath}")

    def retrieve(self, question):
        keyword_prompt = f"""Extract the main search keywords from this question as a comma separated list.
        Only return the keywords, nothing else.
        Question: {question}"""
        keywords_response = self.llm.invoke(keyword_prompt).content
        keywords = [k.strip().lower() for k in keywords_response.split(",")]

        def row_matches(row):
            row_str = " ".join(str(v).lower() for v in row.values)
            return any(kw in row_str for kw in keywords)

        filtered_df = self.df[self.df.apply(row_matches, axis=1)]
        if filtered_df.empty:
            return "No matching records found."

        return "\n".join(
            " | ".join([f"{col}: {row[col]}" for col in self.df.columns])
            for _, row in filtered_df.iterrows()
        )

    def answer(self, question):
        context = self.retrieve(question)
        template = """You are a precise assistant. Answer ONLY using the context below.
        Go through ALL entries and list every match that answers the question.

        Context:
        {context}

        Question: {question}

        Answer:"""

        chain = ChatPromptTemplate.from_template(template) | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})