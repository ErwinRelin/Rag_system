import pandas as pd
import duckdb
import chromadb
import chardet
import sys
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0)

class sql_generation():
    def load(self, filepath):

        if filepath.endswith(".csv"):
            with open(filepath, "rb") as file:
                encoding = chardet.detect(file.read())['encoding']
                self.df = pd.read_csv(filepath, encoding=encoding)
        else:
            raise ValueError(f"unsupported file: {filepath}")
        
        self.filepath = filepath
        self.columns = list(self.df.columns)
        self.rows = list(self.df.iloc[0:3])

        for col in self.df.columns:
            if self.df[col].dtype == object:  # only string columns
            # Check if column looks like it has currency symbols
                sample = self.df[col].dropna().head(10).astype(str)
                has_currency = sample.str.contains(r'[₹$£€?]', regex=True).any()
                if has_currency:
                    print(f"Cleaning currency column: {col}")
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.replace(r'[₹$£€?,]', '', regex=True)  # remove symbols
                        .str.strip()
                        .pipe(pd.to_numeric, errors='coerce')        # convert to float
                    )

        self.filepath = filepath
        self.columns = list(self.df.columns)

        self.con = duckdb.connect()
        self.con.register("data", self.df)   # register dataframe as "data" table
        print("DuckDB table registered with columns:", self.columns)

        # ── ChromaDB setup ────────────────────────────────────────
        chroma_client = chromadb.Client()
        embedding_fn = OllamaEmbeddingFunction(
            url="http://127.0.0.1:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )

        # Fresh collection every load
        try:
            chroma_client.delete_collection("tabular_data")
        except:
            pass

        collection = chroma_client.create_collection(
            name="tabular_data",
            embedding_function=embedding_fn
        )

        # Index each row as a document
        docs, ids, metadatas = [], [], []
        for i, row in self.df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in self.columns])
            docs.append(row_text)
            ids.append(str(i))
            metadatas.append({"row_index": i})

        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        self.collection = collection
        print(f"ChromaDB indexed {len(docs)} rows")

    def _fix_column_quotes(self, sql_query):      # ← must be here
            for col in self.columns:
                if " " in col or "(" in col or "?" in col:
                    print(col)
                    if f'"{col}"' not in sql_query and col in sql_query:
                        print(sql_query)
                        sql_query = sql_query.replace(col, f'"{col}"')
            return sql_query
        
    def _review_sql(self, sql_query, question):
        """Review and fix the generated SQL query before execution"""
        review_prompt = f"""You are a SQL reviewer. Review this DuckDB SQL query and fix any issues.

        Table name: "data"
        Columns: {self.columns}
        Column types: {dict(zip(self.columns, self.df.dtypes.astype(str)))}

        Original question: {question}
        Generated SQL: {sql_query}

        Check for these issues:
        1. All column names must be wrapped in double quotes e.g. "Full Name"
        2. String comparisons must use LOWER(CAST("col" AS VARCHAR))
        3. Table name must be "data"
        4. Query must actually answer the question
        5. No markdown or backticks
        6. Numeric columns should not have CAST for numeric comparisons

        If the query is correct, return it as is.
        If there are issues, return the fixed query.
        Return ONLY the SQL query, nothing else.

        Reviewed SQL:"""

        reviewed_sql = self.llm.invoke(review_prompt).content.strip()
        reviewed_sql = reviewed_sql.replace("```sql", "").replace("```", "").strip()
        return reviewed_sql

    # ── SQL Path ──────────────────────────────────────────────────
    def _sql_retrieve(self, question):

        llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0)
        self.llm = llm

        schema = ", ".join([f'"{col}" ({self.df[col].dtype})' for col in self.columns])

        sql_prompt = f"""You are a SQL expert. Generate a DuckDB SQL query to answer the question.
        The table is named "data" and has these columns with types:
        {schema}

        Sample rows:
        {self.df.head(3).to_string(index=False)}

        Rules:
        - Return ONLY the SQL query, no explanation, no markdown, no backticks
        - ALWAYS wrap ALL column names in double quotes e.g. "Full Name", "Customer ID"
        - Always use the table name "data"
        - For string comparisons ALWAYS cast to VARCHAR first e.g. WHERE LOWER(CAST("City" AS VARCHAR)) = 'chennai'
        - For LIKE queries ALWAYS cast to VARCHAR first e.g. WHERE LOWER(CAST("reviewText" AS VARCHAR)) LIKE '%keyword%'
        - Numeric columns are already cleaned, do NOT use CAST on them for numeric comparisons
        - ALWAYS include relevant filter columns in SELECT for context

        Question: {question}
        SQL:"""

        sql_query = self.llm.invoke(sql_prompt).content.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = self._fix_column_quotes(sql_query)
        print(f"Generated SQL: {sql_query}")

        # ── Review the SQL ────────────────────────────────────────────
        reviewed_sql = self._review_sql(sql_query, question)
        reviewed_sql = self._fix_column_quotes(reviewed_sql)
        print(f"Reviewed SQL:  {reviewed_sql}")

        try:
            result = self.con.execute(reviewed_sql).df()

            if result.empty:
                return "No results found for this query."

            total_rows = len(result)

            col_widths = {col: max(len(str(col)), result[col].astype(str).str.len().max())
                        for col in result.columns}

            header = " | ".join(f"{col:<{col_widths[col]}}" for col in result.columns)
            divider = "-+-".join("-" * col_widths[col] for col in result.columns)

            rows = []
            for _, row in result.iterrows():
                row_str = " | ".join(f"{str(row[col]):<{col_widths[col]}}" for col in result.columns)
                rows.append(row_str)

            formatted = (
                f"Query: {reviewed_sql}\n"
                f"Results: {total_rows} record(s) found\n"
                f"\n"
                f"{header}\n"
                f"{divider}\n"
                + "\n".join(rows)
            )

            return formatted

        except Exception as e:
            print(f"SQL error: {e}")

    def answer(self, question):
        context = self._sql_retrieve(question)
        print(f"Context received in answer(): {context}")
        
        template = """You are a precise assistant. Answer ONLY using the context below.
        Go through ALL entries and list every match that answers the question.
        Do not invent or infer information not explicitly stated.

        Context:
        {context}

        Question: {question}

        Answer:"""

    # ← outside the if/else so it runs for both SQL and SEMANTIC
        chain = ChatPromptTemplate.from_template(template) | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    
if __name__ == "__main__":
    filepath = sys.argv[1]
    question = input("Enter a question: ")

    sql = sql_generation()
    sql.load(filepath)

    result = sql.answer(question)
    print(f"Answer: {result}")



