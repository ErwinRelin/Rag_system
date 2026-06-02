import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="qwen3:8b", temperature=0)

extract_prompt = """
You are a DevOps and IT automated triage assistant. Extract ONLY technical and system-identifiable keywords from the issue below.

### Rules:
- Include: named UI components, specific buttons/fields/pages, services, APIs, protocols, error types, config keys, file paths, frameworks, or dev tools.
- A "named UI component" is any specific, identifiable element in the product (e.g., "Pay Rent button", "Login form", "Dashboard page").
- EXCLUDE: generic colors, vague adjectives, verbs, user emotions, and filler words.
- EXCLUDE colors UNLESS they are part of a named constant or config value (e.g., "RED_ALERT_STATUS").

### Task:
Issue: {issue_description}
Output:"""

chain = (
    {"issue_description": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(extract_prompt)
    | llm
    | StrOutputParser()
)

issue = "Offline contractor update overwrites higher priority assignment changes after reconnection, conflicting with configured synchronization priority."

keywords = chain.invoke(issue)
print("Keywords:", keywords)

def highlight_keywords(sentence, keywords_str):
    # Strip any surrounding quotes from each keyword
    keywords = [kw.strip().strip("'\"`") for kw in keywords_str.split(",")]
    keywords.sort(key=len, reverse=True)
    for keyword in keywords:
        words = keyword.split()
        flexible_pattern = r"[\W]+".join(re.escape(word) for word in words)
        # Match optional surrounding quotes, replace entire match with [keyword]
        sentence = re.sub(
            rf"['\"`]?{flexible_pattern}['\"`]?",
            f"[{keyword}]",
            sentence,
            flags=re.IGNORECASE
        )
    return sentence

result = highlight_keywords(issue, keywords)
print("Result:", result)