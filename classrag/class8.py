from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import re


class Translation:
    def sarvam(self, language: str):
        sarvam_llm = ChatOllama(model="mashriram/sarvam-1:latest", temperature=0)

        translate_system = f"""You are an expert {language} translator specializing in technical content.
Your goal is to translate English technical text into {language} accurately.

Follow these guidelines:
- Preserve the original meaning precisely — do not simplify or omit technical details.
- Sentence structure must follow: subject → object → verb.
- The acting subject always comes first in the sentence.
- Translate "overwrites" strictly as "மாற்றி எழுதுகிறது" — never use மேலோட்டம்.
- For technical terms with no natural {language} equivalent, keep the English term in {language} script.
- Return only the translated {language} text. No explanations or original English."""

        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", translate_system),
            ("human", f"Translate the following technical sentence from English to {language}: {{sentence}}")
        ])

        return (
            {"sentence": RunnablePassthrough()}
            | translate_prompt
            | sarvam_llm
            | StrOutputParser()
        )

    def naturalizer(self, language: str):
        qwen_llm = ChatOllama(model="qwen3:8b", temperature=0)

        naturalize_system = f"""You are a {language} language expert who specializes in
converting formal {language} into natural spoken {language}.

Follow these guidelines:
- Keep all technical meaning intact — do not simplify or omit details.
- Write the way a developer would say it to a colleague in a standup.
- Do NOT use Tanglish or transliterate English words into {language} script.
- Write in pure {language} only.
- Do NOT change the subject or who is performing the action.
- Do NOT turn the sentence into a question or conditional.
- Return only the rewritten {language} text. No explanations."""

        naturalize_prompt = ChatPromptTemplate.from_messages([
            ("system", naturalize_system),
            ("human", f"Rewrite this formal {language} in natural spoken {language}: {{tamil_sentence}}")
        ])

        return (
            {"tamil_sentence": RunnablePassthrough()}
            | naturalize_prompt
            | qwen_llm
            | StrOutputParser()
        )

    @staticmethod
    def clean_thinking(text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def run(self, sentence: str, language: str) -> dict:
        sarvam_chain = self.sarvam(language)
        qwen_chain = self.naturalizer(language)

        print("Translating...")
        accurate = sarvam_chain.invoke(sentence)
        print(f"sarvam-1 output:\n{accurate}\n")

        print("Naturalizing...")
        natural = self.clean_thinking(qwen_chain.invoke(accurate))
        print(f"Final output:\n{natural}")

        return {"accurate": accurate, "natural": natural}


if __name__ == "__main__":
    sentence = input("Enter sentence to translate: ")
    language = input("Enter target language: ")

    t = Translation()
    t.run(sentence, language)