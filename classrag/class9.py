from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# 1. Define the specific "Tanglish" style with examples
# Notice how technical terms (API, Latency, Cache) are kept in English script,
# while the grammar and connecting words are in Tamil/Tanglish.
examples = [
    {
        "input": "The API response time is very slow due to high latency.",
        "output": "High latency karanam-a, API response time romba slow-a iruku."
    },
    {
        "input": "You need to clear the cache before deploying the new build.",
        "output": "Puthu build-a deploy panrathuku munadi, neenga cache-a clear panna vendum."
    },
    {
        "input": "The database connection failed because of incorrect credentials.",
        "output": "Incorrect credentials nala database connection fail aayiduchu."
    },
    {
        "input": "Please merge the pull request after resolving all merge conflicts.",
        "output": "Ellam merge conflicts-aiyum resolve pannitutu, pull request-a merge pannunga."
    },
    {
        "input": "We need to optimize the query because it is causing a performance bottleneck.",
        "output": "Performance bottleneck-a cause panrathala, intha query-a optimize panna vendiyiruku."
    },
    {
        "input": "The authentication token expired, so you need to log in again.",
        "output": "Authentication token expire aayiduchu, so marubadiyum log in pannaanum."
    },
    {
        "input": "Don't hardcode the API keys; store them in the environment variables.",
        "output": "API keys-a hardcode pannathenga; athai environment variables-la store pannunga."
    },
    {
        "input": "The server crashed due to an out-of-memory error.",
        "output": "Out-of-memory error vanthathala server crash aayiduchu."
    },
    {
        "input": "We are migrating our legacy code to a modern microservices architecture.",
        "output": "Namma legacy code-a modern microservices architecture-uku migrate panrom."
    },
    {
        "input": "Implement form validation on the client side to improve user experience.",
        "output": "User experience-a improve panna, client side-la form validation-a implement pannunga."
    },
    {
        "input": "The scheduler triggers the background job every midnight.",
        "output": "Intha scheduler ovvuru midnight-um background job-a trigger pannum."
    },
    {
        "input": "Dockerize the application and push the image to the registry.",
        "output": "Application-a dockerize pannittu, image-a registry-la push pannunga."
    },
    {
        "input": "We need to write unit tests to increase the overall code coverage.",
        "output": "Overall code coverage-a increase panna, namma unit tests ezhuthanum."
    },
    {
        "input": "The frontend is making asynchronous requests using axios.",
        "output": "Frontend-la irunthu axios use panni asynchronous requests-a make panrom."
    },
    {
        "input": "Enable CORS in the backend configuration to allow cross-origin requests.",
        "output": "Cross-origin requests-a allow panna, backend configuration-la CORS-a enable pannunga."
    },
    {
        "input": "The index on the user ID column speeds up search operations.",
        "output": "User ID column-la irukura index, search operations-a speed up pannuthu."
    },
    {
        "input": "Revert the commit if the pipeline fails during the production deployment.",
        "output": "Production deployment appo pipeline fail aana, commit-a revert pannidunga."
    },
    {
        "input": "This state management library automatically handles components re-rendering.",
        "output": "Intha state management library components re-rendering-a automatically handle pannum."
    },
    {
        "input": "The load balancer distributes incoming traffic across multiple instances.",
        "output": "Incoming traffic-a multiple instances-uku load balancer distribute pannuthu."
    },
    {
        "input": "We are experiencing packet loss because of network congestion.",
        "output": "Network congestion karanam-a namaku packet loss aaguthu."
    },
    {
        "input": "The webhook sends a payload immediately after the payment is successful.",
        "output": "Payment successful aanathum webhook udane oru payload-a send pannum."
    },
    {
        "input": "Initialize the state variable with a null value inside the hook.",
        "output": "Hook-uku ulla state variable-a null value vechu initialize pannunga."
    },
    {
        "input": "Encrypt the sensitive user data before storing it in the disk.",
        "output": "Sensitive user data-va disk-la store panrathuku munadi encrypt pannunga."
    },
    {
        "input": "The memory leak is occurring due to an unhandled subscription.",
        "output": "Unhandled subscription nala memory leak occur aaguthu."
    },
    {
        "input": "Refactor this function to reduce its cyclomatic complexity.",
        "output": "Cyclomatic complexity-a koraika intha function-a refactor pannunga."
    }
]


# 2. Create the example prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 3. Create the few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. Define the final system instruction
# We explicitly tell it to use "Tanglish" and keep technical terms in English.
translate_system = """You are a developer-focused translator. 
Translate the following English technical sentence into 'Tanglish' (Tamil + English).
Rules:
1. Keep all technical keywords (e.g., API, loop, function, database, variable) in English.
2. Use Tamil for grammar, verbs, and connecting words.
3. The output should sound like a casual conversation between two Tamil developers."""

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", translate_system),
        few_shot_prompt,  # Insert the examples here
        ("human", "{sentence}"),
    ]
)

# 5. The Chain
sarvam_llm = ChatOllama(model="qwen2.5-custom:latest", temperature=0.1)

sarvam_chain = (
    {"sentence": RunnablePassthrough()}

    | final_prompt
    | sarvam_llm
    | StrOutputParser()
)

# Step 2 — gemma2 to naturalize the output
# gemma_llm = ChatOllama(model="qwen3:8b", temperature=0)

# naturalize_system = """You are a Tamil language expert who specializes in 
# converting formal Tamil into tamil + English.

# Follow these guidelines:
# - use the original sentence: {sentence} and add the technical english words in the final tamil + english sentence
# - Keep all technical meaning intact — do not simplify or omit details.
# - Write the way a developer would say it to a colleague in a standup.
# - Mix English technical terms naturally — do not translate words like: 
#   update, reconnect, contractor, assignment, priority, error, bug, fix.
# - Return only the rewritten Tamil + English text. No explanations.


# """

# naturalize_prompt = ChatPromptTemplate.from_messages([
#     ("system", naturalize_system),
#     ("human", "Rewrite this formal Tamil in natural spoken Tamil + English: {tamil_sentence}")
# ])

# gemma_chain = (
#     {"tamil_sentence": RunnablePassthrough(), "sentence": RunnablePassthrough()}
#     | naturalize_prompt
#     | gemma_llm
#     | StrOutputParser()
# )

# Run the pipeline
sentence = "Partial payment entry accepts values outside configured allocation behavior."

print("Translating...")
accurate = sarvam_chain.invoke(sentence)
print(f"sarvam-1 output:\n{accurate}\n")

print("Naturalizing...")
# natural = gemma_chain.invoke({"tamil_sentence": accurate, "sentence": sentence})
# print(f"Final output:\n{natural}")