import numpy as np
import asyncio
import ollama
import hashlib
import time
import uuid
from pathlib import Path
from mempalace.palace import PalaceRef, EmbeddingCollection
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions 

async def cpu_code_fetcher(code_blocks, fifo_queue):
    """Producer: Fetches code blocks and pushes to FIFO queue."""
    for i, block in enumerate(code_blocks):
        await fifo_queue.put(block)
        print(f"[CPU Pipeline] Staged chunk {i+1}/{len(code_blocks)} into FIFO.")
        await asyncio.sleep(0.01)
    await fifo_queue.put(None)

async def gpu_inference_executor(fifo_queue, model_tag, collection, embedding_fn, wing_name, room_name):
    """Consumer: Pulls blocks, runs inference, stores results."""
    print(f"[GPU Engine] Initializing execution pipeline using model: {model_tag}")
    
    while True:
        block = await fifo_queue.get()
        if block is None:
            print("[GPU Engine] No more blocks in FIFO. Execution complete.")
            break
            
        print(f"\n{'='*40}\n[GPU Engine] Processing code snippet...\n{'='*40}")
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: ollama.generate(model=model_tag, prompt=block)
            )
            
            model_output = response['response']
            print(f"[Model Output]:\n{model_output[:200]}...")
            
            verbatim_content = f"Prompt/Code:\n{block}\n\nModel Response:\n{model_output}"
            doc_id = hashlib.md5(f"{time.time()}_{block}".encode()).hexdigest()
            
            # 1. Generate standard raw embeddings
            raw_embeddings = embedding_fn([verbatim_content])
            
            # 2. FORCE clean primitive types by pulling the scalar out of NumPy completely
            # This completely guarantees a list of standard python floats [ [0.0123, -0.456, ... ] ]
            clean_embeddings = [[float(x) for x in vector] for vector in raw_embeddings]
            
            await loop.run_in_executor(
                None,
                lambda: collection.add(
                    ids=[doc_id],
                    embeddings=clean_embeddings,  # Explicitly stripped of NumPy wrappers
                    documents=[verbatim_content],
                    metadatas=[{
                        "model": model_tag, 
                        "type": "pipeline_execution",
                        "wing": wing_name,
                        "room": room_name,
                        "timestamp": time.time()
                    }]
                )
            )
            
            print(f"[MemPalace] Verbatim block written to {wing_name} -> {room_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to execute block: {e}")
            import traceback
            traceback.print_exc()
        finally:
            fifo_queue.task_done()

async def main():
    MODEL_TAG = 'qwen2.5-custom:latest'
    WING = "CodebaseAnalysis"
    ROOM = "PipelineLogs"
    
    try:
        print("[MemPalace] Initializing local database storage...")
        storage_path = Path("./my_local_palace")
        storage_path.mkdir(exist_ok=True)
        
        palace_id = str(uuid.uuid4())
        palace_ref = PalaceRef(
            id=palace_id,
            local_path=str(storage_path.absolute())
        )
        
        print(f"[MemPalace] Palace reference created with ID: {palace_id}")
        
        chroma_client = chromadb.PersistentClient(
            path=str(storage_path.absolute()),
            settings=Settings(anonymized_telemetry=False)
        )
        
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        collection_name = f"{WING}_{ROOM}"
        
        try:
            inner_collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=default_ef
            )
            print(f"[MemPalace] Got existing ChromaDB collection: {collection_name}")
        except:
            inner_collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=default_ef
            )
            print(f"[MemPalace] Created new ChromaDB collection: {collection_name}")
        
        collection = EmbeddingCollection(inner=inner_collection)
        print(f"[MemPalace] EmbeddingCollection wrapper created")
        print(f"[MemPalace] Document count: {collection.count()}")
        
        massive_codebase = [
            "def analyze_data(matrix):\n    # Task 1: Calculate column means\n    return [sum(col)/len(col) for col in zip(*matrix)]",
            "def optimize_weights(lr, epochs):\n    # Task 2: Standard gradient descent loop\n    print(f'Optimizing with learning rate {lr}')"
        ]
        
        fifo_queue = asyncio.Queue(maxsize=2)
        
        # Pass the embedding function `default_ef` into the executor task
        await asyncio.gather(
            cpu_code_fetcher(massive_codebase, fifo_queue),
            gpu_inference_executor(fifo_queue, MODEL_TAG, collection, default_ef, WING, ROOM)
        )
        
        print(f"\n{'='*40}\n[MemPalace Search Test] Retrieving memory for validation...\n{'='*40}")
        
        loop = asyncio.get_running_loop()
        
        # 1. Manually generate raw query embeddings using your initialized default_ef
        raw_query_embeddings = default_ef(["gradient descent loop"])
        
        # 2. Force format the array elements completely into raw Python primitives
        clean_query_embeddings = [[float(x) for x in vector] for vector in raw_query_embeddings]
        
        # 3. Query the collection using 'query_embeddings' instead of 'query_texts'
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=clean_query_embeddings, # Pass the clean python float list here
                n_results=1
            )
        )
        
        if results and results.get('documents') and results['documents'][0]:
            print(f"Found Match!")
            print(f"Stored Content:\n{results['documents'][0][0][:300]}...")
            if results.get('distances'):
                print(f"\nRelevance Distance: {results['distances'][0][0]}")
        else:
            print("No matching memory found.")
            
        print(f"\n[Lexical Search Test]")
        lexical_results = await loop.run_in_executor(
            None,
            lambda: collection.lexical_search(
                query="gradient descent",
                n_results=1
            )
        )
        
        if lexical_results and lexical_results.get('documents') and lexical_results['documents'][0]:
            print(f"Lexical Match Found!")
            print(f"Content: {lexical_results['documents'][0][0][:200]}...")
        else:
            print("No lexical matches found.")
            
        print(f"\n[Stats] Final document count: {collection.count()}")
            
    except Exception as e:
        print(f"[FATAL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Removed collection.close() to prevent downstream AttributeError
        pass

if __name__ == "__main__":
    asyncio.run(main())