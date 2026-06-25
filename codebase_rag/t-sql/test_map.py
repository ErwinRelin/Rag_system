#!/usr/bin/env python3
"""
SQL Chunk Map-Reduce RAG System
Reads chunks.json from C# parser, creates summaries, routes queries to exact chunks
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import ollama

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql")
SHARED_DATA_DIR = BASE_DIR / "shared_data"
CHUNKS_FILE = SHARED_DATA_DIR / "chunks.json"
MODEL_TAG = 'qwen2.5-custom:latest'  # or 'mistral:7b', 'llama3.1:8b'

# ═══════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════

def load_chunks_from_json(file_path: Path) -> Dict[str, Dict]:
    """
    Load chunks from C# parser JSON output.
    Returns dictionary keyed by chunk ID for instant lookup.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n❌ Chunks file not found: {file_path}\n"
            f"Run the C# parser first to generate chunks.json"
        )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 Loaded {len(data)} chunks from: {file_path}")
    
    # Build CPU RAM store (keyed by chunk ID for instant lookup)
    cpu_ram_store = {}
    
    for item in data:
        chunk_id = f"CHUNK_{item['chunkId']:03d}"  # CHUNK_001, CHUNK_002, etc.
        
        # Build rich context for each chunk
        raw_content = f"""
FILE: {item.get('fileName', 'Unknown')}
TYPE: {item.get('objectType', 'Unknown')}
CATEGORY: {item.get('chunkCategory', 'Unknown')}
NAME: {item.get('objectName', 'Unknown')}
DESCRIPTION: {item.get('nlDescription', 'No description')}
REFERENCES: {item.get('references', 'None')}

SQL CODE:
{item.get('sqlText', 'No SQL code')}

FULL CONTEXT:
{item.get('fullContextBlock', 'No context')}
"""
        
        cpu_ram_store[chunk_id] = {
            "content": raw_content.strip(),
            "metadata": {
                "file_name": item.get('fileName'),
                "object_type": item.get('objectType'),
                "chunk_category": item.get('chunkCategory'),
                "object_name": item.get('objectName'),
                "nl_description": item.get('nlDescription'),
                "references": item.get('references'),
                "sql_text": item.get('sqlText'),
                "full_context_block": item.get('fullContextBlock'),
            }
        }
    
    return cpu_ram_store

# ═══════════════════════════════════════════════════════════════
# MAP PHASE: Create summaries of each chunk
# ═══════════════════════════════════════════════════════════════

async def run_map_phase(model_tag: str, cpu_ram_store: Dict) -> str:
    """
    MAP PHASE: Create high-level summaries of each chunk.
    Returns a master index mapping summaries to chunk IDs.
    """
    print("\n" + "="*60)
    print("🗺️  MAP PHASE: Creating chunk summaries")
    print("="*60)
    
    summaries = []
    total_chunks = len(cpu_ram_store)
    
    for i, (chunk_id, chunk_data) in enumerate(cpu_ram_store.items(), 1):
        raw_code = chunk_data["content"]
        metadata = chunk_data["metadata"]
        
        # Build prompt for summarization
        prompt = f"""
Analyze this SQL database schema chunk. Write a 1-2 sentence summary of:
- What database object it defines (table, view, procedure, etc.)
- Its business purpose or key functionality
- Any important relationships or dependencies

You MUST start your response exactly with: 'ID: {chunk_id} | Summary: '

CHUNK INFO:
- Object: {metadata['object_name']}
- Type: {metadata['object_type']}
- Category: {metadata['chunk_category']}

CODE:
{raw_code[:2000]}  # Limit context to avoid overflow
"""
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda p=prompt: ollama.generate(model=model_tag, prompt=p)
            )
            
            summary_output = response['response'].strip()
            print(f"[{i}/{total_chunks}] {chunk_id}: {summary_output[:80]}...")
            summaries.append(summary_output)
            
        except Exception as e:
            print(f"[{i}/{total_chunks}] {chunk_id}: ⚠️ Error: {e}")
            # Fallback summary using metadata
            summaries.append(
                f"ID: {chunk_id} | Summary: "
                f"{metadata['object_type']} defining {metadata['object_name']} - "
                f"{metadata['nl_description'][:100]}"
            )
    
    return "\n".join(summaries)

# ═══════════════════════════════════════════════════════════════
# QUERY PHASE: Find and retrieve exact chunk
# ═══════════════════════════════════════════════════════════════

async def run_query_phase(
    model_tag: str, 
    master_index: str, 
    user_query: str,
    cpu_ram_store: Dict
) -> Optional[Dict]:
    """
    QUERY PHASE: Use the master index to find the right chunk ID,
    then pull the full raw content from CPU RAM.
    """
    print("\n" + "="*60)
    print("🔍 QUERY PHASE: Finding relevant chunk")
    print("="*60)
    print(f"Query: {user_query}")
    
    prompt = f"""
You are a database schema routing assistant. Review this Master Chunk Index:

{master_index}

Based on the user's query below, identify which CHUNK ID contains the schema object they need.
Your output must ONLY be the chunk ID (e.g., CHUNK_001, CHUNK_002).
Do not write anything else - just the ID.

User Query: {user_query}

Most relevant CHUNK ID:"""
    
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: ollama.generate(model=model_tag, prompt=prompt)
        )
        
        target_id = response['response'].strip()
        # Clean up: extract just the CHUNK_XXX pattern
        import re
        match = re.search(r'CHUNK_\d+', target_id)
        if match:
            target_id = match.group(0)
        
        print(f"🎯 Router identified: '{target_id}'")
        
        # FETCH PHASE: Instant lookup from CPU RAM
        if target_id in cpu_ram_store:
            return {
                "chunk_id": target_id,
                "data": cpu_ram_store[target_id]
            }
        else:
            print(f"❌ Chunk ID '{target_id}' not found in store.")
            # Try fuzzy search by object name
            for chunk_id, chunk_data in cpu_ram_store.items():
                if target_id.lower() in chunk_data["metadata"]["object_name"].lower():
                    print(f"🔍 Found similar: {chunk_id}")
                    return {"chunk_id": chunk_id, "data": chunk_data}
            
            return None
            
    except Exception as e:
        print(f"❌ Error during query phase: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# SEARCH BY KEYWORDS (Fallback)
# ═══════════════════════════════════════════════════════════════

def search_by_keywords(query: str, cpu_ram_store: Dict) -> List[str]:
    """
    Simple keyword search as fallback if LLM routing fails.
    """
    query_lower = query.lower()
    matches = []
    
    for chunk_id, chunk_data in cpu_ram_store.items():
        metadata = chunk_data["metadata"]
        content = chunk_data["content"].lower()
        
        # Score based on keyword matches
        score = 0
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in metadata.get("object_name", "").lower():
                score += 3
            if keyword in metadata.get("nl_description", "").lower():
                score += 2
            if keyword in content:
                score += 1
        
        if score > 0:
            matches.append((score, chunk_id))
    
    matches.sort(reverse=True, key=lambda x: x[0])
    return [chunk_id for _, chunk_id in matches[:5]]

# ═══════════════════════════════════════════════════════════════
# DISPLAY RESULT
# ═══════════════════════════════════════════════════════════════

def display_chunk(chunk_id: str, chunk_data: Dict):
    """Display the retrieved chunk in a readable format"""
    metadata = chunk_data["metadata"]
    
    print("\n" + "="*60)
    print(f"✅ RETRIEVED: {chunk_id}")
    print("="*60)
    print(f"📄 File:      {metadata['file_name']}")
    print(f"📦 Object:    {metadata['object_name']}")
    print(f"🏷️  Type:      {metadata['object_type']}")
    print(f"📂 Category:  {metadata['chunk_category']}")
    print(f"🔗 References: {metadata['references'] or 'None'}")
    print(f"📝 Description: {metadata['nl_description']}")
    print("-"*60)
    print("SQL CODE:")
    print("-"*60)
    print(metadata['sql_text'])
    print("="*60)

# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════

async def interactive_mode(model_tag: str, cpu_ram_store: Dict):
    """Interactive query loop"""
    print("\n" + "="*60)
    print("💬 Interactive Query Mode")
    print("   Type 'exit' to quit")
    print("   Type 'list' to see all chunks")
    print("   Type 'search <keyword>' for keyword search")
    print("="*60)
    
    # Create master index once for reuse
    master_index = await run_map_phase(model_tag, cpu_ram_store)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('exit', 'quit', 'q'):
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'list':
                print("\n📋 Available chunks:")
                for chunk_id, chunk_data in sorted(cpu_ram_store.items()):
                    meta = chunk_data["metadata"]
                    print(f"  {chunk_id}: {meta['object_name']} ({meta['chunk_category']})")
                continue
            
            if query.lower().startswith('search '):
                keyword = query[7:]
                matches = search_by_keywords(keyword, cpu_ram_store)
                if matches:
                    print(f"\n🔍 Found {len(matches)} matches for '{keyword}':")
                    for chunk_id in matches[:5]:
                        display_chunk(chunk_id, cpu_ram_store[chunk_id])
                else:
                    print(f"No matches found for '{keyword}'")
                continue
            
            # Route using LLM
            result = await run_query_phase(model_tag, master_index, query, cpu_ram_store)
            
            if result:
                display_chunk(result["chunk_id"], result["data"])
            else:
                print("\n⚠️  No exact match found. Trying keyword search...")
                matches = search_by_keywords(query, cpu_ram_store)
                if matches:
                    print(f"Found {len(matches)} possible matches. Top result:")
                    display_chunk(matches[0], cpu_ram_store[matches[0]])
                else:
                    print("No matches found. Try rephrasing your question.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    """Main entry point"""
    print("="*60)
    print("  SQL Chunk Map-Reduce RAG System")
    print("  C# Parser → JSON → Python RAG")
    print("="*60)
    
    # Load chunks from C# output
    try:
        cpu_ram_store = load_chunks_from_json(CHUNKS_FILE)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return
    
    # Show statistics
    categories = {}
    for chunk_data in cpu_ram_store.values():
        cat = chunk_data["metadata"]["chunk_category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Chunk Statistics:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} chunks")
    
    # Start interactive mode
    await interactive_mode(MODEL_TAG, cpu_ram_store)


if __name__ == "__main__":
    asyncio.run(main())