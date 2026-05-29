import ollama
import re
from keybert import KeyBERT

# ─────────────────────────────────────────────
# 1. MODE DETECTION
# ─────────────────────────────────────────────

def detect_sentence_mode(english_text):
    text_lower = english_text.lower()

    expected_keywords = [
        "should", "must", "expected", "need to", "supposed to", "have to"
    ]
    bug_active_keywords = [
        "indefinitely", "grayed out", "greyed out", "frozen",
        "keeps spinning", "not responding", "spins", "hangs"
    ]
    bug_subtle_keywords = [
        "accepts", "allows", "skips", "ignores", "overwrites",
        "outside", "beyond", "incorrect", "wrong", "invalid",
        "partial", "missing", "unexpected",
        "instead of", "instead",
        "displays as", "shows as",
        "remains active",     
        "persists after",    
        "still active",      
        "not invalidated",    
        "not terminated",
    ]

    if any(kw in text_lower for kw in expected_keywords):
        return "expected"
    elif any(kw in text_lower for kw in bug_active_keywords):
        return "bug"
    elif any(kw in text_lower for kw in bug_subtle_keywords):
        return "bug_subtle"
    else:
        return "neutral"


# ─────────────────────────────────────────────
# 2. MODE INSTRUCTIONS + EXAMPLES
# ─────────────────────────────────────────────

MODE_INSTRUCTION = {
    "expected": (
        "This is EXPECTED BEHAVIOUR — what should happen. "
        "Translate 'should'/'must' strictly as 'pannanum' or 'aaganum'. "
        "Never use 'pannunga' (command) or nonsense words."
    ),
    "bug": (
        "This is an ACTIVE BUG — something is stuck or looping wrong. "
        "Use verb endings: aagudu (ongoing), aayiduchu (got stuck), aagala (never happened)."
    ),
    "bug_subtle": (
        "This is a SUBTLE BUG — the system IS doing something it should NOT do. "
        "Use 'pannudu' to show the wrong action is actively happening."
    ),
    "neutral": (
        "This is a factual statement. Translate naturally using pannudu or aagudu."
    ),
}

MODE_EXAMPLES = {
    "expected": """
English  : The login should redirect to the dashboard
Tanglish : Login dashboard-ku redirect aaganum

English  : The error message should disappear after 3 seconds
Tanglish : Error message 3 seconds-la disappear aaganum

English  : The handler should follow the configured retry policy
Tanglish : Handler configured retry policy-a follow pannanum

English  : Payment submission should enforce defined payment constraints
Tanglish : Payment submission defined payment constraints-a enforce pannanum
""",

    "bug": """
English  : The page freezes after clicking
Tanglish : Click pannina udane page freeze aagudu

English  : The icon keeps spinning with no end
Tanglish : Icon endha neramum spin aagudu

English  : The request never completes
Tanglish : Request complete aagala

English  : The submit button is grayed out
Tanglish : submit button gray-out aayiduchu
""",

    "bug_subtle": """
English  : The form accepts negative values outside the valid range
Tanglish : Form valid range-ku veliye negative values-a accept pannudu

English  : The filter ignores the configured date constraints
Tanglish : Filter configured date constraints-a ignore pannudu

"English  : User session remains active after password reset\n"
"Tanglish : Password reset aana appuram user session active-aa irukkudhu, terminate aaganum\n"

"English  : The field displays raw ID instead of the configured label\n"
"Tanglish : Field configured label-a display pannanum, aana raw ID-a display pannudu\n"

English  : The scheduler skips locked records during sync
Tanglish : Scheduler sync-la locked records-a skip pannudu

English  : Offline update overwrites higher priority changes after reconnection
Tanglish : Reconnect aana appuram offline update higher priority changes-a overwrite pannudu
""",

    "neutral": """
English  : The sync process skips deleted records after reconnection
Tanglish : Reconnect aana appuram sync process deleted records-a skip pannudu

English  : Background job overwrites the existing configuration on restart
Tanglish : Restart aana appuram background job existing configuration-a overwrite pannudu

English  : The scheduler follows the configured retry limit
Tanglish : Scheduler configured retry limit-a follow pannudu
""",
}

def add_square_brackets():
    kw_model = KeyBERT()
    doc = "KeyBERT is a minimal and easy-to-use keyword extraction technique."

    weighted_keywords = kw_model.extract_keywords(doc)

    # 2. Extract only the keywords
    keywords = [kw[0] for kw in weighted_keywords]

    # Loop through each keyword and replace it with a modified version
    for kw in keywords:
        # \b ensures we only match whole words/phrases, flags=re.IGNORECASE ignores capitalization
        pattern = re.compile(rf'\b{re.escape(kw)}\b', flags=re.IGNORECASE)
        doc = pattern.sub(f"[{kw}]", doc)

    return doc
# ─────────────────────────────────────────────
# 3. SENTENCE SPLITTER  (protects [] blocks)
# ─────────────────────────────────────────────

def split_sentence(text):
    """
    Split on ', but' or commas/ands while keeping [] blocks intact.
    Returns list of (chunk, is_after_but) tuples.
    """
    protected = {}
    counter = [0]

    def protect(m):
        key = f"__BLOCK{counter[0]}__"
        protected[key] = m.group(0)
        counter[0] += 1
        return key

    safe = re.sub(r'\[.*?\]', protect, text)

    # Split on ", but" first
    parts = re.split(r',\s*but\s*', safe, maxsplit=1)
    has_but = len(parts) > 1

    before_but = re.split(r',\s*(?:and\s*)?|\band\b\s*', parts[0])
    after_but  = re.split(r',\s*(?:and\s*)?|\band\b\s*', parts[1]) if has_but else []

    chunks = (
        [(c.strip(), False) for c in before_but if c.strip()] +
        [(c.strip(), True)  for c in after_but  if c.strip()]
    )

    # Restore [] blocks
    restored = []
    for chunk, is_after_but in chunks:
        for key, val in protected.items():
            chunk = chunk.replace(key, val)
        restored.append((chunk.strip(), is_after_but))

    return restored


# ─────────────────────────────────────────────
# 4. CHUNK TRANSLITERATOR
# ─────────────────────────────────────────────

def tanglish_chunk(english_chunk, mode="neutral"):
    prompt = f"""You are a Tanglish transliterator. Tanglish = Tamil grammar written in English letters (Latin alphabet only, never Tamil script).

CONTEXT: {MODE_INSTRUCTION[mode]}

EXAMPLES:
{MODE_EXAMPLES[mode]}

STRICT RULES:
- Translate ONLY the given phrase — do NOT add explanations, suggestions, or extra sentences
- ONE line output only — no second sentence, no period mid-output
- "should" / "must"        → pannanum or aaganum (NEVER pannunga, NEVER aaru)
- "overwrites"             → overwrite pannudu  (present wrong action, never in brackets)
- "follow" + should        → follow pannanum     (ONE verb, never "pannanum follow aaganum")
- "indefinitely"           → endha neramum ... aagudu  (ongoing, never aagala)
- Words inside [] stay EXACTLY as written, brackets included — do NOT invent new brackets
- Keep as English: button, spinner, checkout, coupon, amount, price, discount, sync, priority, config, scheduler
- Do NOT start output with "aana", "and", or "but"
- Output ONLY the Tanglish phrase. One line. No explanation.
- For "X instead of Y" sentences: always say the correct behaviour first, 
  then the wrong behaviour.
  Pattern: "[correct thing]-a display pannanum, aana [wrong thing]-a display pannudu

English phrase: {english_chunk}
Tanglish:"""

    response = ollama.chat(
        model="tanglish-gemma:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


# ─────────────────────────────────────────────
# 5. CHUNK CLEANER
# ─────────────────────────────────────────────

def clean_chunk(text):
    """Strip trailing punctuation and leading connectors the model may add."""
    text = text.strip().rstrip(".,;")
    text = re.sub(
        r"^(aana|and|but|amma)[,\.\s]+", "", text, flags=re.IGNORECASE
    ).strip()
    return text


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def english_to_tanglish(english_text, verbose=True):
    mode = detect_sentence_mode(english_text)
    chunks = split_sentence(english_text)

    if verbose:
        print(f"  → Mode detected : {mode}")
        print(f"  → Split into {len(chunks)} chunk(s):")
        for i, (c, after_but) in enumerate(chunks):
            print(f"     [{i+1}] {'[BUT] ' if after_but else ''}{c}")

    tanglish_parts = []
    for chunk, is_after_but in chunks:
        
        raw     = tanglish_chunk(chunk, mode=mode)
        cleaned = clean_chunk(raw)
        if verbose:
            print(f"  → Chunk tanglish: {cleaned}")
        tanglish_parts.append((cleaned, is_after_but))

    # Join — comma for list items, "aana" only across a "but" boundary
    output = ""
    for i, (part, is_after_but) in enumerate(tanglish_parts):
        if i == 0:
            output = part
        elif is_after_but:
            output += ", aana " + part
        else:
            output += ", " + part

    return output


# ─────────────────────────────────────────────
# 7. TEST SENTENCES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "Users intermittently receive a successful login confirmation, but the dashboard fails to load completely.",
        "Successful authentication should consistently redirect users to a fully rendered dashboard without missing components.",
        "The customer profile API returns incomplete address details when optional fields are omitted during account creation.",
        "The API should maintain consistent response structures regardless of optional input availability.",
        "Action buttons overlap with form labels when browser zoom exceeds standard display scaling.",
        "UI components should maintain alignment and readability across supported display and zoom configurations."
    ]

    for sentence in test_sentences:
        print(f"\nEnglish  : {sentence}\n")
        result = english_to_tanglish(sentence)
        print(f"\nTanglish : {result}")
        print("=" * 70)