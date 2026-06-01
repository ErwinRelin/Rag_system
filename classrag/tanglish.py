import ollama
import re
from keybert import KeyBERT
import spacy
import pytextrank

nlp = spacy.load("en_core_web_md")

if "textrank" not in nlp.pipe_names:
    nlp.add_pipe("textrank")
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
        "lacks",
        "fails to",
        "does not",
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
        """This is a factual defect statement.

        Use:

        missing → missing-aa irukku
        lacks → illa / missing-aa irukku
        fails to → aagala
        does not → aagala
        after clicking → click pannina appuram

        Prefer natural spoken Tanglish grammar.
        Avoid literal English grammar ordering."""
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

English : UI components should maintain alignment and readability across supported display and zoom configurations
Tanglish : Supported display-um zoom configurations-um la UI components alignment-um readability-um maintain aaganum

English : The API should maintain consistent response structures regardless of optional input availability
Tanglish : Optional input irundhaalum illaatiyum API consistent response structure-a maintain pannanum
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

English : The login button lacks a functional aria-label attribute
Tanglish : Login button-ku functional aria-label attribute missing-aa irukku

English : The button fails to redirect the user after clicking
Tanglish : Click pannina appuram button user-a redirect aagala

English : The field lacks configured validation rules
Tanglish : Field-ku configured validation rules illa
""",
}

PROTECTED_TERMS = [

# AUTH
"OAuth2",
"JWT",
"MFA",
"OTP",
"CSPRNG",

"Persistent_Refresh_Token",
"access token",
"refresh token",
"token rotation",
"token blocklist",
"token lifecycle",
"session persistence",
"session timeout",
"session validation",

"HttpOnly cookie",
"SameSite=Strict",
"SHA-256",
"Redis",

"Remember Me",
"new IP detection",
"security notice",
"security alert email",

"credential complexity",
"social sign-in",
"Google",
"Microsoft",

"GDPR",
"CCPA",
"SOC 2",


# PAYMENTS
"Stripe",
"Plaid",
"ACH",
"Credit Card",
"Debit Card",

"Idempotency Key",
"UUID",

"payment_intent.succeeded",
"payment_intent.payment_failed",
"payment_intent.processing",

"POST /api/v1/payments/initiate",

"gateway integration",
"payment method",
"payment orchestration",
"partial payment",

"Service Fee",
"processing fee",
"ledger mapping",
"Charge_ID",

"Rent",
"Utility",
"Late Fee",

"bank reconciliation",
"QuickBooks sync",
"NSF late fee",

"PaymentMethod API",
"link_token",


# DATABASE
"Tenant_Transactions",
"Property settings table",

"General_Ledger",
"Tax_Ledger",
"Financial_Summary_View",

"Persistent_Tokens",

"Maintenance_Request",
"Property_Listings",

"GL_Entries",

"user_id",
"property_id",
"fiscal_year",

"Issued_IP",
"Charge_ID",

# API_ENDPOINTS
"POST /api/auth/refresh",
"POST /api/v1/payments/initiate",
"POST /api/v1/work-orders",

"DELETE /v1/payment_methods/:id",

"/login",

"QuickBooks Online API v3",
"Zillow Partner API",
"HotPads Feed API",
"Apartments.com API",
"DocuSign API",
"Twilio API",

# CLOUD
"S3",
"IndexedDB",

"Lambda pipeline",
"CRON Job",

"OAuth authentication",

"encrypted storage",
"AES-256 encryption",

"webhook verification",
"secure archival",

"client-side compression",
"pre-signed URL",

"offline synchronization",
"local caching",
"batch synchronization",

# LISTINGS
"Hard-Mandatory",
"Soft-Mandatory",

"422 Unprocessable Entity",

"Draft",
"Ready",
"Published",
"Paused",
"Leased",
"Archived",

"ZIP+4",
"USPS ZIP validation",

"Current_Year",

"EXIF metadata",

"syndication",
"listing rejection",

"Monthly_Rent",
"Bedrooms",
"Bathrooms",
"Square_Footage",

"Pet_Policy",
"Available_Date"

# MAINTENANCE
"work order",
"maintenance request",
"media upload",

"SMS escalation",
"urgency level",

"Emergency",
"High",
"Standard",
"Low",

"Water Leak",
"Gas Smell",

"Twilio SMS alerts",

"SLA monitoring",
"state machine",

"contractor assignment",
"tenant sign-off",

"GPS validation",
"digital signature collection",

"proof of service",
"offline job caching",

"assignment visibility",
"work order prioritization",

"Vendor_ID",
"repair evidence",

"Base64 Encoded String",

# ACCOUNTING
"NOI",
"GOI",
"Vacancy_Loss",
"Operating_Expenses",

"General_Ledger",
"Tax_Ledger",

"Net Operating Income",

"QuickBooks Online Ledger",
"Financial Ledger",

"transaction mapping",
"reconciliation monitoring",

"audit trail",
"audit readiness",

"tax liability",
"expense benchmarking",

"Debt Service Coverage Ratio",

"Income",
"Expense",

"estimated_tax",
"actual_tax",

# LEASE
"DocuSign integration",
"embedded signing workflow",

"counter-signature",
"document completion verification",

"secure document delivery",

"audit logging",

"signing ceremony",

"completed PDF retrieval",

"identity verification",

"tamper-proof storage",

"paperless execution",

# ROLES
"Tenant",
"Owner",
"Manager",
"Admin",
"Contractor",

"Portfolio Manager",
"Finance Administrator",

"Property Manager",

"Security Officer",

"QA Lead",
"Engineering Lead",

"Product Owner",

# VALIDATION_ERRORS
"HTTP 500",
"422 Unprocessable Entity",

"validation error",

"duplicate submission",

"connection failure",

"integration instability",

"failed upload",

"API rejection",

"unsupported platform",

"unsupported version",

"null value",

"incomplete metadata",

"missing required fields",

"security violation"
]

def add_square_brackets(text):

    output = text

    # ---------- BRD TERMS ----------

    terms = sorted(
        PROTECTED_TERMS,
        key=len,
        reverse=True
    )

    for term in terms:

        pattern = re.compile(
            rf'(?<!\[)\b{re.escape(term)}\b(?!\])',
            flags=re.IGNORECASE
        )

        output = pattern.sub(
            f"[{term}]",
            output
        )

    # ---------- TEXTRANK ----------

    doc = nlp(output)

    for phrase in doc._.phrases[:5]:

        phrase_text = phrase.text.strip()

        if len(phrase_text.split()) < 3:
            continue

        pattern = re.compile(
            rf'(?<!\[)\b{re.escape(phrase_text)}\b(?!\])',
            flags=re.IGNORECASE
        )

        output = pattern.sub(
            f"[{phrase_text}]",
            output
        )

    return output
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

    before_but = [parts[0]]
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
    - Output ONE Tanglish line only.
    - Keep [] text EXACTLY unchanged.
    - Preserve all technical clauses; do not omit information.
    - Maintain polarity:
        positive → pannudhu / aagudhu
        negative ("fails to","does not","missing") → pannala / aagala / missing-aa irukku
    - "should"/"must" → pannanum or aaganum
    - "after clicking"/"upon clicking" → click pannina appuram
    - Keep technical words in English.
    - Do not add explanations or extra text.

    Example:

    English:
    Field displays raw ID instead of configured label

    Tanglish:
    Configured label-a display pannanum,
    aana raw ID-a display pannudu

    English phrase: {english_chunk}
    Tanglish:"""

    response = ollama.chat(
        model="translategemma_custom:latest",
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

def normalize_patterns(text):

    replacements = {

        r'\blacks\b': 'is missing',

         r'\blacks\b':'is missing',

        r'\bfails to\b':'does not',

        r'\bdoes not\b':'does not',

        r'\bupon clicking\b':'after clicking',

        r'\bwhen clicked\b':'after clicking',

        r'\bon click\b':'after clicking',

        r'\bafter login\b':'login aana appuram',
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.I)

    return text
# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
def should_chunk(text):

    # very long sentence
    if len(text) > 150:
        return True

    # multiple clauses / heavy complexity
    complex_markers = [
        ";", "whereas", "while", "however",
        "regardless of", "provided that",
        "depending on"
    ]

    if any(m in text.lower() for m in complex_markers):
        return True

    return False


def english_to_tanglish(english_text, verbose=True):
    english_text = normalize_patterns(english_text)
    # english_text = add_square_brackets(english_text)
    mode = detect_sentence_mode(english_text)
    if should_chunk(english_text):
        chunks = split_sentence(english_text)
    else:
        chunks = [(english_text, False)]

    if verbose:
        print(f"  → Mode detected : {mode}")
        print(f"  → Split into {len(chunks)} chunk(s):")
        for i, (c, after_but) in enumerate(chunks):
            print(f"     [{i+1}] {'[BUT] ' if after_but else ''}{c}")

    tanglish_parts = []
    for chunk, is_after_but in chunks:
        
        raw = tanglish_chunk(chunk, mode=mode)
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
        "Offline contractor update overwrites higher priority assignment changes after reconnection, conflicting with configured synchronization priority.",
        "The system fails to enforce the defined Charge_ID allocation priority for partial payments, leading to incorrect distribution of payments.",
        "The Year_Built validation rule incorrectly displays a Studio unit value as numeric zero instead of the configured 'Studio' label.",
        "The mobile action footer overlaps the progress section on smaller screens, causing the layout to be unreadable across supported devices.",
        "The MFA screen displays inconsistent masked delivery destination formatting, and the expected outcome is for verification details to follow the configured display formatting.",
        "The MFA verification endpoint POST /api/auth/mfa/verify returns a 403 Account locked response when the account is suspended, failing to align with the expected behavior of returning a 401 Unauthorized response."
    ]

    for sentence in test_sentences:
        print(f"\nEnglish  : {sentence}\n")
        result = english_to_tanglish(sentence)
        print(f"\nTanglish : {result}")
        print("=" * 70)