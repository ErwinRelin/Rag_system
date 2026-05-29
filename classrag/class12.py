import spacy
import json
import os
from spacy.matcher import PhraseMatcher
from spacy.language import Language

MODEL_PATH = "my_model"

# --- Register custom component ---
@Language.factory("keyword_matcher")
def create_keyword_matcher(nlp, name):
    return KeywordMatcher(nlp)

class KeywordMatcher:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.keyword_groups = {}

    # --- Add a whole group ---
    def add_keywords(self, label: str, keywords: list):
        if label not in self.keyword_groups:
            self.keyword_groups[label] = []
        # Only add keywords that don't already exist
        new_keywords = [kw for kw in keywords if kw not in self.keyword_groups[label]]
        if new_keywords:
            self.keyword_groups[label].extend(new_keywords)
            patterns = [self.nlp.make_doc(kw) for kw in new_keywords]
            self.matcher.add(label, patterns)
            print(f"Added {len(new_keywords)} keywords to '{label}'")
        else:
            print(f"No new keywords to add to '{label}'")

    # --- Add a single keyword ---
    def add_keyword(self, label: str, keyword: str):
        self.add_keywords(label, [keyword])

    # --- Remove a single keyword ---
    def remove_keyword(self, label: str, keyword: str):
        if label not in self.keyword_groups:
            print(f"Label '{label}' not found")
            return
        if keyword not in self.keyword_groups[label]:
            print(f"Keyword '{keyword}' not found in '{label}'")
            return
        # Remove from group and rebuild matcher for that label
        self.keyword_groups[label].remove(keyword)
        self.matcher.remove(label)
        if self.keyword_groups[label]:
            patterns = [self.nlp.make_doc(kw) for kw in self.keyword_groups[label]]
            self.matcher.add(label, patterns)
        print(f"Removed '{keyword}' from '{label}'")

    # --- Remove an entire label group ---
    def remove_group(self, label: str):
        if label not in self.keyword_groups:
            print(f"Label '{label}' not found")
            return
        del self.keyword_groups[label]
        self.matcher.remove(label)
        print(f"Removed group '{label}'")

    # --- List all keywords ---
    def list_keywords(self):
        if not self.keyword_groups:
            print("No keywords loaded.")
            return
        for label, keywords in self.keyword_groups.items():
            print(f"\n{label}:")
            for kw in keywords:
                print(f"  - {kw}")

    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            print(f"{label} → {doc[start:end].text}")
        return doc

    # --- Called by nlp.to_disk() ---
    def to_disk(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "keywords.json"), "w") as f:
            json.dump(self.keyword_groups, f, indent=2)

    # --- Called by nlp.from_disk() ---
    def from_disk(self, path, **kwargs):
        keywords_path = os.path.join(path, "keywords.json")
        if os.path.exists(keywords_path):
            with open(keywords_path) as f:
                keyword_groups = json.load(f)
            for label, keywords in keyword_groups.items():
                self.add_keywords(label, keywords)
        return self


# --- First time setup ---
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("keyword_matcher", last=True)

km = nlp.get_pipe("keyword_matcher")
km.add_keywords("AI_TERMS", ["machine learning", "deep learning", "neural network"])
km.add_keywords("MATH_TERMS", ["linear algebra", "calculus"])

nlp.to_disk(MODEL_PATH)
print(f"\nModel saved to ./{MODEL_PATH}")

# --- Load and update ---
nlp2 = spacy.load(MODEL_PATH)
km2 = nlp2.get_pipe("keyword_matcher")

print("\n--- Before update ---")
km2.list_keywords()

# Add a single keyword
km2.add_keyword("AI_TERMS", "reinforcement learning")

# Add a new group
km2.add_keywords("SCIENCE_TERMS", ["quantum physics", "thermodynamics"])

# Remove a single keyword
km2.remove_keyword("MATH_TERMS", "calculus")

# Remove an entire group
km2.remove_group("MATH_TERMS")

print("\n--- After update ---")
km2.list_keywords()

# Save again to persist the updates
nlp2.to_disk(MODEL_PATH)
print(f"\nUpdated model saved to ./{MODEL_PATH}")

# --- Test matching ---
print("\n--- Match results ---")
doc = nlp2("I study deep learning and reinforcement learning.")