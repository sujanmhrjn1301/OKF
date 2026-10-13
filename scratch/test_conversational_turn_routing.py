"""Test script for conversational turn routing logic in chat_cli.py."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chat_cli import is_follow_up_heuristic, INTENT_CATEGORIES

print("1. Testing INTENT_CATEGORIES...")
assert "FOLLOW_UP" in INTENT_CATEGORIES
print("   FOLLOW_UP is registered in INTENT_CATEGORIES.")

print("\n2. Testing is_follow_up_heuristic on diverse prompts...")
queries_to_test = [
    # Should be follow-ups
    ("tell me in simple term, what kinds of charges will be applied to the person who carried weapon.", True),
    ("what do you think about this situation?", True),
    ("can you explain in simple words?", True),
    ("simplify this for me", True),
    ("make it simple", True),
    ("in layman terms please", True),
    ("summarize in 3 bullet points", True),
    ("translate to nepali", True),
    ("in nepali", True),
    ("draft an application based on this", True),
    ("what kinds of charges will be framed?", True),
    ("who will be charged?", True),
    ("can he be arrested?", True),
    ("why?", True),
    ("how come?", True),
    
    # Should be new queries (not follow-ups)
    ("What are the provisions relating to preventive detention under Article 23?", False),
    ("If police conducts search without warrant, what section applies?", False),
    ("What is the penalty for culpable homicide not amounting to murder?", False),
    ("How is bail granted under Section 67?", False),
]

for prompt, expected in queries_to_test:
    res = is_follow_up_heuristic(prompt)
    assert res == expected, f"Failed for '{prompt}': expected {expected}, got {res}"
    status = "FOLLOW-UP" if res else "NEW SEARCH"
    print(f"   [{status:10}] '{prompt[:65]}'")

print("\n3. All test assertions passed with 100% precision!")
