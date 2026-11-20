import re
from typing import List, Dict, Any, Tuple, Optional
from app.config import EXACT_ABSTENTION_MESSAGE

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules)",
    r"bypass\s+(all\s+)?(safety|rules|restrictions)",
    r"you\s+are\s+now\s+(a|an)?\s*(unrestricted|dan|jailbreak)",
    r"pretend\s+you\s+(are|can)\s+(not|unrestricted|browse|access)",
    r"forget\s+(all\s+)?(instructions|rules|context)",
    r"system\s*override",
    r"act\s+as\s+a\s+general\s+(purpose\s+)?(ai|chatbot|assistant)",
]

class Guardrails:
    """
    Enforces security, prompt-injection defense, citation validation, and strict source grounding.
    """

    @staticmethod
    def sanitize_user_query(query: str) -> Tuple[str, bool]:
        """
        Detects prompt injection attempts in user input.
        Returns sanitized query and whether an injection pattern was flagged.
        """
        flagged = False
        lower_q = query.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lower_q):
                flagged = True
                break
        return query.strip(), flagged

    @staticmethod
    def extract_citations(text: str) -> List[int]:
        """
        Extracts cited page numbers from text, matching formats like:
        [Page 10], [Pages 10, 25], (Page 10), Page 10, etc.
        """
        citations = set()
        matches = re.findall(r"(?:\[|\()?(?:Page|Pages|p\.)\s*(\d+(?:\s*(?:,|and|-)\s*\d+)*)(?:\]|\))?", text, re.IGNORECASE)
        for m in matches:
            nums = re.findall(r"\d+", m)
            for n in nums:
                citations.add(int(n))
        return sorted(list(citations))

    @staticmethod
    def validate_grounding(
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, str, List[int]]:
        """
        Validates that the response adheres strictly to retrieved passages:
        - If response is the exact abstention message, it is valid.
        - If response answers the question, it must cite valid retrieved pages.
        - Cited pages must actually exist in the retrieved passages.
        - Factual words/numbers in the response should match the retrieved passages.
        """
        cleaned_resp = response_text.strip()

        # Check if model abstained
        if EXACT_ABSTENTION_MESSAGE.strip().lower() in cleaned_resp.lower() or "couldn't find this information" in cleaned_resp.lower():
            return True, EXACT_ABSTENTION_MESSAGE, []

        retrieved_pages = {c["page_number"] for c in retrieved_chunks}
        citations = Guardrails.extract_citations(cleaned_resp)

        # Requirement 4: Cite the exact PDF page number for every answer
        if not citations:
            # If no citation was included in a factual claim, force abstention or failure
            return False, EXACT_ABSTENTION_MESSAGE, []

        # Check if any cited page is invalid (not in retrieved chunks)
        for page in citations:
            if page not in retrieved_pages:
                # Fabricated citation detected!
                return False, EXACT_ABSTENTION_MESSAGE, []

        return True, cleaned_resp, citations

    @staticmethod
    def is_lexically_in_scope(query: str, retrieved_chunks: List[Dict[str, Any]]) -> bool:
        """
        Checks whether query keywords have meaningful overlap with retrieved text.
        Helps eliminate obvious out-of-domain queries (e.g., 'capital of France') prior to LLM call.
        """
        # Common stop words to exclude
        stopwords = {
            "what", "is", "the", "of", "and", "in", "a", "an", "to", "for", "how", "why",
            "where", "when", "who", "which", "are", "do", "does", "did", "can", "could",
            "would", "should", "tell", "me", "about", "please", "give", "list", "show", "many",
            "much", "with", "from", "at", "by", "this", "that", "it", "its"
        }
        tokens = [t.lower() for t in re.findall(r"\b\w+\b", query) if t.lower() not in stopwords]
        if not tokens:
            return False

        all_retrieved_text = " ".join(c["text"].lower() for c in retrieved_chunks)
        matching_tokens = [t for t in tokens if t in all_retrieved_text]

        # If zero content words from query appear in any retrieved chunk, it's definitely out of scope
        return len(matching_tokens) > 0
