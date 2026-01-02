import json, re, requests
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# =========================
# user setting
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_S1 = "llama3.1:8b"
# fine tune temperature in call_llm function



# =========================
# 1) Schema
# =========================
@dataclass
class EmailBlock:
    to: Optional[str] = None
    subject: Optional[str] = None
   


@dataclass
class RouterOutput:
    intent: str
    medical_question: str
    email: EmailBlock


# =========================
# 2) Regex + helpers
# =========================
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

def is_valid_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.fullmatch(email.strip()))

def extract_email_regex(text: str) -> Optional[str]:
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else None

def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON robustly even if the model returns surrounding text.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # attempt to locate JSON object in response
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except json.JSONDecodeError:
                return None
        return None

def clean_medical_question(question: str) -> str:
    """
    Remove email/sending phrases while keeping the medical question.
    """
    if not question:
        return ""

    q = question.strip()

    # remove common trailing send/email instructions
    patterns = [
        r"(?i)\bplease\s+send.*$",
        r"(?i)\bsend\s+(the\s+)?answer.*$",
        r"(?i)\bemail\s+(the\s+)?answer.*$",
        r"(?i)\bsend\s+it\s+to.*$",
        r"(?i)\bto\s+[\w\.-]+@[\w\.-]+\.\w+.*$",
    ]

    for p in patterns:
        q = re.sub(p, "", q).strip()

    # If it contains a question mark, keep only up to the first '?'
    if "?" in q:
        q = q.split("?")[0].strip() + "?"

    return q


# =========================
# 3) Ollama LLM call (your streaming version)
# =========================
def call_llm(prompt: str) -> str:
   

    payload = {
        "model": OLLAMA_MODEL_S1,
        "prompt": prompt,
        "stream": True,
        # Optional: If your Ollama supports JSON mode, uncomment:
        "format": "json",
        # Optional: reduce randomness for structured output
        "options": {"temperature": 0.01}
    }

    r = requests.post(OLLAMA_URL, json=payload, stream=True)
    r.raise_for_status()

    output = []
    for line in r.iter_lines():
        if not line:
            continue
        obj = json.loads(line.decode("utf-8"))
        output.append(obj.get("response", ""))

    return "".join(output).strip()


# =========================
# 4) Router Prompt
# =========================
#2) medical_question MUST contain only the medical question part, rewritten cleanly (remove email sending phrases).
ROUTER_SYSTEM_PROMPT = """
You are an intent router for a medical chatbot pipeline.

Given a user message, produce ONLY valid JSON matching this schema:


{
  "intent": "send_email" | "answer_only",
  "medical_question": "string",
  "email": {
    "to": null | "string",
    "subject": null | "string"
  }
}

Rules:
1) intent="send_email" if the user asks to send/email the answer OR includes any email address.
2) medical_question MUST contain only the user question part without email sending phrases. For example, in "What is aspirin? please send the answer to Sam@gmail.com", the medical_question is "What is aspirin?".

3) If intent="answer_only", set email fields to null.
4) If intent="send_email":
   - "to" must be recipient email string.
   - "subject" should be short and relevant and can be inferred if missing.
Return JSON only. No markdown. No extra text.
""".strip()

def build_router_prompt(user_text: str) -> str:
    return f"{ROUTER_SYSTEM_PROMPT}\n\nUser message:\n{user_text}\n"


# =========================
# 5) Subject heuristic
# =========================
def guess_subject(medical_question: str) -> str:
    if not medical_question:
        return "Medical information"
    q = medical_question.strip().rstrip("?")
    if len(q) > 45:
        q = q[:42].rstrip() + "..."
    return f"{q} information"


# =========================
# 6) Main router function
# =========================
def route_user_question(user_text: str) -> Dict[str, Any]:
    """
    Runs the LLM router + postprocesses output into final strict schema dict.
    """

    prompt = build_router_prompt(user_text)
    llm_raw = call_llm(prompt)

    parsed = safe_json_loads(llm_raw)

    # -------------------------
    # Fallback if model didn't return JSON
    # -------------------------
    if parsed is None:
        fallback_email = extract_email_regex(user_text)
        intent = "send_email" if fallback_email else "answer_only"

        parsed = {
            "intent": intent,
            "medical_question": clean_medical_question(user_text),
            "email": {
                "to": fallback_email,
                "subject": None
                
            }
        }

    # -------------------------
    # Normalize / validate
    # -------------------------
    intent = parsed.get("intent", "answer_only")
    if intent not in ("send_email", "answer_only"):
        intent = "answer_only"

    medical_question = clean_medical_question(parsed.get("medical_question", ""))

    email_block = parsed.get("email") or {}
    to = email_block.get("to")
    subject = email_block.get("subject")


    # If answer_only => clear email section
    if intent == "answer_only":
        to, subject = None, None
    else:
        # intent=send_email
        if not to or not is_valid_email(to):
            to = extract_email_regex(user_text)

        if not subject:
            subject = guess_subject(medical_question)

      

    out = RouterOutput(
        intent=intent,
        medical_question=medical_question,
        email=EmailBlock(to=to, subject=subject)
    )

    return asdict(out)


