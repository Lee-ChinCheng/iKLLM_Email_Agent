import json, os, requests, smtplib
from router import route_user_question
from neo4j import GraphDatabase
from email.mime.text import MIMEText
from google import genai # pip install google-genai -U
from google.oauth2.service_account import Credentials # pip install google-auth -U
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  

# conda activate app_py311
# streamlit run main_stUI.py


# =========================
# user setting
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:latest" 

key_path = os.path.join(os.path.dirname(__file__), "your_project-123456.json")
credentials = Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
client = genai.Client(vertexai=True, project="your_project", location="us-central1", credentials=credentials)
GEMINI_MODEL = "gemini-2.5-flash"
 
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
SMTP_APP_PASS = os.getenv("SMTP_APP_PASS")
SMTP_SENDER = os.getenv("SMTP_SENDER")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))






def call_ollama(prompt: str) -> str:
    """
    Call Ollama generate API and return full text output.
    Handles streaming response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }
    r = requests.post(OLLAMA_URL, json=payload, stream=True)
    r.raise_for_status()
    output = []
    for line in r.iter_lines():
        if line:
            obj = json.loads(line.decode("utf-8"))
            output.append(obj.get("response", ""))

    return "".join(output).strip()


# =========================
# 1) User Question -> Cypher
# =========================
def question_to_cypher(question: str) -> str:
    """
    Convert user question into iKraph Cypher query using LLM.
    """
    prompt = f"""
You are an expert Neo4j Cypher generator for the iKraph medical knowledge graph.

Your job:
- Convert the user question into a valid Cypher query.
- Use UMLS node label: UMLS
- Use iKraph node label: iKraph
- Relationship mapping:
   (:UMLS)-[:iKraph2UMLS]-(:iKraph)
   (:iKraph)-[r]-(:iKraph)
- Return paths and citation info: r.pmids and r.pubmedCitations
- Limit results to 15.

IMPORTANT:
- Output ONLY Cypher query text.
- Do NOT explain anything.
- Do NOT use markdown.
- Assume user is asking about a medical concept in UMLS.

Example user question: "what is aspirin?"
Example Cypher:
MATCH path = (u:UMLS {{CUI: "C0004057"}})-[:iKraph2UMLS]-(i:iKraph)-[r]-(i2:iKraph)-[:iKraph2UMLS]-(u2:UMLS)
WHERE u <> u2
RETURN path, r.pmids, r.pubmedCitations
LIMIT 15

Now generate Cypher for this user question:
"{question}"
"""
    cypher = call_ollama(prompt)

    # Basic safety cleanup:
    cypher = cypher.replace("```", "").strip()
    return cypher


# =========================
# 2) Run Cypher on Neo4j
# =========================
def run_cypher(cypher_query: str):
    """
    Execute Cypher and return records as JSON-safe data.
    """
    with driver.session() as session:
        result = session.run(cypher_query)

        records = []
        for record in result:
            # Neo4j path objects are not JSON-serializable
            # We'll convert them into a readable format
            path = record.get("path")

            # Extract nodes + relationships from the path
            nodes = []
            rels = []

            for n in path.nodes:
                nodes.append({
                    "id": n.id,
                    "labels": list(n.labels),
                    "properties": dict(n)
                })

            for rel in path.relationships:
                rels.append({
                    "type": rel.type,
                    "start": rel.start_node.id,
                    "end": rel.end_node.id,
                    "properties": dict(rel)
                })

            records.append({
                "nodes": nodes,
                "relationships": rels,
                "pmids": record.get("r.pmids"),
                "pubmedCitations": record.get("r.pubmedCitations")
            })

        return records


# =========================
# 3) Results -> Natural Language Answer
# =========================
def summarize_results(question: str, graph_results):
    """
    Ask LLM to summarize iKraph query results for the user question.
    """
    prompt = f"""
You are a medical assistant summarizing results from the iKraph knowledge graph.

User question: "{question}"

You are given Neo4j graph query results in JSON format.
Explain the concept in natural language:
- Start with a short definition.
- Then list key gene/drug/disease relationships.
- Mention supporting citations if available.
- Use bullet points for readability.

Graph Results JSON:
{json.dumps(graph_results, indent=2)}

Now write the final answer for the user.
"""
    return call_ollama(prompt)


def send_email_smtp(to_email, subject, body):
 
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SMTP_SENDER, SMTP_APP_PASS)
        server.send_message(msg)
    print('email has been sent')



def rewrite_email_body(subject: str, content: str, tone: str = "professional") -> str:
    """
    Rewrite a given email content into a polished email format.
    
    :param subject: Email subject line
    :param content: The prewritten content that needs rewriting
    :param tone: Writing tone (default: professional)
    :return: Rewritten email body text
    """
    prompt = f"""
You are an assistant helping to rewrite email content.

Task:
- Rewrite the content into a {tone} email body.
- Keep the meaning and facts unchanged.
- Improve clarity, organization, and readability.
- Do NOT add new medical claims.
- Output email body only (no subject line).

Email Subject:
{subject}

Original Content:
{content}
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )

    return getattr(response, "text", "").strip() or "No output returned from model."


# -----------------------------
# Agent controller
# -----------------------------
def ai_email_agent(ikraph_answer: str, result: dict):

    if result["intent"] == "send_email":
        email_info = result["email"]

        # Safety check
        if not email_info["to"]:
            return "No recipient email detected."
        
        body = rewrite_email_body(result["email"]["subject"], ikraph_answer)
        #body = generate_email_body(email_info["body_prompt"])
        

        send_email_smtp(
            to_email=email_info["to"],
            subject=email_info["subject"] or "No Subject",
            body=body
        )
        #print(f"Email sent to {email_info['to']}")
        return body

    else:
        #print(f"No Email be sent")
        return body



def main():
      
        question = result["medical_question"]
        print(question)
       
        print("\n[1] Converting question to Cypher...")
        cypher = question_to_cypher(question)
        print("\nGenerated Cypher:\n", cypher)

        print("\n[2] Running Cypher on iKraph Neo4j...")
        results = run_cypher(cypher)
        print(f"Retrieved {len(results)} records from iKraph.")

        print("\n[3] Summarizing results with LLM...")
        answer = summarize_results(question, results)

        #print("\n======================")
        #print("Final Answer:")
        #print(answer)
        return  answer




# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="AI Email Agent",
    page_icon="✉️",
    layout="centered"
)

st.title("✉️ LLM + IKraph + Email Agent Demo ")
#st.caption("example:\n what is the capital city of Japan? please also send your response to this address, leeeo1211@gmail.com.")

# Input box (USER PROMPT ONLY)
user_prompt = st.text_area(
    "Your prompt",
    placeholder="Ask a question or instruct me to send an email...",
    height=140
)

col1, col2 = st.columns([1, 3])
with col1:
    send_btn = st.button("Send", type="primary")

with col2:
    st.markdown("Example:\n\n"
    "What is the application of Panadol? "
    "Please send the answer to IAN123@gmail.com \n\n"
    "where are the major international airports in Tokyo? please also send your response to "
    "IAN456@gapp.nthu.edu.tw, with recipient name, Ian")


# Response panel
if send_btn:
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = route_user_question(user_prompt)

                answer = main()
                

                st.subheader(f"Response")             
                st.write(answer)

                with st.expander("Email content (if Active)"):
                    emailresult = ai_email_agent(answer, result)
                    st.write(emailresult)

                # Optional debugging display
                with st.expander("Debug: extracted intent JSON"):
                    result_js = json.dumps(result, indent=2)
                    st.json(result_js)

            except requests.exceptions.RequestException as e:
                st.error(f"Ollama connection error: {e}")

            except Exception as e:
                st.error(f"Error: {e}")













