# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# import logging
# import smtplib, imaplib, email
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from contextlib import asynccontextmanager
# from typing import List
# import asyncio

# from langchain_community.document_loaders import BSHTMLLoader
# from pathlib import Path
# import glob

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings.base import Embeddings
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain.chains import RetrievalQA

# from dotenv import load_dotenv
# load_dotenv()
# from langchain.schema import Document
# from bs4 import BeautifulSoup



# class LocalSentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(texts, show_progress_bar=False).tolist()

#     def embed_query(self, text):
#         return self.model.encode(text).tolist()


# # ------------------ Logging ------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------ Global Variables ------------------
# rag_chain = None

# # ------------------ Email Config ------------------
# EMAIL_USER = os.getenv("EMAIL_USER", "your_email@gmail.com")
# EMAIL_PASS = os.getenv("EMAIL_PASS", "your_app_password")
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 465
# IMAP_SERVER = "imap.gmail.com"

# # ------------------ Filters ------------------
# ALLOWED_DOMAINS = ["byldgroup.com", "yourcompany.com"]
# REQUIRED_SUBJECT_KEYWORDS = ["question:", "query:"]
# REQUIRED_BODY_KEYWORDS = ["?"]

# # ------------------ RAG Pipeline Init ------------------
# def initialize_rag_pipeline():
#     try:
#         logger.info("Initializing RAG pipeline...")

#         folder_path = os.getenv(
#             "HTML_FOLDER",
#             r"C:\Users\Vineet Kumar\Downloads\Byld_url (2)"
#         )
#         logger.info(f"Loading from folder: {folder_path}")

#         documents = []

#         html_files = glob.glob(
#             os.path.join(folder_path, "**", "*.html"),
#             recursive=True
#         )

#         for file in html_files:
#             try:
#                 with open(file, "r", encoding="utf-8", errors="ignore") as f:
#                     html = f.read()

#                 soup = BeautifulSoup(html, "lxml")

#                 # Clean text
#                 text = soup.get_text(separator=" ", strip=True)

#                 if text:
#                     documents.append(
#                         Document(
#                             page_content=text,
#                             metadata={"source": file}
#                         )
#                     )

#             except Exception as e:
#                 logger.warning(f"Skipping file {file}: {e}")

#         if not documents:
#             raise RuntimeError("No valid HTML documents loaded")

#         logger.info(f"Loaded {len(documents)} HTML documents")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=150
#         )
#         splitted_docs = text_splitter.split_documents(documents)

#         logger.info(f"Total chunks created: {len(splitted_docs)}")

#         embeddings = LocalSentenceTransformerEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )

#         logger.info("Creating FAISS vector store...")
#         vectorstore = FAISS.from_documents(splitted_docs, embeddings)

#         logger.info("Initializing NVIDIA LLM...")
#         llm = ChatNVIDIA(
#             api_key=os.getenv("NVIDIA_API_KEY"),
#             model="meta/llama-3.1-8b-instruct"
#         )

#         global rag_chain
#         rag_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(),
#             return_source_documents=True
#         )

#         logger.info("RAG pipeline initialized successfully")
#         return True

#     except Exception as e:
#         logger.error(f"[FIX] RAG init failed: {e}", exc_info=True)
#         return False

# # ------------------ Email Helpers ------------------
# def send_email(to, subject, body):
#     try:
#         msg = MIMEMultipart()
#         msg["From"] = EMAIL_USER
#         msg["To"] = to
#         msg["Subject"] = subject
#         msg.attach(MIMEText(body, "plain"))

#         with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
#             server.login(EMAIL_USER, EMAIL_PASS)
#             server.sendmail(EMAIL_USER, to, msg.as_string())
#     except Exception as e:
#         logger.error(f"[FIX] Error sending email to {to}: {str(e)}", exc_info=True)
#         raise

# def fetch_emails(limit=5):
#     try:
#         mail = imaplib.IMAP4_SSL(IMAP_SERVER)
#         mail.login(EMAIL_USER, EMAIL_PASS)
#         mail.select("inbox")

#         status, messages = mail.search(None, "UNSEEN")
#         if status != "OK":
#             logger.warning("No unseen emails.")
#             return []

#         email_ids = messages[0].split()[-limit:]
#         emails = []
#         for e_id in email_ids:
#             _, msg_data = mail.fetch(e_id, "(RFC822)")
#             raw_email = msg_data[0][1]
#             msg = email.message_from_bytes(raw_email)

#             body = ""
#             if msg.is_multipart():
#                 for part in msg.walk():
#                     if part.get_content_type() == "text/plain":
#                         body = part.get_payload(decode=True).decode(errors="ignore")
#                         break
#             else:
#                 body = msg.get_payload(decode=True).decode(errors="ignore")

#             emails.append({
#                 "id": e_id.decode(),
#                 "from": msg["From"],
#                 "subject": msg["Subject"],
#                 "body": body.strip()
#             })
#         return emails
#     except Exception as e:
#         logger.error(f"[FIX] Error fetching emails: {str(e)}", exc_info=True)
#         raise

# def email_passes_filters(email_obj):
#     try:
#         sender = (email_obj["from"] or "").lower()
#         subject = (email_obj["subject"] or "").lower()
#         body = (email_obj["body"] or "").strip().lower()

#         if not any(sender.endswith("@" + d) for d in ALLOWED_DOMAINS):
#             logger.info(f"Skipped (sender domain not allowed): {sender}")
#             return False

#         if not any(keyword in subject for keyword in REQUIRED_SUBJECT_KEYWORDS):
#             logger.info(f"Skipped (subject missing keywords): {subject}")
#             return False

#         if not any(keyword in body for keyword in REQUIRED_BODY_KEYWORDS):
#             logger.info("Skipped (body missing required keywords)")
#             return False

#         return True
#     except Exception as e:
#         logger.error(f"[FIX] Error in email filters: {str(e)}", exc_info=True)
#         return False

# def process_incoming_emails():
#     if rag_chain is None:
#         raise RuntimeError("RAG pipeline not initialized")

#     try:
#         emails = fetch_emails(limit=5)
#         responses = []

#         for e in emails:
#             logger.info(f"Checking email from {e['from']} with subject {e['subject']}")

#             if not email_passes_filters(e):
#                 continue

#             logger.info("Email passed filters -> Processing...")

#             # [FIX] Offload blocking RAG and SMTP to executor for async endpoints if needed
#             loop = asyncio.get_event_loop()
#             result = loop.run_in_executor(None, rag_chain.invoke, {"query": e["body"]})
#             result = asyncio.run_coroutine_threadsafe(result, loop).result()

#             answer = result["result"]

#             reply_body = f"Your question:\n{e['body']}\n\nAnswer:\n{answer}"
#             send_email(to=e["from"], subject="Re: " + (e["subject"] or "Query"), body=reply_body)

#             responses.append({
#                 "from": e["from"],
#                 "subject": e["subject"],
#                 "answer": answer
#             })

#         return responses
#     except Exception as e:
#         logger.error(f"[FIX] Error processing incoming emails: {str(e)}", exc_info=True)
#         raise

# # ------------------ FastAPI App ------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Starting Email Agent API server...")
#     success = initialize_rag_pipeline()
#     if not success:
#         raise RuntimeError("RAG pipeline initialization failed")
#     yield
#     logger.info("Shutting down Email Agent API server...")

# app = FastAPI(
#     title="Email Q&A Agent",
#     description="Fetch emails, process via RAG, and auto-reply with answers",
#     version="1.0.0",
#     lifespan=lifespan
# )

# @app.get("/")
# async def root():
#     return {"message": "Email Q&A Agent is running", "status": "healthy"}

# @app.get("/process-emails")
# async def process_emails_endpoint():
#     try:
#         loop = asyncio.get_event_loop()
#         # [FIX] Run the possibly-blocking call in thread executor
#         responses = await loop.run_in_executor(None, process_incoming_emails)
#         # [FIX] To avoid UI hang, limit response size:
#         limit = min(3, len(responses))
#         return {"status": "success", "processed": responses[:limit]}
#     except Exception as e:
#         logger.error(f"Error in email processing endpoint: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/ask")
# async def ask_question(payload: dict):
#     if rag_chain is None:
#         raise HTTPException(status_code=500, detail="RAG not initialized")

#     result = rag_chain.invoke({"query": payload["question"]})
#     return {
#         "question": payload["question"],
#         "answer": result["result"]
#     }


# # ------------------ Run ------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("email_agent:app", host="0.0.0.0", port=8000, reload=True, log_level="info")



from fastapi import FastAPI, HTTPException
import os, logging, smtplib, imaplib, email, asyncio, glob, re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA

# ---------------- ENV ----------------
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

HTML_FOLDER = os.getenv(
    "HTML_FOLDER",
    r"C:\Users\Vineet Kumar\Downloads\Byld_url (2)"
)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email_agent")

# ---------------- GLOBAL ----------------
rag_chain = None

# ---------------- EMBEDDINGS ----------------
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# ---------------- HELPERS ----------------
def extract_email(sender):
    match = re.search(r'<(.+?)>', sender)
    return match.group(1) if match else sender

# ---------------- RAG INIT ----------------
def initialize_rag_pipeline():
    global rag_chain
    try:
        logger.info("Initializing RAG pipeline...")

        documents = []
        html_files = glob.glob(
            os.path.join(HTML_FOLDER, "**", "*.html"),
            recursive=True
        )

        for file in html_files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    documents.append(
                        Document(page_content=text, metadata={"source": file})
                    )

        if not documents:
            raise RuntimeError("No HTML documents found")

        logger.info(f"Loaded {len(documents)} HTML documents")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        logger.info(f"Total chunks created: {len(chunks)}")

        embeddings = LocalSentenceTransformerEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatNVIDIA(
            api_key=NVIDIA_API_KEY,
            model="meta/llama-3.1-8b-instruct"
        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )

        logger.info("RAG pipeline ready")
        return True

    except Exception as e:
        logger.error(f"RAG init failed: {e}", exc_info=True)
        return False

# ---------------- EMAIL ----------------
def send_email(to, subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to, msg.as_string())

def fetch_unseen_emails(limit=5):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    if status != "OK" or not messages[0]:
        return [], mail

    email_ids = messages[0].split()[-limit:]
    emails = []

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        emails.append({
            "id": eid,
            "from": msg["From"],
            "subject": msg["Subject"] or "",
            "body": body.strip()
        })

    return emails, mail

def process_unseen_emails():
    if not rag_chain:
        raise RuntimeError("RAG not initialized")

    emails, mail = fetch_unseen_emails()
    responses = []

    for e in emails:
        sender = extract_email(e["from"])
        question = e["body"]

        logger.info(f"Replying to NEW mail from {sender}")

        result = rag_chain.invoke({"query": question})
        answer = result["result"]

        reply = f"""
Hi,

You asked:
{question}

Answer:
{answer}

Regards,
BYLD AI Assistant
"""

        send_email(
            to=sender,
            subject="Re: " + e["subject"],
            body=reply
        )

        # IMPORTANT: mark as SEEN
        mail.store(e["id"], "+FLAGS", "\\Seen")

        responses.append({
            "from": sender,
            "answer": answer
        })

    mail.logout()
    return responses    

# ---------------- FASTAPI ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not initialize_rag_pipeline():
        raise RuntimeError("Startup failed")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health():
    return {"status": "healthy"}

@app.get("/process-emails")
def process_emails():
    return {
        "processed": process_unseen_emails()
    }

@app.post("/ask")
def ask(payload: dict):
    result = rag_chain.invoke({"query": payload["question"]})
    return {"answer": result["result"]}

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("email_agent:app", host="0.0.0.0", port=8000, reload=True)
