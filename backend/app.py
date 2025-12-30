from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import io
import re

try:
    import joblib
except Exception:
    joblib = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

stored_data = {}   # temporary storage
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

RAG_INDEX_DIR = os.environ.get("RAG_INDEX_DIR", os.path.join(os.path.dirname(__file__), "rag_index"))
RAG_INDEX_FILE = os.path.join(RAG_INDEX_DIR, "index.joblib")
RAG_DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))


def _ensure_index_dir():
    os.makedirs(RAG_INDEX_DIR, exist_ok=True)


def _save_rag_index(index_payload: dict):
    if joblib is None:
        return
    _ensure_index_dir()
    joblib.dump(index_payload, RAG_INDEX_FILE)


def _load_rag_index():
    if joblib is None:
        return None
    if not os.path.exists(RAG_INDEX_FILE):
        return None
    try:
        return joblib.load(RAG_INDEX_FILE)
    except Exception:
        return None


def _init_rag_from_disk():
    payload = _load_rag_index()
    if not payload:
        return
    stored_data["rag"] = payload
    if "filename" in payload:
        stored_data["filename"] = payload["filename"]
    if "text" in payload:
        stored_data["text"] = payload["text"]


_init_rag_from_disk()


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_frontend_assets(path):
    return send_from_directory(FRONTEND_DIR, path)

def extract_text_from_file(file):
    """Extract text from various file formats"""
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith(('.doc', '.docx')):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        # Try to decode as text
        try:
            return file.read().decode('utf-8')
        except:
            raise ValueError(f"Unsupported file format: {filename}")

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text += page_text + "\n"
    return text


def extract_pages_from_pdf(file):
    """Extract per-page text for better citations."""
    pdf_reader = PyPDF2.PdfReader(file)
    pages = []
    for idx, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append({"page": idx, "text": page_text})
    return pages

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = Document(io.BytesIO(file.read()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def preprocess_text(text):
    """Clean and preprocess text while preserving important information"""
    # Remove extra whitespace but keep newlines for paragraph detection
    text = re.sub(r'[ \t]+', ' ', text)
    # Keep more punctuation and special characters that might be important
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\"\']', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size=3):
    """Split text into chunks of sentences for better context"""
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if len(chunk.strip()) > 20:  # Only add meaningful chunks
            chunks.append(chunk)
    return chunks, sentences


def build_rag_chunks_from_text(text: str, filename: str, chunk_size_sentences: int = 3):
    text_processed = preprocess_text(text)
    chunks, _sentences = split_into_chunks(text_processed, chunk_size=chunk_size_sentences)
    out = []
    for i, chunk in enumerate(chunks):
        out.append({
            "chunk_id": f"c{i}",
            "text": chunk,
            "source": filename,
            "page": None,
        })
    return out


def build_rag_chunks_from_pdf_pages(pages, filename: str, chunk_size_sentences: int = 3):
    out = []
    chunk_counter = 0
    for p in pages:
        page_num = p.get("page")
        page_text = preprocess_text(p.get("text") or "")
        if not page_text.strip():
            continue
        chunks, _sentences = split_into_chunks(page_text, chunk_size=chunk_size_sentences)
        for chunk in chunks:
            out.append({
                "chunk_id": f"c{chunk_counter}",
                "text": chunk,
                "source": filename,
                "page": page_num,
            })
            chunk_counter += 1
    return out


def build_rag_index(chunks):
    """Build a persistent TF-IDF index over chunks."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts) if texts else None
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }


def retrieve_chunks(question: str, rag_payload: dict, top_k: int = RAG_DEFAULT_TOP_K):
    vectorizer = rag_payload.get("vectorizer")
    matrix = rag_payload.get("matrix")
    chunks = rag_payload.get("chunks") or []
    if vectorizer is None or matrix is None or not chunks:
        return []

    q = (question or "").strip()
    if not q:
        return []

    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_k = max(1, min(int(top_k), len(chunks)))
    top_indices = sims.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append({
            "chunk_id": c.get("chunk_id"),
            "source": c.get("source"),
            "page": c.get("page"),
            "score": float(sims[int(idx)]),
            "text": c.get("text", ""),
        })
    return results


def _format_context_for_llm(retrieved):
    lines = []
    for r in retrieved:
        label = r.get("source") or "document"
        if r.get("page") is not None:
            label = f"{label} (page {r.get('page')})"
        lines.append(f"[source: {label} | chunk: {r.get('chunk_id')} | score: {r.get('score'):.3f}]\n{r.get('text','').strip()}")
    return "\n\n".join(lines)


def generate_rag_answer(question: str, retrieved, history=None):
    """Generate answer grounded in retrieved chunks. Uses OpenAI if configured, else falls back to extractive."""
    context = _format_context_for_llm(retrieved)

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer ONLY using the provided sources. If the sources do not contain the answer, say you don't know. Provide a concise answer.",
            }
        ]

        if history:
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                    messages.append({"role": h["role"], "content": h["content"]})

        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nSources:\n{context}",
        })

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    # Fallback: extractive answer from retrieved chunks
    if not retrieved:
        return "I couldn't find relevant information in the uploaded document." 

    combined = "\n\n".join([r.get("text", "") for r in retrieved if r.get("text")])
    if combined.strip():
        return find_answer(question, combined)
    return "I couldn't find relevant information in the uploaded document."

def extract_key_terms(text, top_n=15):
    """Extract key terms and important keywords from text"""
    # Preprocess text
    text = preprocess_text(text)
    
    # Create TF-IDF vectorizer to find important terms
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for phrases
        max_features=1000
    )
    
    # Fit on the entire document
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores
    scores = tfidf_matrix.toarray()[0]
    
    # Get top terms
    top_indices = scores.argsort()[-top_n:][::-1]
    key_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
    
    return key_terms

def generate_summary(text, num_sentences=5):
    """Generate extractive summary with key terms"""
    # Extract key terms from original text first (before preprocessing)
    key_terms = extract_key_terms(text, top_n=15)
    
    # Preprocess text for summary generation
    text_processed = preprocess_text(text)
    
    # Split into sentences
    sentences = sent_tokenize(text_processed)
    
    if len(sentences) <= num_sentences:
        summary_text = text_processed
    else:
        # Remove very short sentences
        sentences = [s for s in sentences if len(s.split()) > 5]
        
        if len(sentences) <= num_sentences:
            summary_text = ' '.join(sentences)
        else:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF scores)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top N sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Maintain original order
            
            # Combine top sentences
            summary_sentences = [sentences[i] for i in top_indices]
            summary_text = ' '.join(summary_sentences)
    
    # Format summary with key terms
    if key_terms:
        terms_str = ", ".join(key_terms[:10])  # Show top 10 terms
        return f"{summary_text}\n\n📌 Key Terms: {terms_str}"
    
    return summary_text

def find_answer(question, text):
    """Find relevant answer using TF-IDF similarity with improved keyword matching"""
    # Preprocess text but keep original for answer extraction
    text_processed = preprocess_text(text)
    text_original = text  # Keep original for better answer quality
    
    if not text_processed or len(text_processed.strip()) < 10:
        return "I couldn't find relevant information in the document."
    
    # Extract keywords from question for better matching
    question_lower = question.lower()
    question_words = [w for w in question_lower.split() if len(w) > 2 and w not in stop_words]
    
    # Split into chunks (3 sentences each) and individual sentences
    chunks, sentences = split_into_chunks(text_processed, chunk_size=3)
    
    if not chunks or not sentences:
        return "I couldn't find relevant information in the document."
    
    # First, try keyword-based matching (more lenient)
    keyword_matches = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Count how many question keywords appear in chunk
        matches = sum(1 for word in question_words if word in chunk_lower)
        if matches > 0:
            keyword_matches.append((i, matches, chunk))
    
    # If we have keyword matches, use the best one
    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        best_match = keyword_matches[0]
        if best_match[1] >= len(question_words) * 0.3:  # At least 30% of keywords match
            return chunks[best_match[0]].strip()
    
    # Then try TF-IDF similarity (more lenient threshold)
    all_texts = [question_lower] + [chunk.lower() for chunk in chunks]
    
    try:
        # Create TF-IDF vectors with better parameters
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2),  # Use bigrams
            min_df=1,
            max_df=0.98  # More lenient
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarity between question and each chunk
        question_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(question_vector, chunk_vectors)[0]
        
        # Get top chunks (more lenient - lower threshold)
        top_indices = similarities.argsort()[-3:][::-1]  # Get top 3
        
        # Use more lenient threshold - return best match even if low
        best_idx = top_indices[0]
        max_similarity = similarities[best_idx]
        
        # Lower threshold to 0.05 for more lenient matching
        if max_similarity > 0.05 or len(keyword_matches) > 0:
            # Get the best matching chunk
            best_chunk = chunks[best_idx].strip()
            
            # If similarity is very low but we have keyword matches, use keyword match
            if max_similarity < 0.05 and keyword_matches:
                best_chunk = keyword_matches[0][2].strip()
            
            return best_chunk
        
        # If chunk similarity is low, try sentence-level matching
        filtered_sentences = [s for s in sentences if len(s.split()) > 3]
        
        if len(filtered_sentences) > 0:
            # Try keyword matching on sentences
            sentence_keyword_matches = []
            for i, sent in enumerate(filtered_sentences):
                sent_lower = sent.lower()
                matches = sum(1 for word in question_words if word in sent_lower)
                if matches > 0:
                    sentence_keyword_matches.append((i, matches, sent))
            
            if sentence_keyword_matches:
                sentence_keyword_matches.sort(key=lambda x: x[1], reverse=True)
                best_sent_match = sentence_keyword_matches[0]
                if best_sent_match[1] >= 1:  # At least 1 keyword match
                    return filtered_sentences[best_sent_match[0]].strip()
            
            # Try TF-IDF on sentences
            all_texts_sent = [question_lower] + [s.lower() for s in filtered_sentences]
            
            if len(all_texts_sent) > 1:
                vectorizer_sent = TfidfVectorizer(
                    stop_words='english', 
                    ngram_range=(1, 2),
                    min_df=1
                )
                tfidf_matrix_sent = vectorizer_sent.fit_transform(all_texts_sent)
                
                question_vector_sent = tfidf_matrix_sent[0:1]
                sentence_vectors = tfidf_matrix_sent[1:]
                
                similarities_sent = cosine_similarity(question_vector_sent, sentence_vectors)[0]
                
                # Get top sentences with very lenient threshold
                top_indices = similarities_sent.argsort()[-5:][::-1]
                valid_sentences = [
                    filtered_sentences[i] for i in top_indices 
                    if (similarities_sent[i] > 0.05 or any(w in filtered_sentences[i].lower() for w in question_words)) 
                    and i < len(filtered_sentences)
                ]
                
                if valid_sentences:
                    # Return up to 3 most relevant sentences
                    answer = ' '.join(valid_sentences[:3])
                    return answer.strip()
        
        # Last resort: return top chunk even if similarity is very low
        if len(chunks) > 0:
            return chunks[best_idx].strip()
        
        return "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about a different topic from the document."
        
    except Exception as e:
        # Fallback: try simple keyword search
        text_lower = text_original.lower()
        for word in question_words:
            if word in text_lower:
                # Find sentences containing the keyword
                for sent in sentences:
                    if word in sent.lower():
                        return sent.strip()
        return "I encountered an error processing your question. Please try again."

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Extract text from file (and keep PDF page info for citations)
        original_filename = file.filename
        lower_name = (original_filename or "").lower()

        pages = None
        text = None
        if lower_name.endswith('.pdf'):
            pages = extract_pages_from_pdf(file)
            text = "\n".join([(p.get("text") or "") for p in pages if (p.get("text") or "").strip()])
        else:
            text = extract_text_from_file(file)
        
        if not text or len(text.strip()) < 10:
            return jsonify({"error": "File appears to be empty or could not be processed"}), 400
        
        global stored_data
        stored_data["text"] = text
        stored_data["filename"] = original_filename

        # Build RAG chunks + index
        if pages is not None:
            chunks = build_rag_chunks_from_pdf_pages(pages, original_filename, chunk_size_sentences=3)
        else:
            chunks = build_rag_chunks_from_text(text, original_filename, chunk_size_sentences=3)

        rag_payload = build_rag_index(chunks)
        rag_payload["doc_id"] = str(uuid.uuid4())
        rag_payload["filename"] = original_filename
        rag_payload["text"] = text
        stored_data["rag"] = rag_payload
        _save_rag_index(rag_payload)
        
        return jsonify({
            "message": "File uploaded successfully!",
            "filename": file.filename,
            "char_count": len(text),
            "chunk_count": len(chunks)
        })
    
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route("/summary", methods=["GET"])
def summary():
    if "text" not in stored_data:
        return jsonify({"error": "No document uploaded yet!"}), 400
    
    try:
        text = stored_data["text"]
        
        # Generate intelligent summary
        summary_text = generate_summary(text, num_sentences=5)
        
        return jsonify({
            "summary": summary_text,
            "original_length": len(text),
            "summary_length": len(summary_text)
        })
    
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

def find_answer_with_context(question, text, history=None):
    """Find relevant answer using TF-IDF similarity with conversation context"""
    # Don't preprocess here - find_answer will do it
    if not text or len(text.strip()) < 10:
        return "I couldn't find relevant information in the document."
    
    # Enhance question with context from history for better understanding
    enhanced_question = question
    if history:
        # Extract recent user questions and answers for context
        recent_context = []
        for h in history[-4:]:  # Last 4 messages
            if h.get("role") == "user":
                recent_context.append(h["content"])
            elif h.get("role") == "assistant" and len(recent_context) > 0:
                # Add assistant response for context (but keep it short)
                assistant_text = h["content"][:100]  # First 100 chars
                recent_context.append(assistant_text)
        
        if recent_context:
            context = " ".join(recent_context[-2:])  # Last 2 items
            enhanced_question = f"{context} {question}"
    
    # Use the improved find_answer function but with enhanced question
    # First try with enhanced question, then fallback to original
    answer = find_answer(enhanced_question, text)
    
    # If enhanced question didn't work well, try with original question
    if "couldn't find" in answer.lower() and enhanced_question != question:
        answer = find_answer(question, text)
    
    # Return answer - we're more lenient now, so trust the matching
    return answer

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data["question"].strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        if "text" not in stored_data:
            return jsonify({"error": "Upload a document first!"}), 400
        
        text = stored_data["text"]
        
        # Find answer using NLP
        answer = find_answer(question, text)
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data["question"].strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        if "text" not in stored_data:
            return jsonify({"error": "Upload a document first! I need a document to answer your questions."}), 400

        text = stored_data["text"]
        history = data.get("history", [])

        rag_payload = stored_data.get("rag")
        if rag_payload is None:
            # Backward compatible fallback
            answer = find_answer_with_context(question, text, history)
            return jsonify({"answer": answer, "citations": []})

        top_k = data.get("top_k", RAG_DEFAULT_TOP_K)
        retrieved = retrieve_chunks(question, rag_payload, top_k=top_k)
        answer = generate_rag_answer(question, retrieved, history=history)

        citations = []
        for r in retrieved:
            excerpt = (r.get("text") or "").strip()
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "…"
            citations.append({
                "source": r.get("source"),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
                "score": r.get("score"),
                "excerpt": excerpt,
            })

        return jsonify({"answer": answer, "citations": citations})
    
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    _init_rag_from_disk()
    app.run(debug=True, port=5000)
