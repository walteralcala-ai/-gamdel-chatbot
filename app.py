import os
import shutil
import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Usar Google Gemini en lugar de OpenAI
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY no está configurada en .env")

import google.generativeai as genai
import pdfplumber

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# ============= CONFIGURACIÓN =============
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)

app = FastAPI(title="GAMDEL RAG MVP - v5.1 (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_CACHE = {}
EMBEDDINGS_CACHE = {}
VECTORIZERS_CACHE = {}

# ============= EXTRACCIÓN DE TEXTO =============
def extract_text_from_pdf(file_path: str):
    """Extrae texto de PDF usando pdfplumber"""
    text = ""
    page_count = 0
    
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Página {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    pass
    except Exception as e:
        pass
    
    return text, page_count

def load_documents_from_disk():
    """Carga documentos del disco en memoria"""
    print("🔄 Cargando documentos del disco...")
    if not DATA_DIR.exists():
        print("⚠️ Directorio data no existe")
        return
    
    for tenant_dir in DATA_DIR.iterdir():
        if not tenant_dir.is_dir():
            continue
            
        tenant = tenant_dir.name
        DOCUMENTS_CACHE[tenant] = {}
        
        # Cargar archivos .txt primero
        txt_files = sorted(tenant_dir.glob("*.txt"))
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                if text_content.strip():
                    pdf_name = txt_file.stem + ".pdf"
                    DOCUMENTS_CACHE[tenant][pdf_name] = text_content
                    print(f"  ✅ {pdf_name}: {len(text_content)} chars")
            except Exception as e:
                print(f"  ❌ Error en {txt_file.name}: {e}")
        
        # Cargar PDFs
        pdf_files = sorted(tenant_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            pdf_name = pdf_file.name
            
            if pdf_name in DOCUMENTS_CACHE[tenant]:
                continue
            
            try:
                print(f"  📖 Procesando {pdf_file.name}...")
                text_content, page_count = extract_text_from_pdf(str(pdf_file))
                
                if text_content.strip():
                    DOCUMENTS_CACHE[tenant][pdf_file.name] = text_content
                    print(f"  ✅ {pdf_file.name}: {len(text_content)} chars")
            except Exception as e:
                print(f"  ❌ Error en {pdf_file.name}: {e}")
        
        if DOCUMENTS_CACHE[tenant]:
            create_embeddings(tenant)
            print(f"✅ Tenant '{tenant}': {len(DOCUMENTS_CACHE[tenant])} documentos cargados")

@app.on_event("startup")
async def startup_event():
    """Inicia el servidor"""
    print("🚀 Iniciando GAMDEL Chatbot v5.1 (con Google Gemini)...")
    print("✅ Servidor iniciado. Los documentos se cargarán bajo demanda.")

# ============= BASE DE DATOS =============
def init_db(tenant: str):
    """Inicializa la base de datos para un tenant"""
    db_path = DB_DIR / f"{tenant}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_question TEXT,
            assistant_response TEXT,
            sources TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            upload_date TEXT,
            file_size INTEGER,
            page_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def save_document_metadata(tenant: str, filename: str, file_size: int, page_count: int):
    """Guarda metadatos del documento en la BD"""
    db_path = DB_DIR / f"{tenant}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO documents (filename, upload_date, file_size, page_count)
        VALUES (?, ?, ?, ?)
    ''', (filename, datetime.now().isoformat(), file_size, page_count))
    
    conn.commit()
    conn.close()

def get_documents(tenant: str):
    """Obtiene lista de documentos de la BD"""
    db_path = DB_DIR / f"{tenant}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT filename, upload_date, file_size, page_count FROM documents ORDER BY upload_date DESC')
    docs = cursor.fetchall()
    conn.close()
    
    return docs

def save_conversation(tenant: str, question: str, response: str, sources: list):
    """Guarda la conversación en la BD"""
    db_path = DB_DIR / f"{tenant}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO conversations (timestamp, user_question, assistant_response, sources)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), question, response, json.dumps(sources)))
    
    conn.commit()
    conn.close()

def get_conversation_history(tenant: str, limit: int = 10):
    """Obtiene el historial de conversaciones"""
    db_path = DB_DIR / f"{tenant}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, user_question, assistant_response, sources 
        FROM conversations 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    history = cursor.fetchall()
    conn.close()
    
    return history

# ============= BÚSQUEDA Y EMBEDDINGS =============
def create_embeddings(tenant: str):
    """Crea embeddings TF-IDF para los documentos"""
    if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
        return None
    
    texts = list(DOCUMENTS_CACHE[tenant].values())
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    VECTORIZERS_CACHE[tenant] = vectorizer
    EMBEDDINGS_CACHE[tenant] = embeddings
    
    return vectorizer, embeddings

def search_relevant_documents(tenant: str, query: str, top_k: int = 1):
    """Busca documentos - PRIMERO por código, LUEGO por nombre, LUEGO por contenido"""
    if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
        return []
    
    doc_names = list(DOCUMENTS_CACHE[tenant].keys())
    query_lower = query.lower()
    
    # PASO 1: Buscar por código (ej: GAM-SIG-PR-021, DESPA-PG, G_003_2026)
    code_pattern = r'(GAM-SIG-PR-\d+|DESPA-PG-\d+|G_\d{3}_\d{4}|[A-Z]+-[A-Z]+-\d+)'
    code_matches = re.findall(code_pattern, query, re.IGNORECASE)
    
    if code_matches:
        for code in code_matches:
            for doc_name in doc_names:
                if code.lower() in doc_name.lower():
                    print(f"✅ Encontrado por código '{code}': {doc_name}")
                    return [doc_name]
    
    # PASO 2: Buscar por nombre de documento exacto o parcial
    name_matches = []
    for doc_name in doc_names:
        doc_name_lower = doc_name.lower()
        if doc_name_lower in query_lower or query_lower in doc_name_lower:
            name_matches.append(doc_name)
    
    if name_matches:
        print(f"✅ Encontrado por nombre: {name_matches[0]}")
        return [name_matches[0]]
    
    # PASO 3: Buscar por palabras clave en el nombre
    query_words = set(query_lower.split())
    word_matches = []
    for doc_name in doc_names:
        doc_words = set(doc_name.lower().split())
        common_words = query_words & doc_words
        if len(common_words) >= 2:
            word_matches.append((doc_name, len(common_words)))
    
    if word_matches:
        word_matches.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ Encontrado por palabras clave: {word_matches[0][0]}")
        return [word_matches[0][0]]
    
    # PASO 4: Búsqueda por contenido usando TF-IDF
    if tenant not in VECTORIZERS_CACHE or tenant not in EMBEDDINGS_CACHE:
        return [doc_names[0]] if doc_names else []
    
    vectorizer = VECTORIZERS_CACHE[tenant]
    embeddings = EMBEDDINGS_CACHE[tenant]
    
    try:
        query_embedding = vectorizer.transform([query]).toarray()
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        result = [doc_names[i] for i in top_indices if i < len(doc_names)]
        if result:
            print(f"✅ Encontrado por contenido: {result[0]}")
        return result
    except Exception as e:
        print(f"Error en búsqueda: {e}")
        return [doc_names[0]] if doc_names else []


def extract_version_and_date(filename: str) -> dict:
    """Extrae versión y fecha del nombre del documento"""
    metadata = {
        "version": None,
        "date": None
    }
    
    # Buscar versión (v1.0, v2.1, etc)
    version_patterns = [
        r'v(\d+\.\d+)',  # v1.0, v2.1
        r'v(\d+)',       # v1, v2
        r'version\s*(\d+\.\d+)',  # version 1.0
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            metadata["version"] = match.group(1) if match.lastindex else match.group(0)
            break
    
    # Buscar fecha (YYYY-MM-DD, DD/MM/YYYY, etc)
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # 2026-01-15
        r'(\d{2}/\d{2}/\d{4})',  # 15/01/2026
        r'(\d{1,2}\.\d{1,2}\.\d{4})',  # 15.01.2026
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            metadata["date"] = match.group(1)
            break
    
    return metadata

def is_document_table_request(question: str) -> bool:
    """Detecta si se pide generar una tabla/cuadro de documentos"""
    q_lower = question.lower()
    keywords = [
        'prepara', 'genera', 'crea', 'haz', 'haga', 'hacer',
        'cuadro', 'tabla', 'lista', 'reporte'
    ]
    doc_keywords = [
        'documento', 'documentos', 'archivo', 'archivos',
        'subido', 'cargado', 'pdf'
    ]
    
    has_action = any(kw in q_lower for kw in keywords)
    has_doc = any(kw in q_lower for kw in doc_keywords)
    
    return has_action and has_doc

def generate_documents_table(tenant: str, metadata_requested: list = None) -> str:
    """Genera una tabla con información de los documentos"""
    if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
        return "No hay documentos cargados para este cliente."
    
    doc_names = sorted(DOCUMENTS_CACHE[tenant].keys())
    
    # Crear tabla en formato texto
    table = "📊 CUADRO DE DOCUMENTOS SUBIDOS\n"
    table += "=" * 100 + "\n"
    table += f"{'Nº':<3} {'Documento':<60} {'Versión':<12} {'Fecha':<15}\n"
    table += "-" * 100 + "\n"
    
    for idx, doc_name in enumerate(doc_names, 1):
        metadata = extract_version_and_date(doc_name)
        version = metadata.get('version') or 'N/A'
        date = metadata.get('date') or 'N/A'
        
        # Truncar nombre si es muy largo
        doc_display = doc_name[:57] + "..." if len(doc_name) > 60 else doc_name
        
        table += f"{idx:<3} {doc_display:<60} {version:<12} {date:<15}\n"
    
    table += "=" * 100 + "\n"
    table += f"Total de documentos: {len(doc_names)}\n"
    
    return table

def is_meta_question(question: str) -> bool:
    """Detecta si la pregunta es SOLO sobre el sistema (no sobre documentos)"""
    q_lower = question.lower()
    
    # Palabras clave que indican pregunta sobre el sistema
    system_keywords = [
        'cuántos documentos', 'cuantos documentos',
        'cuántas preguntas', 'cuantas preguntas',
        'última pregunta', 'ultima pregunta',
        'historial', 'estadísticas', 'estadisticas',
        'tamaño total', 'tamaño de',
        'información del sistema', 'informacion del sistema',
        'cuántas páginas', 'cuantas paginas',
    ]
    
    # Palabras clave que indican pregunta sobre CONTENIDO de documentos
    content_keywords = [
        'prepara', 'genera', 'crea', 'haz', 'haga', 'hacer',
        'cuadro', 'tabla', 'lista', 'resumen', 'reporte',
        'indicando', 'mostrando', 'con', 'que incluya'
    ]
    
    # Si tiene palabras de contenido, NO es meta-pregunta
    if any(keyword in q_lower for keyword in content_keywords):
        return False
    
    # Si tiene palabras del sistema, SÍ es meta-pregunta
    return any(keyword in q_lower for keyword in system_keywords)

def get_system_info(tenant: str) -> str:
    """Obtiene información del sistema"""
    if tenant not in DOCUMENTS_CACHE:
        return "No hay documentos cargados para este cliente."
    
    docs = DOCUMENTS_CACHE[tenant]
    if not docs:
        return "No hay documentos cargados para este cliente."
    
    total_chars = sum(len(content) for content in docs.values())
    total_pages = len(docs)
    doc_list = "\n".join(f"- {name}" for name in sorted(docs.keys()))
    
    return f"""📊 INFORMACIÓN DEL SISTEMA

**Documentos Cargados:** {len(docs)}
**Caracteres Totales:** {total_chars:,}
**Páginas Procesadas:** {total_pages}

**Documentos:**
{doc_list}"""

def check_hallucination(answer: str, doc_name: str, doc_content: str) -> bool:
    """Verifica si la respuesta contiene alucinaciones"""
    if "no encontr" in answer.lower():
        return False
    
    # Buscar referencias a otros documentos
    other_doc_patterns = [
        r'GAM-SIG-PR-\d+',
        r'DESPA-PG',
        r'G_\d{3}_\d{4}',
        r'[\w\s]*\.pdf'
    ]
    
    for pattern in other_doc_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for match in matches:
            if match not in doc_name and match not in doc_content:
                return True
    
    return False

# ============= HTML INTERFACE =============
HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <title>GAMDEL Chatbot</title>
</head>
<body class="bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen">
  <div class="max-w-7xl mx-auto p-4">
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-3xl font-bold text-white">GAMDEL</h1>
        <p class="text-slate-400 text-sm">Chat Inteligente con Documentos</p>
      </div>
      <span class="bg-blue-600 text-white px-3 py-1 rounded-full text-xs font-semibold">v5.1</span>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
      <!-- Panel Izquierdo: Gestión -->
      <div class="lg:col-span-1">
        <div class="bg-white rounded-xl shadow-lg p-4 mb-4 sticky top-4">
          <h2 class="text-lg font-semibold mb-3">📁 Gestión</h2>
          
          <div class="mb-3">
            <label class="text-sm font-medium block mb-2">Cliente/Tenant</label>
            <input id="tenant" class="w-full border rounded-lg p-2 text-sm" value="demo"/>
          </div>

          <div class="mb-3">
            <label class="text-sm font-medium block mb-2">Subir PDF(s)</label>
            <input id="file" type="file" accept="application/pdf" multiple class="w-full border rounded-lg p-2 text-sm"/>
            <p class="text-xs text-slate-500 mt-1">Múltiples archivos</p>
          </div>

          <button id="uploadBtn" class="w-full bg-blue-600 text-white rounded-lg px-3 py-2 text-sm font-semibold hover:bg-blue-700">
            Subir e indexar
          </button>

          <div id="status" class="text-xs text-slate-600 mt-2"></div>

          <hr class="my-4"/>

          <button id="historyBtn" class="w-full bg-slate-600 text-white rounded-lg px-3 py-2 text-sm font-semibold hover:bg-slate-700">
            Ver Historial
          </button>
          
          <hr class="my-4"/>
          
          <button id="refreshBtn" class="w-full bg-green-600 text-white rounded-lg px-3 py-2 text-sm font-semibold hover:bg-green-700">
            Recargar Documentos
          </button>
          
          <hr class="my-4"/>
          
          <button id="deleteAllBtn" class="w-full bg-red-600 text-white rounded-lg px-3 py-2 text-sm font-semibold hover:bg-red-700">
            Eliminar Todos
          </button>
        </div>
      </div>

      <!-- Panel Central: Chat -->
      <div class="lg:col-span-2">
        <div class="bg-white rounded-xl shadow-lg p-4 h-full flex flex-col">
          <h2 class="text-lg font-semibold mb-3">💬 Chat</h2>
          
          <div class="flex gap-2 mb-4">
            <input id="q" class="flex-1 border rounded-lg p-2 text-sm" placeholder="Pregunta sobre tus documentos... (Enter para enviar)"/>
            <button id="askBtn" class="bg-blue-600 text-white rounded-lg px-4 py-2 text-sm font-semibold hover:bg-blue-700">
              Enviar
            </button>
          </div>
          
          <div id="messages" class="flex-1 overflow-y-auto space-y-3 mb-4 bg-slate-50 rounded-lg p-3"></div>
        </div>
      </div>

      <!-- Panel Derecho: Documentos -->
      <div class="lg:col-span-1">
        <div class="bg-white rounded-xl shadow-lg p-4 sticky top-4">
          <h2 class="text-lg font-semibold mb-3">📄 Documentos</h2>
          <div id="docList" class="text-sm text-slate-600">Sin documentos cargados</div>
        </div>
      </div>
    </div>
  </div>

  <script>
const qInput = document.getElementById('q');
const askBtn = document.getElementById('askBtn');
const uploadBtn = document.getElementById('uploadBtn');
const messagesDiv = document.getElementById('messages');
const statusEl = document.getElementById('status');
const docListDiv = document.getElementById('docList');

function addMsg(role, text) {
  const msg = document.createElement('div');
  msg.className = role === 'user' 
    ? 'bg-blue-100 text-blue-900 p-3 rounded-lg ml-8' 
    : 'bg-slate-100 text-slate-900 p-3 rounded-lg mr-8';
  msg.innerHTML = text.replace(/\\n/g, '<br>');
  messagesDiv.appendChild(msg);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function loadDocuments() {
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) return;
  
  try {
    const res = await fetch(`/documents?tenant=${tenant}`);
    const data = await res.json();
    
    if (data.ok && data.documents.length > 0) {
      docListDiv.innerHTML = data.documents
        .map((doc, i) => `<div class="flex justify-between items-center py-2 border-b">
          <span>${doc[0]}</span>
          <button onclick="deleteDocument('${doc[0]}')" class="text-red-600 text-xs hover:text-red-800">✕</button>
        </div>`)
        .join('');
    } else {
      docListDiv.innerHTML = 'Sin documentos cargados';
    }
  } catch (err) {
    docListDiv.innerHTML = 'Error cargando documentos';
  }
}

uploadBtn.onclick = async () => {
  const files = document.getElementById('file').files;
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) return alert("Escribe tenant");
  if (!files.length) return alert("Selecciona archivos");

  statusEl.textContent = "Subiendo...";
  const fd = new FormData();
  fd.append('tenant', tenant);
  for (let f of files) fd.append('files', f);

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    statusEl.textContent = data.ok ? "✅ Subido" : "❌ Error: " + data.error;
    if (data.ok) {
      document.getElementById('file').value = '';
      setTimeout(loadDocuments, 1000);
    }
  } catch (err) {
    statusEl.textContent = "❌ Error: " + err.message;
  }
};

async function sendMessage() {
  const tenant = document.getElementById('tenant').value.trim();
  const q = qInput.value.trim();
  if (!tenant) return alert("Escribe tenant");
  if (!q) return;

  addMsg("user", q);
  qInput.value = "";

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant, q })
    });
    const data = await res.json();
    if (!data.ok) return addMsg("assistant", "❌ Error: " + (data.error || "desconocido"));

    let answer = data.answer || "";
    if (data.sources?.length && data.sources[0] !== "SISTEMA") {
      answer += "\\n\\n📄 Fuente: " + data.sources[0];
    }
    addMsg("assistant", answer);
  } catch (err) {
    addMsg("assistant", "❌ Error: " + err.message);
  }
}

askBtn.onclick = sendMessage;

qInput.onkeypress = (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
};

document.getElementById('refreshBtn').onclick = async () => {
  statusEl.textContent = "Recargando documentos...";
  await loadDocuments();
  statusEl.textContent = "✅ Documentos recargados";
  setTimeout(() => { statusEl.textContent = ""; }, 2000);
};

document.getElementById('historyBtn').onclick = async () => {
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) return alert("Escribe tenant");

  try {
    const res = await fetch(`/history?tenant=${tenant}`);
    const data = await res.json();
    
    if (data.ok && data.history.length > 0) {
      const historyText = data.history.map(h => {
        const q = h[1];
        const r = h[2].substring(0, 100);
        return `P: ${q}\\nR: ${r}...`;
      }).join('\\n\\n---\\n\\n');
      addMsg("assistant", "📋 Historial:\\n\\n" + historyText);
    } else {
      addMsg("assistant", "Sin historial");
    }
  } catch (err) {
    addMsg("assistant", "Error cargando historial");
  }
};

async function deleteDocument(filename) {
  const tenant = document.getElementById('tenant').value.trim();
  if (!confirm(`¿Eliminar "${filename}"?`)) return;
  
  const fd = new FormData();
  fd.append('tenant', tenant);
  fd.append('filename', filename);
  
  try {
    const res = await fetch('/delete-document', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      statusEl.textContent = `✅ ${filename} eliminado`;
      setTimeout(loadDocuments, 500);
    } else {
      alert('Error: ' + data.error);
    }
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

document.getElementById('deleteAllBtn').onclick = async () => {
  const tenant = document.getElementById('tenant').value.trim();
  if (!confirm('¿Eliminar TODOS los documentos? Esta acción no se puede deshacer.')) return;
  
  const fd = new FormData();
  fd.append('tenant', tenant);
  
  try {
    const res = await fetch('/delete-all-documents', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      statusEl.textContent = "✅ Todos los documentos eliminados";
      setTimeout(loadDocuments, 500);
    } else {
      alert('Error: ' + data.error);
    }
  } catch (err) {
    alert('Error: ' + err.message);
  }
};

loadDocuments();
  </script>
</body>
</html>
"""

# ============= ENDPOINTS =============
@app.get("/")
async def root():
    return HTMLResponse(HTML)

@app.get("/documents")
async def get_docs(tenant: str):
    try:
        if tenant not in DOCUMENTS_CACHE:
            load_documents_from_disk()
        
        if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
            return {"ok": True, "documents": []}
        
        docs = DOCUMENTS_CACHE[tenant]
        doc_list = [[name, len(content)] for name, content in docs.items()]
        return {"ok": True, "documents": doc_list}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/upload")
async def upload(tenant: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        tenant = (tenant or "").strip()
        if not tenant:
            return JSONResponse({"ok": False, "error": "tenant requerido"}, status_code=400)
        
        init_db(tenant)
        tenant_dir = DATA_DIR / tenant
        tenant_dir.mkdir(exist_ok=True)
        
        if tenant not in DOCUMENTS_CACHE:
            DOCUMENTS_CACHE[tenant] = {}
        
        for file in files:
            if not file.filename:
                continue
                
            out_path = tenant_dir / file.filename
            file_content = await file.read()
            
            with open(out_path, 'wb') as f:
                f.write(file_content)
            
            text_content, page_count = extract_text_from_pdf(str(out_path))
            
            if text_content.strip():
                DOCUMENTS_CACHE[tenant][file.filename] = text_content
                
                txt_path = tenant_dir / (out_path.stem + ".txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            
            file_size = out_path.stat().st_size
            save_document_metadata(tenant, file.filename, file_size, page_count)

            create_embeddings(tenant)
            print(f"✅ Documento '{file.filename}' cargado y re-indexado")

        return {"ok": True, "files": [p.name for p in tenant_dir.iterdir()]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/ask")
async def ask(payload: dict):
    try:
        tenant = (payload.get("tenant") or "").strip()
        q = (payload.get("q") or "").strip()
        if not tenant or not q:
            return JSONResponse({"ok": False, "error": "tenant y q requeridos"}, status_code=400)

        # Detectar solicitudes de tabla de documentos
        if is_document_table_request(q):
            table = generate_documents_table(tenant)
            save_conversation(tenant, q, table, ["SISTEMA"])
            return {"ok": True, "answer": table, "sources": ["SISTEMA"]}
        
        # Detectar meta-preguntas
        if is_meta_question(q):
            system_info = get_system_info(tenant)
            save_conversation(tenant, q, system_info, ["SISTEMA"])
            return {"ok": True, "answer": system_info, "sources": ["SISTEMA"]}

        if tenant not in DOCUMENTS_CACHE:
            load_documents_from_disk()

        if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
            return JSONResponse({"ok": False, "error": "No hay documentos cargados para este cliente"}, status_code=400)

        # Buscar el documento MÁS relevante
        relevant_docs = search_relevant_documents(tenant, q, top_k=1)
        
        if not relevant_docs:
            return JSONResponse({"ok": False, "error": "No se encontraron documentos relevantes"}, status_code=400)
        
        doc_name = relevant_docs[0]
        doc_content = DOCUMENTS_CACHE[tenant].get(doc_name, "")
        
        if not doc_content.strip():
            return JSONResponse({"ok": False, "error": f"El documento {doc_name} no tiene contenido"}, status_code=400)
        
        # Limitar contexto
        if len(doc_content) > 8000:
            doc_content = doc_content[:8000] + "..."

        # Prompt simple y directo
        system_prompt = f"""Eres un asistente que responde preguntas basadas ÚNICAMENTE en el documento: {doc_name}

REGLAS:
1. Solo responde con información del documento
2. Si no encuentras la respuesta, di: "No encontré esta información en el documento"
3. NUNCA cites otros documentos
4. Sé conciso"""

        full_prompt = f"{system_prompt}\n\nDocumento:\n{doc_content}\n\nPregunta: {q}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=400,
                top_p=0.7
            )
        )
        
        answer = response.text
        
        # Verificar alucinaciones
        if check_hallucination(answer, doc_name, doc_content):
            answer = "No encontré esta información en el documento."

        save_conversation(tenant, q, answer, [doc_name])

        # Extraer versión y fecha del documento
        metadata = extract_version_and_date(doc_name)
        source_info = doc_name
        if metadata["version"]:
            source_info += f" (v{metadata['version']})"
        if metadata["date"]:
            source_info += f" - {metadata['date']}"
        
        return {"ok": True, "answer": answer, "sources": [source_info]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/history")
async def get_history(tenant: str):
    try:
        init_db(tenant)
        history = get_conversation_history(tenant)
        return {"ok": True, "history": history}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/delete-document")
async def delete_document(tenant: str = Form(...), filename: str = Form(...)):
    try:
        tenant_dir = DATA_DIR / tenant
        pdf_path = tenant_dir / filename
        txt_path = tenant_dir / (Path(filename).stem + ".txt")
        
        if pdf_path.exists():
            pdf_path.unlink()
        if txt_path.exists():
            txt_path.unlink()
        
        if tenant in DOCUMENTS_CACHE and filename in DOCUMENTS_CACHE[tenant]:
            del DOCUMENTS_CACHE[tenant][filename]
            create_embeddings(tenant)
        
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/delete-all-documents")
async def delete_all_documents(tenant: str = Form(...)):
    try:
        tenant_dir = DATA_DIR / tenant
        if tenant_dir.exists():
            shutil.rmtree(str(tenant_dir))
        
        if tenant in DOCUMENTS_CACHE:
            del DOCUMENTS_CACHE[tenant]
        if tenant in VECTORIZERS_CACHE:
            del VECTORIZERS_CACHE[tenant]
        if tenant in EMBEDDINGS_CACHE:
            del EMBEDDINGS_CACHE[tenant]
        
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
