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
    
    # PASO 2: Buscar por nombre de documento
    for doc_name in doc_names:
        if query_lower in doc_name.lower():
            print(f"✅ Encontrado por nombre: {doc_name}")
            return [doc_name]
    
    # PASO 3: Búsqueda por contenido (TF-IDF)
    if tenant not in VECTORIZERS_CACHE or tenant not in EMBEDDINGS_CACHE:
        return []
    
    vectorizer = VECTORIZERS_CACHE[tenant]
    embeddings = EMBEDDINGS_CACHE[tenant]
    
    try:
        query_vec = vectorizer.transform([query]).toarray()
        similarities = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [doc_names[i] for i in top_indices if similarities[i] > 0.1]
        return results if results else []
    except:
        return []

def is_meta_question(question: str) -> bool:
    """Verifica si la pregunta es sobre el sistema (meta-pregunta)"""
    q_lower = question.lower()
    system_keywords = ['cuántos documentos', 'qué documentos', 'lista de documentos', 'documentos cargados', 'cuántas páginas', 'total de']
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
            <input id="file" type="file" accept="application/pdf" multiple class="text-sm"/>
            <p class="text-xs text-gray-500 mt-1">Múltiples archivos</p>
          </div>

          <button id="uploadBtn" class="w-full bg-blue-600 text-white py-2 rounded-lg font-semibold mb-2 hover:bg-blue-700">Subir e indexar</button>
          
          <div id="status" class="text-xs text-gray-600 mb-3 min-h-6"></div>

          <button id="historyBtn" class="w-full bg-gray-700 text-white py-2 rounded-lg text-sm mb-2 hover:bg-gray-800">Ver Historial</button>
          <button id="refreshBtn" class="w-full bg-green-600 text-white py-2 rounded-lg text-sm mb-2 hover:bg-green-700">Recargar Documentos</button>
          <button id="deleteAllBtn" class="w-full bg-red-600 text-white py-2 rounded-lg text-sm hover:bg-red-700">Eliminar Todos</button>
        </div>
      </div>

      <!-- Panel Central: Chat -->
      <div class="lg:col-span-2">
        <div class="bg-white rounded-xl shadow-lg p-6 flex flex-col" style="height: 600px;">
          <h2 class="text-lg font-semibold mb-4">💬 Chat</h2>
          <div id="messages" class="flex-1 overflow-y-auto mb-4 space-y-2 pr-2"></div>
          
          <div class="flex gap-2">
            <input id="q" type="text" placeholder="Pregunta sobre tus documentos... (Enter para enviar)" class="flex-1 border rounded-lg p-2"/>
            <button id="askBtn" class="bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-blue-700">Enviar</button>
          </div>
        </div>
      </div>

      <!-- Panel Derecho: Documentos -->
      <div class="lg:col-span-1">
        <div class="bg-white rounded-xl shadow-lg p-4 h-96 overflow-y-auto">
          <h2 class="text-lg font-semibold mb-4">📄 Documentos</h2>
          <div id="docList" class="space-y-2 text-sm"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
const statusEl = document.getElementById('status');
const messagesEl = document.getElementById('messages');
const docListEl = document.getElementById('docList');

function addMsg(role, text) {
  const now = new Date();
  const time = now.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  
  const div = document.createElement('div');
  div.className = role === 'user' ? 'bg-blue-100 p-2 rounded text-xs mb-2' : 'bg-gray-100 p-2 rounded text-xs mb-2';
  
  const timeEl = document.createElement('small');
  timeEl.className = 'text-gray-500 block text-xs mb-1';
  timeEl.textContent = time;
  
  const textEl = document.createElement('div');
  textEl.innerHTML = text.replace(/\\n/g, '<br>');
  
  div.appendChild(timeEl);
  div.appendChild(textEl);
  
  messagesEl.insertBefore(div, messagesEl.firstChild);
  messagesEl.scrollTop = 0;
}

async function loadDocuments() {
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) return;
  
  try {
    const res = await fetch(`/documents?tenant=${tenant}`);
    const data = await res.json();
    
    docListEl.innerHTML = '';
    if (data.documents && data.documents.length > 0) {
      data.documents.forEach(([name, size]) => {
        const div = document.createElement('div');
        div.className = 'flex justify-between items-center p-2 bg-gray-100 rounded text-xs';
        div.innerHTML = `<span>${name}</span><button onclick="deleteDocument('${name}')" class="text-red-600 hover:text-red-800">✕</button>`;
        docListEl.appendChild(div);
      });
    } else {
      docListEl.innerHTML = '<p class="text-gray-500">Sin documentos cargados</p>';
    }
  } catch (err) {
    docListEl.innerHTML = '<p class="text-red-500">Error cargando documentos</p>';
  }
}

document.getElementById('uploadBtn').onclick = async () => {
  const files = document.getElementById('file').files;
  if (files.length === 0) {
    alert('Selecciona al menos un archivo');
    return;
  }
  
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) {
    alert('Ingresa un tenant');
    return;
  }
  
  const fd = new FormData();
  fd.append('tenant', tenant);
  for (let file of files) {
    fd.append('files', file);
  }
  
  const totalFiles = files.length;
  statusEl.textContent = `⏳ Subiendo 0/${totalFiles}...`;
  
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      const uploadedCount = data.files ? Math.floor(data.files.length / 2) : totalFiles;
      statusEl.textContent = `✅ ${uploadedCount}/${totalFiles} archivos subidos`;
      document.getElementById('file').value = '';
      setTimeout(loadDocuments, 500);
    } else {
      statusEl.textContent = `❌ Error: ${data.error}`;
    }
  } catch (err) {
    statusEl.textContent = `❌ Error: ${err.message}`;
  }
};

document.getElementById('askBtn').onclick = async () => {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) {
    alert('Ingresa un tenant');
    return;
  }
  
  addMsg('user', q);
  document.getElementById('q').value = '';
  
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tenant, question: q })
    });
    const data = await res.json();
    
    if (data.ok) {
      let response = data.answer;
      if (data.source) {
        response += `\\n\\n📄 Fuente: ${data.source}`;
      }
      addMsg('assistant', response);
    } else {
      addMsg('assistant', `❌ ${data.error}`);
    }
  } catch (err) {
    addMsg('assistant', `❌ Error: ${err.message}`);
  }
};

document.getElementById('q').onkeypress = (e) => {
  if (e.key === 'Enter') document.getElementById('askBtn').click();
};

document.getElementById('historyBtn').onclick = async () => {
  const tenant = document.getElementById('tenant').value.trim();
  if (!tenant) return;
  
  try {
    const res = await fetch(`/history?tenant=${tenant}`);
    const data = await res.json();
    if (data.ok) {
      alert('Historial: ' + JSON.stringify(data.history, null, 2));
    }
  } catch (err) {
    alert('Error cargando historial');
  }
};

document.getElementById('refreshBtn').onclick = loadDocuments;

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
async def upload(tenant: str = Form(...), files: list[UploadFile] = File(None)):
    try:
        tenant = (tenant or "").strip()
        if not tenant:
            return JSONResponse({"ok": False, "error": "tenant requerido"}, status_code=400)
        
        init_db(tenant)
        tenant_dir = DATA_DIR / tenant
        tenant_dir.mkdir(exist_ok=True)
        
        if tenant not in DOCUMENTS_CACHE:
            DOCUMENTS_CACHE[tenant] = {}
        
        uploaded_files = []
        
        # Si no hay archivos, retornar error
        if not files:
            return JSONResponse({"ok": False, "error": "No files provided"}, status_code=400)
        
        # Asegurar que files es una lista
        if not isinstance(files, list):
            files = [files]
        
        for file in files:
            # Validar que el archivo tenga nombre
            if not file or not hasattr(file, 'filename') or not file.filename:
                continue
            
            try:
                # Leer el contenido del archivo
                file_content = await file.read()
                
                # Guardar el archivo
                out_path = tenant_dir / file.filename
                with open(out_path, 'wb') as f:
                    f.write(file_content)
                
                # Extraer texto del PDF
                text_content, page_count = extract_text_from_pdf(str(out_path))
                
                if text_content.strip():
                    # Guardar en caché
                    DOCUMENTS_CACHE[tenant][file.filename] = text_content
                    uploaded_files.append(file.filename)
                    
                    # Guardar archivo de texto
                    txt_path = tenant_dir / (out_path.stem + ".txt")
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    
                    # Guardar metadatos
                    file_size = out_path.stat().st_size
                    save_document_metadata(tenant, file.filename, file_size, page_count)
                    
                    print(f"✅ {file.filename} procesado exitosamente")
                else:
                    print(f"⚠️ {file.filename} no contiene texto")
                    
            except Exception as file_err:
                print(f"❌ Error procesando {file.filename}: {file_err}")
                import traceback
                traceback.print_exc()
                continue
        
        # Crear embeddings si hay archivos
        if uploaded_files:
            create_embeddings(tenant)
        
        return {"ok": True, "files": uploaded_files}
    except Exception as e:
        print(f"❌ Error en /upload: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/ask")
async def ask(tenant: str = Form(...), question: str = Form(...)):
    try:
        tenant = (tenant or "").strip()
        if not tenant:
            return JSONResponse({"ok": False, "error": "tenant requerido"}, status_code=400)
        
        if tenant not in DOCUMENTS_CACHE:
            load_documents_from_disk()
        
        if tenant not in DOCUMENTS_CACHE or not DOCUMENTS_CACHE[tenant]:
            return {"ok": False, "error": "No hay documentos cargados para este cliente"}
        
        if is_meta_question(question):
            answer = get_system_info(tenant)
            return {"ok": True, "answer": answer, "source": "Sistema"}
        
        relevant_docs = search_relevant_documents(tenant, question, top_k=1)
        
        if not relevant_docs:
            return {"ok": False, "error": "No se encontraron documentos relevantes"}
        
        doc_name = relevant_docs[0]
        doc_content = DOCUMENTS_CACHE[tenant][doc_name]
        
        prompt = f"""Eres un asistente experto. Responde la siguiente pregunta basándote ÚNICAMENTE en el contenido del documento proporcionado.

DOCUMENTO: {doc_name}
CONTENIDO:
{doc_content}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde de forma concisa y clara
- Si la respuesta no está en el documento, di "No encontré información sobre esto en el documento"
- NO inventes información
- NO hagas referencias a otros documentos que no estén en el contenido proporcionado"""
        
        response = model.generate_content(prompt)
        answer = response.text if response.text else "No se pudo generar una respuesta"
        
        if check_hallucination(answer, doc_name, doc_content):
            answer = "⚠️ La respuesta podría contener información no verificada. Por favor, revisa el documento original."
        
        save_conversation(tenant, question, answer, [doc_name])
        
        return {"ok": True, "answer": answer, "source": doc_name}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/history")
async def history(tenant: str):
    try:
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
