import http.server
import json
import os
import socketserver
import sys

PORT = 7860
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAPTOR Self-RAG Engine — Interactive Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --accent: #58a6ff; --green: #3fb950; --text: #c9d1d9; }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 2rem; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 1.5rem; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.3); padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 2rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; }
    .btn { background: var(--accent); color: #fff; border: none; padding: 0.75rem 1.25rem; border-radius: 6px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 1rem; }
    .btn:hover { opacity: 0.9; }
    textarea, input { width: 100%; background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem; outline: none; }
    .chat-box { height: 400px; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1rem; }
    .msg { padding: 1rem; border-radius: 8px; font-size: 0.95rem; line-height: 1.5; }
    .user { background: #1f242d; border: 1px solid #383e4a; align-self: flex-end; max-width: 80%; }
    .assistant { background: #161b22; border: 1px solid var(--border); align-self: flex-start; max-width: 90%; }
    .tag { display: inline-block; background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid rgba(63, 185, 80, 0.3); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-right: 0.4rem; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <header>
    <h1>🧠 RAPTOR Self-RAG Engine <span class="badge">Local-First (Ollama/vLLM)</span></h1>
    <div><span class="badge" style="color:var(--green); border-color:var(--green);">Qdrant Connected</span></div>
  </header>
  <div class="grid">
    <div class="card">
      <h3>📁 Ingest Document</h3>
      <p style="font-size:0.85rem; color:#8b949e; margin-top:0.25rem;">Builds RAPTOR Hierarchical Summarization Tree</p>
      <input type="text" id="docId" placeholder="Document ID (e.g. doc_001)" value="architecture_doc">
      <textarea id="docContent" rows="8" placeholder="Paste document content here...">Industrial IoT edge platforms support MQTT, OPC-UA, and Modbus TCP/RTU.
Security standard TLS 1.3 is enforced for telemetry stream transport encryption.
ISO 27001 Zero Trust Architecture compliance is certified.</textarea>
      <button class="btn" onclick="ingestDoc()">Build RAPTOR Tree Index</button>
      <div id="ingestStatus" style="margin-top:1rem; font-size:0.85rem;"></div>
    </div>
    <div class="card" style="display:flex; flex-direction:column;">
      <h3>💬 Grounded Retrieval Chat</h3>
      <div class="chat-box" id="chatBox">
        <div class="msg assistant">
          Hello! I am your RAPTOR Self-RAG Assistant. Ask any question about your ingested documents.
        </div>
      </div>
      <div style="display:flex; gap:0.5rem;">
        <input type="text" id="queryInput" placeholder="Type query (e.g. What protocols are supported?)..." onkeypress="if(event.key==='Enter') sendQuery()">
        <button class="btn" style="width:auto; margin:0;" onclick="sendQuery()">Query</button>
      </div>
    </div>
  </div>
  <script>
    async function ingestDoc() {
      const docId = document.getElementById('docId').value;
      const content = document.getElementById('docContent').value;
      const status = document.getElementById('ingestStatus');
      status.innerHTML = '⏳ Building RAPTOR Tree...';
      try {
        const res = await fetch('http://localhost:8000/ingest', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ document_id: docId, content: content })
        });
        const data = await res.json();
        status.innerHTML = `✅ Ingested! Created ${data.leaf_chunks_created} leaves, ${data.raptor_tree_nodes} RAPTOR tree nodes.`;
      } catch (err) {
        status.innerHTML = '❌ Error: ' + err.message;
      }
    }
    async function sendQuery() {
      const input = document.getElementById('queryInput');
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      const box = document.getElementById('chatBox');
      box.innerHTML += `<div class="msg user">${q}</div>`;
      try {
        const res = await fetch('http://localhost:8000/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query: q })
        });
        const data = await res.json();
        let tags = (data.reflection_tokens || []).map(t => `<span class="tag">${t}</span>`).join('');
        box.innerHTML += `<div class="msg assistant"><div>${data.answer}</div><div>${tags}</div><div style="font-size:0.75rem; color:#8b949e; margin-top:0.4rem;">Model: ${data.routing.target_model} | Latency: ${data.latency_ms} ms</div></div>`;
        box.scrollTop = box.scrollHeight;
      } catch (err) {
        box.innerHTML += `<div class="msg assistant">Error: ${err.message}</div>`;
      }
    }
  </script>
</body>
</html>
"""

class UIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_CONTENT.encode("utf-8"))

def launch_ui():
    print(f"=== Launching RAPTOR Self-RAG Web Dashboard on http://localhost:{PORT} ===")
    with socketserver.TCPServer(("", PORT), UIHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("UI Dashboard stopped.")

if __name__ == "__main__":
    launch_ui()
