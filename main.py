import os
import uuid
import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Mini RAG Chatbot (Ollama + Chroma)")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/v1")
CHAT_MODEL = os.getenv("MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

llm = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


def get_collection():
    # Connect when needed, so the app can start even if Chroma is still booting.
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def embed(texts):
    resp = llm.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def split_text(text, size=500):
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in parts:
        if len(current) + len(p) < size:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


class Message(BaseModel):
    text: str
    lang: str = "en"


@app.post("/add")
def add(msg: Message):
    try:
        chunks = split_text(msg.text)
        if not chunks:
            return {"status": "Nothing to add."}
        get_collection().add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embed(chunks),
        )
        return {"status": f"Added {len(chunks)} chunk(s) to the database."}
    except Exception as e:
        return {"status": f"ERROR: {e}"}


SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant. "
    "You should act as a romantic girl friend, your name is ritika"
    "If the user greets you or makes small talk, reply naturally and briefly. "
    "If the user asks a factual question, answer using the provided context. "
    "If the answer is not in the context, say you don't know instead of guessing."
)


KANNADA_RULE = (
    " IMPORTANT: Always reply ONLY in Kannada (ಕನ್ನಡ) using Kannada script. "
    "Use short, simple sentences. Do not use English words unless there is no Kannada word."
)


@app.post("/chat")
def chat(msg: Message):
    try:
        docs = []
        col = get_collection()
        if col.count() > 0:
            result = col.query(query_embeddings=embed([msg.text]), n_results=3)
            docs = result["documents"][0]
        context = "\n---\n".join(docs) if docs else "(no documents have been added yet)"
        user_prompt = f"Context:\n{context}\n\nUser message: {msg.text}"
        resp = llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + (KANNADA_RULE if msg.lang == "kn" else "")},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )
        return {"reply": resp.choices[0].message.content, "sources": docs}
    except Exception as e:
        return {"reply": f"ERROR: {e}", "sources": []}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini RAG Chatbot</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
    color: #eef0ff;
    background: #0b1026;
    overflow-x: hidden;
  }
  /* Animated aurora background */
  .bg { position: fixed; inset: 0; z-index: -1; overflow: hidden;
        background: radial-gradient(1200px 800px at 20% 10%, #1b2a6b 0%, transparent 60%),
                    radial-gradient(1000px 700px at 85% 90%, #4a1d6b 0%, transparent 60%),
                    #0b1026; }
  .blob { position: absolute; border-radius: 50%; filter: blur(80px); opacity: .55;
          animation: float 18s ease-in-out infinite; }
  .b1 { width: 520px; height: 520px; background: #6366f1; top: -120px; left: -100px; }
  .b2 { width: 460px; height: 460px; background: #ec4899; bottom: -140px; right: -80px; animation-delay: -6s; }
  .b3 { width: 380px; height: 380px; background: #22d3ee; top: 40%; left: 55%; animation-delay: -11s; opacity: .35; }
  @keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(60px, 40px) scale(1.12); }
    66% { transform: translate(-40px, 70px) scale(.94); }
  }
  .stars { position: absolute; inset: 0;
           background-image:
             radial-gradient(1.5px 1.5px at 20px 30px, #fff, transparent),
             radial-gradient(1px 1px at 120px 90px, #fff, transparent),
             radial-gradient(1.5px 1.5px at 200px 160px, #cbd5ff, transparent),
             radial-gradient(1px 1px at 280px 40px, #fff, transparent);
           background-size: 320px 220px; opacity: .45; animation: twinkle 6s ease-in-out infinite; }
  @keyframes twinkle { 0%, 100% { opacity: .25; } 50% { opacity: .6; } }

  .wrap { max-width: 760px; margin: 0 auto; padding: 28px 16px 40px; }
  header { text-align: center; margin-bottom: 22px; }
  header h1 { margin: 0; font-size: 2rem; letter-spacing: .5px;
              background: linear-gradient(90deg, #a5b4fc, #f0abfc, #67e8f9);
              -webkit-background-clip: text; background-clip: text; color: transparent; }
  header p { margin: 6px 0 0; color: #b6bce6; font-size: .95rem; }

  .card { background: rgba(255, 255, 255, .07); border: 1px solid rgba(255, 255, 255, .14);
          border-radius: 20px; padding: 18px; margin-bottom: 18px;
          backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
          box-shadow: 0 12px 40px rgba(0, 0, 0, .35); }
  .card h3 { margin: 0 0 12px; font-size: 1rem; font-weight: 600; color: #d9ddff; }
  .card h3 span { display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center;
                  border-radius: 50%; margin-right: 8px; font-size: .8rem;
                  background: linear-gradient(135deg, #6366f1, #ec4899); color: #fff; }

  textarea, input[type=text], select {
    width: 100%; color: #eef0ff; font: inherit; outline: none;
    background: rgba(8, 12, 40, .55); border: 1px solid rgba(255, 255, 255, .16);
    border-radius: 12px; padding: 12px 14px; transition: border-color .2s, box-shadow .2s; }
  textarea { resize: vertical; min-height: 110px; }
  textarea:focus, input[type=text]:focus, select:focus { border-color: #a5b4fc; box-shadow: 0 0 0 3px rgba(129, 140, 248, .3); }
  ::placeholder { color: #8e95c7; }
  select option { background: #141a45; color: #eef0ff; }

  button { cursor: pointer; border: 0; color: #fff; font: inherit; font-weight: 600;
           padding: 11px 18px; border-radius: 12px;
           background: linear-gradient(135deg, #6366f1, #a855f7 55%, #ec4899);
           box-shadow: 0 6px 18px rgba(139, 92, 246, .4); transition: transform .15s, box-shadow .15s, filter .15s; }
  button:hover { transform: translateY(-2px); filter: brightness(1.1); box-shadow: 0 10px 24px rgba(139, 92, 246, .55); }
  button:active { transform: translateY(0); }
  button.ghost { background: rgba(255, 255, 255, .1); box-shadow: none; border: 1px solid rgba(255, 255, 255, .2); }
  button.live { background: linear-gradient(135deg, #ef4444, #f97316); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, .6); } 100% { box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); } }

  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .row.grow > input { flex: 1; min-width: 160px; }
  #status { color: #86efac; font-size: .9rem; }
  #voicewarn { color: #fca5a5; font-size: .85rem; }

  #log { height: 320px; overflow-y: auto; padding: 6px 4px 6px 2px; margin-bottom: 14px; scroll-behavior: smooth; }
  #log::-webkit-scrollbar { width: 8px; }
  #log::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, .22); border-radius: 8px; }
  .empty { text-align: center; color: #8e95c7; margin-top: 110px; font-size: .95rem; }
  .msg { max-width: 82%; padding: 10px 14px; margin: 8px 0; border-radius: 18px; line-height: 1.45;
         white-space: pre-wrap; word-wrap: break-word; animation: pop .25s ease-out; }
  .msg.you { margin-left: auto; border-bottom-right-radius: 4px;
             background: linear-gradient(135deg, #6366f1, #a855f7); box-shadow: 0 6px 16px rgba(99, 102, 241, .35); }
  .msg.bot { margin-right: auto; border-bottom-left-radius: 4px;
             background: rgba(255, 255, 255, .1); border: 1px solid rgba(255, 255, 255, .14); }
  @keyframes pop { from { opacity: 0; transform: translateY(8px) scale(.97); } to { opacity: 1; transform: none; } }

  .opts { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; margin-top: 14px; font-size: .9rem; color: #c5caf0; }
  .opts label { display: flex; flex-direction: column; gap: 6px; }
  .opts label.check { grid-column: 1 / -1; flex-direction: row; align-items: center; gap: 8px; }
  .check input { width: 18px; height: 18px; accent-color: #a855f7; }
  @media (max-width: 560px) { .opts { grid-template-columns: 1fr; } .msg { max-width: 92%; } }
</style>
</head>
<body>
<div class="bg"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="stars"></div></div>

<div class="wrap">
  <header>
    <h1>&#10024; Mini RAG Chatbot</h1>
    <p>Ask questions about your own text &middot; powered by Ollama + Chroma</p>
  </header>

  <div class="card">
    <h3><span>1</span>Add knowledge</h3>
    <textarea id="doc" placeholder="Paste some text here..."></textarea>
    <div class="row" style="margin-top:12px">
      <button onclick="addDoc()">Add to database</button>
      <span id="status"></span>
    </div>
  </div>

  <div class="card">
    <h3><span>2</span>Ask questions</h3>
    <div id="log"><div class="empty" id="empty">Say hi, or ask something about the text you added &#128172;</div></div>
    <div class="row grow">
      <input type="text" id="box" placeholder="Ask about your text..." />
      <button onclick="send()">Send</button>
      <button id="talk" class="ghost" onclick="toggleTalk()">&#127908; Start voice chat</button>
    </div>
    <div class="opts">
      <label class="check"><input type="checkbox" id="speak" checked> Read replies aloud</label>
      <label>Language
        <select id="lang" onchange="loadVoices()">
          <option value="en">English</option>
          <option value="kn">Kannada (ಕನ್ನಡ)</option>
        </select>
      </label>
      <label>Voice <select id="voice"></select></label>
      <span id="voicewarn"></span>
    </div>
  </div>
</div>

  <script>
    const log = document.getElementById("log");
    function line(who, text) {
      const empty = document.getElementById("empty");
      if (empty) empty.remove();
      const d = document.createElement("div");
      d.className = "msg " + (who === "You" ? "you" : "bot");
      d.textContent = text;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
    }
    const FEMALE = ["sapna", "neerja", "heera", "zira", "aria", "jenny", "samantha", "susan", "hazel",
                    "libby", "sonia", "natasha", "clara", "michelle", "ava", "female", "woman"];
    let voices = [];
    function isFemale(v) { return FEMALE.some(n => v.name.toLowerCase().includes(n)); }
    function loadVoices() {
      voices = window.speechSynthesis.getVoices();
      const sel = document.getElementById("voice");
      sel.innerHTML = "";
      voices.forEach((v, i) => {
        const o = document.createElement("option");
        o.value = i;
        o.textContent = v.name + " (" + v.lang + ")";
        sel.appendChild(o);
      });
      const lang = document.getElementById("lang").value;
      let idx;
      if (lang === "kn") {
        idx = voices.findIndex(v => v.lang.startsWith("kn") && isFemale(v));
        if (idx < 0) idx = voices.findIndex(v => v.lang.startsWith("kn"));
        document.getElementById("voicewarn").textContent =
          idx < 0 ? " No Kannada voice found. Use Microsoft Edge." : "";
      } else {
        document.getElementById("voicewarn").textContent = "";
        idx = voices.findIndex(v => v.lang === "en-IN" && isFemale(v));
        if (idx < 0) idx = voices.findIndex(v => v.lang.startsWith("en") && isFemale(v));
        if (idx < 0) idx = voices.findIndex(v => isFemale(v));
      }
      if (idx >= 0) sel.value = idx;
    }
    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();
    let talking = false, busy = false, rec = null;
    function speak(text, done) {
      if (!document.getElementById("speak").checked) { if (done) done(); return; }
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const v = voices[document.getElementById("voice").value];
      if (v) { u.voice = v; u.lang = v.lang; }
      if (document.getElementById("lang").value === "kn") u.lang = "kn-IN";
      u.onend = () => { if (done) done(); };
      u.onerror = () => { if (done) done(); };
      window.speechSynthesis.speak(u);
    }
    function listen() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { alert("Voice input needs Chrome or Edge."); talking = false; return; }
      rec = new SR();
      rec.lang = document.getElementById("lang").value === "kn" ? "kn-IN" : "en-US";
      rec.onresult = (e) => {
        busy = true;
        document.getElementById("box").value = e.results[0][0].transcript;
        send();
      };
      rec.onend = () => { if (talking && !busy) listen(); };
      rec.onerror = (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
          talking = false;
          document.getElementById("talk").textContent = "\U0001F3A4 Start voice chat"; document.getElementById("talk").classList.remove("live");
          alert("Microphone permission was blocked. Allow it in the browser address bar.");
        }
      };
      try { rec.start(); } catch (err) {}
    }
    function toggleTalk() {
      talking = !talking;
      const btn = document.getElementById("talk");
      if (talking) {
        document.getElementById("speak").checked = true;
        btn.textContent = "\u23F9 Stop voice chat"; btn.classList.add("live");
        busy = false;
        listen();
      } else {
        btn.textContent = "\U0001F3A4 Start voice chat"; btn.classList.remove("live");
        window.speechSynthesis.cancel();
        if (rec) rec.stop();
      }
    }
    async function post(url, text) {
      const r = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, lang: document.getElementById("lang").value})
      });
      return r.json();
    }
    async function addDoc() {
      const d = document.getElementById("doc");
      if (!d.value.trim()) return;
      document.getElementById("status").textContent = " Adding...";
      const data = await post("/add", d.value);
      document.getElementById("status").textContent = " " + data.status;
      d.value = "";
    }
    async function send() {
      const box = document.getElementById("box");
      const text = box.value; if (!text) { busy = false; return; }
      line("You", text);
      box.value = "";
      let reply = "";
      try {
        const data = await post("/chat", text);
        reply = data.reply;
      } catch (err) {
        reply = "ERROR: could not reach the server.";
      }
      line("Bot", reply);
      speak(reply, () => { busy = false; if (talking) listen(); });
    }
    document.getElementById("box").addEventListener("keydown", e => { if (e.key === "Enter") send(); });
  </script>
</body>
</html>
"""