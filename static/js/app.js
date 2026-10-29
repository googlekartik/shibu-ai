const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const typingEl = document.getElementById("typing");
const detailsEl = document.getElementById("details");

function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// Chat history is saved in this browser only (localStorage), so closing
// and reopening the page restores your chat. The server still keeps its
// own session history separately, and Clear wipes both.
function saveToHistory(text, role) {
  try {
    const saved = JSON.parse(localStorage.getItem("rb_chat_history") || "[]");
    saved.push({ text, role, time: new Date().toLocaleString() });
    localStorage.setItem("rb_chat_history", JSON.stringify(saved.slice(-100)));
  } catch (e) { /* private browsing etc. - chat still works without saving */ }
}

function renderHistory() {
  const list = document.getElementById("historyList");
  const empty = document.getElementById("historyEmpty");
  list.innerHTML = "";
  let questions = [];
  try {
    const saved = JSON.parse(localStorage.getItem("rb_chat_history") || "[]");
    questions = saved.filter(m => m.role === "user").map(m => m.text);
  } catch (e) {}
  // Most recent first, duplicates removed, last 20.
  const unique = [...new Set(questions.reverse())].slice(0, 20);
  empty.style.display = unique.length ? "none" : "block";
  unique.forEach(q => {
    const li = document.createElement("li");
    li.textContent = q;
    li.onclick = () => {
      document.getElementById("historyPanel").classList.add("hidden");
      sendMessage(q);
    };
    list.appendChild(li);
  });
}

function addMessage(text, role, extra, saveHistory = true) {
  if (saveHistory) saveToHistory(text, role);
  const div = document.createElement("div");
  div.className = "msg " + role;
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(body);
  // AI Environment Properties: rendered as a separate table below the
  // main answer (never mixed into it), when the server classified a
  // scenario. The text version is already in the message for copy /
  // API clients; this table is the readable UI section.
  if (role === "bot" && extra && extra.environment_analysis) {
    const env = extra.environment_analysis;
    const wrap = document.createElement("div");
    wrap.className = "env-table";
    const title = document.createElement("div");
    title.className = "env-title";
    title.textContent = "AI Environment Properties — " + (env.scenario || "");
    wrap.appendChild(title);
    const table = document.createElement("table");
    const head = document.createElement("tr");
    ["Property", "Classification", "Reason"].forEach(h => {
      const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
    });
    table.appendChild(head);
    (env.properties || []).forEach(p => {
      const tr = document.createElement("tr");
      [p.property, p.classification, p.reason].forEach(v => {
        const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    wrap.appendChild(table);
    div.appendChild(wrap);
  }
  if (role === "bot") {
    const tools = document.createElement("div");
    tools.className = "tools";
    const copy = document.createElement("button");
    copy.textContent = "Copy";
    copy.onclick = () => navigator.clipboard.writeText(text);
    tools.appendChild(copy);
    if (extra && (extra.file_url || extra.pdf_url)) {
      const dl = document.createElement("a");
      dl.href = extra.file_url || extra.pdf_url;
      dl.textContent = extra.pdf_url && !extra.file_url ? "Download PDF" : "Download File";
      tools.appendChild(dl);
    }
    div.appendChild(tools);
  }
  messagesEl.appendChild(div);
  scrollDown();
}

function updateExplainability(d) {
  if (!d) return;
  document.getElementById("exIntent").textContent = d.intent || "—";
  document.getElementById("exRule").textContent = d.rule_id || "—";
  document.getElementById("exConf").textContent = d.confidence != null
    ? Math.round(d.confidence * 100) + "% (" + (d.confidence_label || "") + ")" : "—";
  document.getElementById("exAction").textContent = d.action || "—";
  document.getElementById("exTool").textContent = d.tool || "—";
  document.getElementById("exTime").textContent = d.processing_ms != null ? d.processing_ms + " ms" : "—";
}

async function sendMessage(text) {
  text = (text || "").trim();
  if (!text) return;
  addMessage(text, "user");
  inputEl.value = "";
  inputEl.style.height = "auto";
  typingEl.classList.remove("hidden");
  try {
    const res = await fetch("/api/chat", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }) });
    const data = await res.json();
    typingEl.classList.add("hidden");
    if (!res.ok) { addMessage("Error: " + (data.error || "Something went wrong."), "bot"); return; }
    addMessage(data.response, "bot", data);
    updateExplainability(data);
  } catch (e) {
    typingEl.classList.add("hidden");
    addMessage("Could not reach the server. Is Flask running on http://127.0.0.1:5000 ?", "bot");
  }
}

document.getElementById("attachBtn").onclick = () => document.getElementById("fileInput").click();
document.getElementById("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  addMessage("Uploaded: " + file.name, "user");
  typingEl.classList.remove("hidden");
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    typingEl.classList.add("hidden");
    if (!res.ok) { addMessage("Upload failed: " + (data.error || "Unknown error."), "bot"); return; }
    addMessage(data.message + "\n\nPreview: " + data.summary_preview, "bot");
  } catch (err) {
    typingEl.classList.add("hidden");
    addMessage("Upload failed: could not reach the server.", "bot");
  }
  e.target.value = "";
});
document.getElementById("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value);
});
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(inputEl.value); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
});
// Google login state in the header: Sign in -> /login page, or
// display name + Log out when already signed in. Chat works either way.
async function refreshLoginState() {
  const loginBtn = document.getElementById("loginBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const userLabel = document.getElementById("userLabel");
  if (!loginBtn || !logoutBtn || !userLabel) return;
  try {
    const res = await fetch("/api/me");
    const me = await res.json();
    if (me.logged_in) {
      loginBtn.classList.add("hidden");
      logoutBtn.classList.remove("hidden");
      userLabel.textContent = me.display_name ? "Signed in as " + me.display_name : "Signed in";
      userLabel.classList.remove("hidden");
    } else {
      loginBtn.classList.remove("hidden");
      logoutBtn.classList.add("hidden");
      userLabel.classList.add("hidden");
    }
  } catch (e) { /* header stays on Sign in; chat still works */ }
}
document.getElementById("logoutBtn").onclick = async () => {
  await fetch("/auth/logout", { method: "POST" });
  await refreshLoginState();
};
refreshLoginState();

document.getElementById("detailsBtn").onclick = () => detailsEl.classList.toggle("hidden");
document.getElementById("historyBtn").onclick = () => {
  renderHistory();
  document.getElementById("historyPanel").classList.toggle("hidden");
};
document.getElementById("clearBtn").onclick = async () => {
  await fetch("/api/clear", { method: "POST" });
  try { localStorage.removeItem("rb_chat_history"); } catch (e) {}
  messagesEl.innerHTML = "";
  renderHistory();
  addMessage("Hello! I'm Shibu AI. How can I help you today?", "bot");
};

// Voice input (browser Speech Recognition, no server needed).
// The mic button is hidden on browsers that do not support it.
const micBtn = document.getElementById("micBtn");
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRec) {
  micBtn.style.display = "none";
} else {
  micBtn.onclick = () => {
    const rec = new SpeechRec();
    rec.lang = navigator.language || "en-US";
    rec.interimResults = false;
    micBtn.classList.add("listening");
    rec.onresult = (e) => { inputEl.value = e.results[0][0].transcript; inputEl.focus(); };
    rec.onend = () => micBtn.classList.remove("listening");
    rec.onerror = () => micBtn.classList.remove("listening");
    rec.start();
  };
}

// Restore saved chat if there is one, otherwise show a minimal greeting.
let restored = false;
try {
  const saved = JSON.parse(localStorage.getItem("rb_chat_history") || "[]");
  saved.forEach(m => addMessage(m.text, m.role, null, false));
  restored = saved.length > 0;
} catch (e) {}
if (!restored) addMessage("Hello! I'm Shibu AI. How can I help you today?", "bot");
