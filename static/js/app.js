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

async function renderHistory() {
  const list = document.getElementById("sidebarList");
  const empty = document.getElementById("sidebarEmpty");
  if (!list || !empty) return;
  list.innerHTML = "";
  let questions = [];
  try {
    const saved = JSON.parse(localStorage.getItem("rb_chat_history") || "[]");
    questions = saved.filter(m => m.role === "user").map(m => m.text);
  } catch (e) {}
  // Logged-in users: merge the persistent SQLite conversation
  // list with this browser's history, so neither replaces the
  // other when one of them is empty.
  try {
    const res = await fetch("/api/conversations");
    if (res.ok) {
      const data = await res.json();
      const titles = (data.conversations || []).map(c => c.title).filter(Boolean);
      if (titles.length) questions = [...titles, ...questions];
    }
  } catch (e) {}
  // Most recent first, duplicates removed, last 20.
  const unique = [...new Set(questions.reverse())].slice(0, 20);
  empty.style.display = unique.length ? "none" : "block";
  unique.forEach(q => {
    const li = document.createElement("li");
    li.textContent = q;
    li.onclick = () => { sendMessage(q); };
    list.appendChild(li);
  });
}

// Structured presentation blocks (from the content-processing
// pipeline). Rendered only with safe DOM calls: block text is
// always textContent, never innerHTML, so model/web text cannot
// inject markup. If blocks are absent or invalid, the caller
// falls back to the plain-text body below.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function drawBarChart(canvas, labels, values) {
  const ctx = canvas.getContext("2d");
  if (!ctx || !values.length) return;
  const W = canvas.width, H = canvas.height, pad = 28;
  const max = Math.max(...values, 1);
  const barW = Math.min(48, (W - pad * 2) / values.length - 10);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#7c5cff";
  values.forEach((v, i) => {
    const h = (v / max) * (H - pad * 2 - 18);
    const x = pad + i * ((W - pad * 2) / values.length) + 6;
    ctx.fillRect(x, H - pad - h, barW, h);
    ctx.fillStyle = "#e6e6e6";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(String(v), x, H - pad - h - 5);
    ctx.fillText((labels[i] || "").slice(0, 16), x, H - 10);
    ctx.fillStyle = "#7c5cff";
  });
}

function renderBlocks(blocks, container) {
  blocks.forEach(block => {
    if (!block || !block.kind) return;
    if (block.kind === "heading") {
      container.appendChild(el(block.level === 3 ? "h4" : "h3", "pres-heading", block.text));
    } else if (block.kind === "paragraph") {
      if (block.expandable && typeof block.text === "string" && block.text.length > 450) {
        const cut = block.text.indexOf(". ", 40);
        const details = document.createElement("details");
        details.className = "pres-details";
        const summary = document.createElement("summary");
        summary.textContent = cut > 0 ? block.text.slice(0, cut + 1) : block.text.slice(0, 120);
        details.appendChild(summary);
        details.appendChild(el("p", "", cut > 0 ? block.text.slice(cut + 2) : block.text));
        container.appendChild(details);
      } else {
        container.appendChild(el("p", "pres-para", block.text));
      }
    } else if (block.kind === "callout") {
      const box = el("div", "pres-callout");
      if (block.title) box.appendChild(el("strong", "", block.title + ": "));
      box.appendChild(document.createTextNode(block.text || ""));
      container.appendChild(box);
    } else if (block.kind === "list" || block.kind === "steps") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      const list = document.createElement(block.kind === "steps" ? "ol" : "ul");
      list.className = block.kind === "steps" ? "pres-steps" : "pres-list";
      (block.items || []).forEach(item => list.appendChild(el("li", "", item)));
      container.appendChild(list);
    } else if (block.kind === "table") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      const table = document.createElement("table");
      table.className = "pres-table";
      if (block.headers) {
        const head = document.createElement("tr");
        block.headers.forEach(h => head.appendChild(el("th", "", h)));
        table.appendChild(head);
      }
      (block.rows || []).forEach(row => {
        const tr = document.createElement("tr");
        row.forEach(cell => tr.appendChild(el("td", "", cell)));
        table.appendChild(tr);
      });
      container.appendChild(table);
    } else if (block.kind === "pros_cons") {
      const wrap = el("div", "pres-pros-cons");
      const pros = el("div", "pres-pros"); pros.appendChild(el("div", "pres-subtitle", "Advantages"));
      const prosList = document.createElement("ul"); (block.pros || []).forEach(i => prosList.appendChild(el("li", "", i))); pros.appendChild(prosList);
      const cons = el("div", "pres-cons"); cons.appendChild(el("div", "pres-subtitle", "Disadvantages"));
      const consList = document.createElement("ul"); (block.cons || []).forEach(i => consList.appendChild(el("li", "", i))); cons.appendChild(consList);
      wrap.appendChild(pros); wrap.appendChild(cons); container.appendChild(wrap);
    } else if (block.kind === "code") {
      const pre = document.createElement("pre"); pre.className = "pres-code";
      pre.appendChild(el("code", "", block.text)); container.appendChild(pre);
    } else if (block.kind === "equation") {
      container.appendChild(el("div", "pres-equation", block.text));
    } else if (block.kind === "chart") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      const canvas = document.createElement("canvas");
      canvas.className = "pres-chart"; canvas.width = 560; canvas.height = 200;
      container.appendChild(canvas);
      drawBarChart(canvas, block.labels || [], block.values || []);
    } else if (block.kind === "timeline") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      const wrap = el("div", "pres-timeline");
      (block.events || []).forEach(ev => {
        const row = el("div", "pres-timeline-row");
        row.appendChild(el("span", "pres-year", ev.year));
        row.appendChild(el("span", "", ev.text));
        wrap.appendChild(row);
      });
      container.appendChild(wrap);
    } else if (block.kind === "flowchart") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      const wrap = el("div", "pres-flow");
      (block.nodes || []).forEach((node, i) => {
        if (i) wrap.appendChild(el("div", "pres-arrow", "↓"));
        wrap.appendChild(el("div", "pres-node pres-node-" + (node.type || "process"), node.label));
      });
      container.appendChild(wrap);
    } else if (block.kind === "gallery") {
      const wrap = el("div", "pres-gallery");
      (block.images || []).forEach(img => {
        if (typeof img.url === "string" && /^https?:\/\//.test(img.url)) {
          const image = document.createElement("img");
          image.src = img.url; image.alt = img.caption || "Image"; image.loading = "lazy";
          wrap.appendChild(image);
        }
      });
      if (wrap.children.length) container.appendChild(wrap);
    } else if (block.kind === "quiz") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      (block.questions || []).forEach(q => {
        const qWrap = el("div", "pres-quiz");
        qWrap.appendChild(el("div", "pres-quiz-q", q.question));
        const feedback = el("div", "pres-quiz-feedback");
        (q.options || []).forEach((opt, i) => {
          const b = document.createElement("button");
          b.type = "button"; b.className = "pres-quiz-opt"; b.textContent = opt;
          b.onclick = () => {
            const correct = i === q.answer_index;
            b.classList.add(correct ? "correct" : "wrong");
            feedback.textContent = (correct ? "Correct. " : "Not quite. ") + (q.explanation || "");
            [...qWrap.querySelectorAll("button")].forEach(btn => { btn.disabled = true; });
          };
          qWrap.appendChild(b);
        });
        qWrap.appendChild(feedback);
        container.appendChild(qWrap);
      });
    } else if (block.kind === "flashcards") {
      if (block.title) container.appendChild(el("div", "pres-subtitle", block.title));
      (block.cards || []).forEach(card => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "pres-card"; b.textContent = card.front;
        let showingBack = false;
        b.onclick = () => { showingBack = !showingBack; b.textContent = showingBack ? card.back : card.front; };
        container.appendChild(b);
      });
    }
  });
}

function addMessage(text, role, extra, saveHistory = true) {
  if (saveHistory) saveToHistory(text, role);
  const div = document.createElement("div");
  div.className = "msg " + role;
  const body = document.createElement("div");
  const presentation = role === "bot" && extra && extra.presentation;
  if (presentation && Array.isArray(presentation.blocks) && presentation.blocks.length) {
    renderBlocks(presentation.blocks, body);
    if (presentation.verification && presentation.verification.summary) {
      body.appendChild(el("div", "pres-verification", "Checked against sources: " + presentation.verification.summary));
    }
  } else {
    body.textContent = text;
  }
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
  if (role === "bot" && extra && extra.quick_actions) {
    const qa = document.createElement("div");
    qa.className = "quick-actions";
    extra.quick_actions.forEach(label => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label;
      b.onclick = () => sendMessage(label);
      qa.appendChild(b);
    });
    div.appendChild(qa);
  }
  if (role === "bot" && extra && Array.isArray(extra.sources) && extra.sources.length) {
    const sourceWrap = document.createElement("div");
    sourceWrap.className = "sources";
    const sourceTitle = document.createElement("div");
    sourceTitle.className = "source-title";
    sourceTitle.textContent = "Sources";
    sourceWrap.appendChild(sourceTitle);
    extra.sources.forEach(source => {
      const line = document.createElement(source.url ? "a" : "div");
      line.className = "source-item";
      if (source.url) { line.href = source.url; line.target = "_blank"; line.rel = "noopener noreferrer"; }
      const parts = [];
      if (source.citation) parts.push("[" + source.citation + "]");
      if (source.source_name) parts.push(source.source_name);
      if (source.chunk_index !== undefined && source.chunk_index !== null) parts.push("chunk " + source.chunk_index);
      if (source.score !== undefined) parts.push("relevance " + Math.round(Number(source.score) * 100) + "%");
      line.textContent = parts.join(" · ") || "Source";
      sourceWrap.appendChild(line);
    });
    div.appendChild(sourceWrap);
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
    if (sidebarEl && !sidebarEl.classList.contains("hidden")) renderHistory();
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
    let uploadMessage = data.message + "\n\nPreview: " + data.summary_preview;
    if (data.rag && data.rag.status === "failed") uploadMessage += "\n\nDocument preview is ready, but semantic indexing is unavailable: " + (data.rag.error || "try again later.");
    addMessage(uploadMessage, "bot", data);
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
  const avatarLink = document.getElementById("userAvatarLink");
  const avatarImg = document.getElementById("userAvatar");
  if (!loginBtn || !logoutBtn || !userLabel) return;
  try {
    const res = await fetch("/api/me");
    const me = await res.json();
    if (me.logged_in) {
      loginBtn.classList.add("hidden");
      logoutBtn.classList.remove("hidden");
      userLabel.textContent = me.display_name ? "Signed in as " + me.display_name : "Signed in";
      userLabel.classList.remove("hidden");
      if (avatarLink && avatarImg) {
        avatarImg.src = me.avatar_url || "/static/shibu.png";
        avatarLink.classList.remove("hidden");
      }
      if (me.onboarding_stage === "need_avatar") { window.location.href = "/avatar"; return; }
    } else {
      loginBtn.classList.remove("hidden");
      logoutBtn.classList.add("hidden");
      userLabel.classList.add("hidden");
      if (avatarLink) avatarLink.classList.add("hidden");
    }
  } catch (e) { /* header stays on Sign in; chat still works */ }
}
async function showWelcomeIfNeeded() {
  try {
    const res = await fetch("/api/onboarding/status");
    const data = await res.json();
    if (data.needs_welcome && data.welcome_message) {
      addMessage(data.welcome_message, "bot", null, false);
    }
  } catch (e) {}
}
document.getElementById("logoutBtn").onclick = async () => {
  await fetch("/auth/logout", { method: "POST" });
  await refreshLoginState();
};
refreshLoginState();

document.getElementById("detailsBtn").onclick = () => detailsEl.classList.toggle("hidden");
function resetExplainability() {
  detailsEl.classList.add("hidden");
  ["exIntent", "exRule", "exConf", "exAction", "exTool", "exTime"].forEach(id => {
    const node = document.getElementById(id);
    if (node) node.textContent = "—";
  });
}
const sidebarEl = document.getElementById("sidebar");
document.getElementById("sidebarBtn").onclick = () => {
  if (sidebarEl) { sidebarEl.classList.toggle("hidden"); if (!sidebarEl.classList.contains("hidden")) renderHistory(); }
};
document.getElementById("newChatBtn").onclick = async () => {
  await fetch("/api/clear", { method: "POST" });
  messagesEl.innerHTML = "";
  resetExplainability();
  await showWelcomeIfNeeded();
  if (!messagesEl.children.length) addMessage("Hello! I'm Shibu AI. How can I help you today?", "bot");
};
document.getElementById("clearBtn").onclick = async () => {
  await fetch("/api/clear", { method: "POST" });
  try { localStorage.removeItem("rb_chat_history"); } catch (e) {}
  messagesEl.innerHTML = "";
  resetExplainability();
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

// Restore saved chat if there is one. First-time logged-in users get
// the welcome question instead of the generic greeting; returning
// users (onboarding completed) get the generic greeting only when
// there is no restored chat, and skip onboarding entirely.
(async () => {
  let restored = false;
  try {
    const saved = JSON.parse(localStorage.getItem("rb_chat_history") || "[]");
    saved.forEach(m => addMessage(m.text, m.role, null, false));
    restored = saved.length > 0;
  } catch (e) {}
  let welcomed = false;
  try {
    const res = await fetch("/api/onboarding/status");
    const data = await res.json();
    if (data.needs_welcome && data.welcome_message) {
      addMessage(data.welcome_message, "bot", null, false);
      welcomed = true;
    }
  } catch (e) {}
  if (!restored && !welcomed) addMessage("Hello! I'm Shibu AI. How can I help you today?", "bot");
  renderHistory();
})();
