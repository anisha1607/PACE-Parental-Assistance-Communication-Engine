const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");

let isLoading = false;
let sessionId = Math.random().toString(36).substring(2, 11);

function esc(str) {
  return String(str ?? "").replace(/[&<>"']/g, s => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[s]));
}

function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage({ role, text, metaChips = [], refused = false }) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}${refused ? " refused" : ""}`;

  const content = document.createElement("div");
  content.className = "msg-content";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "P";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  content.appendChild(avatar);
  content.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.innerHTML = `<span>${esc(nowTime())}</span>` + metaChips.map(c => c).join("");

  msg.appendChild(content);
  msg.appendChild(meta);

  chatEl.appendChild(msg);
  scrollToBottom();
}

function chip(text, kind = "good") {
  const kindClass = kind === "good" ? "chip-ok" : kind === "warn" ? "chip-warn" : "chip-bad";
  return `<span class="status-chip ${kindClass}">${esc(text)}</span>`;
}

function setLoading(v) {
  isLoading = v;
  sendBtn.disabled = v;
  inputEl.disabled = v;
}

function showTyping(show) {
  const existing = document.getElementById("typing");
  if (show) {
    if (existing) return;
    const wrap = document.createElement("div");
    wrap.id = "typing";
    wrap.className = "typing-indicator";
    wrap.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
    chatEl.appendChild(wrap);
    scrollToBottom();
  } else {
    if (existing) existing.remove();
  }
}

async function send() {
  const situation = inputEl.value.trim();
  if (!situation || isLoading) return;

  addMessage({ role: "user", text: situation });
  inputEl.value = "";
  inputEl.style.height = "auto";

  setLoading(true);
  showTyping(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ situation, session_id: sessionId })
    });

    const data = await res.json();
    showTyping(false);

    const g = data.guard_label ?? "UNKNOWN";
    const conf = typeof data.guard_confidence === "number" ? data.guard_confidence : null;
    const refused = !!data.refused;

    const chips = [];
    if (refused) {
      const confText = conf === null ? "" : ` ${conf.toFixed(2)}`;
      chips.push(chip(`guard=${g}${confText}`, "bad"));
      chips.push(chip("REFUSED", "bad"));
    }

    addMessage({
      role: "pace",
      text: (data.response ?? "").trim(),
      metaChips: chips,
      refused
    });

  } catch (err) {
    showTyping(false);
    addMessage({
      role: "pace",
      text: "Sorry — something went wrong with the connection.",
      metaChips: [chip("ERROR", "bad")],
      refused: true
    });
  } finally {
    setLoading(false);
    inputEl.focus();
  }
}

function clearChat() {
  chatEl.innerHTML = "";
  sessionId = Math.random().toString(36).substring(2, 11);
  addMessage({
    role: "pace",
    text: `Hi, I am PACE. Tell me what you are noticing with your child, and I will help you find a calm way to respond.`,
    metaChips: [chip("ready", "good")]
  });
}

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
});

sendBtn.addEventListener("click", send);
clearBtn.addEventListener("click", clearChat);

// init
clearChat();