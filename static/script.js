
/* ------------------------------------
   FIXED TAB SWITCHING (NO INLINE STYLES)
-------------------------------------*/
function openTab(evt, tabName) {
    const contents = document.getElementsByClassName("tabcontent");
    const tabs = document.getElementsByClassName("tablink");

    // Hide all tab content
    for (let i = 0; i < contents.length; i++) {
        contents[i].style.display = "none";
    }

    // Remove active class from all tabs
    for (let i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove("active");
    }

    // Show selected tab
    document.getElementById(tabName).style.display = "block";

    // Add active class to selected tab
    evt.currentTarget.classList.add("active");
}

// Auto-click first tab
document.getElementsByClassName("tablink")[0].click();



/* ------------------------------------
        AUDIO EMOTION PREDICTION
-------------------------------------*/
async function predictAudio() {
    const fileInput = document.getElementById("audio_file");
    if (!fileInput.files.length) { alert("Upload a file"); return; }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch("/predict_audio", { method: "POST", body: formData });
    const data = await res.json();

    let html = `
<div class="result-card">

    <div class="result-title">🎧 Audio Emotion Predictions</div>

    <div class="result-section">
        <div class="result-label">RAVDESS</div>
        <span class="emotion-badge badge-green">${data.ravdess}</span>
    </div>

    <div class="result-section">
        <div class="result-label">TESS</div>
        <span class="emotion-badge badge-green">${data.tess.replace("OAF_", "")}</span>
    </div>

    <div class="result-section">
        <div class="result-label">SAVEE</div>
        <span class="emotion-badge badge-green">${data.savee}</span>
    </div>

    <div class="result-title" style="margin-top: 15px;">Ensemble Output</div>

    <div class="result-section">
        <span class="emotion-badge badge-green">${data.ensemble}</span>
    </div>

</div>
`;

    document.getElementById("audio_result").innerHTML = html;
}





/* ------------------------------------
        TEXT EMOTION PREDICTION
-------------------------------------*/
async function predictText() {
    const text = document.getElementById("text_input").value;
    if (!text.trim()) { alert("Enter text"); return; }

    const res = await fetch("/predict_text", {
        method: "POST",
        body: JSON.stringify({ text }),
        headers: { "Content-Type": "application/json" }
    });

    const data = await res.json();

    let single = data.single;
    let multi = data.multi;

    let html = `
<div class="result-box">
    <div class="result-title">🎯 Single Label Emotion</div>
    <span class="emotion-badge badge-green">${single}</span>

    <div class="result-title" style="margin-top:15px;">🎯 Multi-Label Emotions</div>
`;

    if (multi.length > 0) {
        multi.forEach(e => {
            html += `<span class="emotion-badge badge-green">${e}</span>`;
        });
    } else {
        html += `<div style="color:#666;">No strong emotions detected.</div>`;
    }

    html += `</div>`;

    document.getElementById("text_result").innerHTML = html;
}



/* ------------------------------------
         SUICIDE DETECTION
-------------------------------------*/
async function predictSuicide() {
    const text = document.getElementById("suicide_input").value;
    if (!text.trim()) { alert("Enter text"); return; }

    const res = await fetch("/predict_suicide", {
        method: "POST",
        body: JSON.stringify({ text }),
        headers: { "Content-Type": "application/json" }
    });

    const data = await res.json();

    let html = `
<div class="result-card">
    <div class="result-title">☠️ Suicide Risk Assessment</div>

    ${
        data.prediction 
        ? `<span class="emotion-badge badge-red">⚠️ Suicidal</span>`
        : `<span class="emotion-badge badge-green">✅ Non-Suicidal</span>`
    }
</div>`;

    document.getElementById("suicide_result").innerHTML = html;
}



/* ------------------------------------
              CHAT SYSTEM
-------------------------------------*/
let chat_history = [];

async function sendChat() {
    const msg = document.getElementById("chat_input").value;
    if (!msg.trim()) return;

    const res = await fetch("/therapist_chat", {
        method: "POST",
        body: JSON.stringify({ message: msg, history: chat_history }),
        headers: { "Content-Type": "application/json" }
    });

    const data = await res.json();

    chat_history = data.history;   // full history update

    renderChat();
    document.getElementById("chat_input").value = "";
}


function renderChat() {
    const box = document.getElementById("chat_box");
    box.innerHTML = "";

    chat_history.forEach(msg => {
        const div = document.createElement("div");
        const role = (msg.type === "human") ? "human" : "ai";

        div.className = "chat-message " + role;
        div.innerText = msg.content || "";

        box.appendChild(div);
    });

    box.scrollTop = box.scrollHeight;
}


function clearChat() {
    chat_history = [];
    const box = document.getElementById("chat_box");
    box.innerHTML = "";
    box.scrollTop = 0;
}



/* ------------------------------------
                AUDIO PREVIEW
-------------------------------------*/
function previewAudio() {
    const file = document.getElementById("audio_file").files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const audio = document.getElementById("audio_player");
    const source = document.getElementById("audio_source");

    source.src = url;
    audio.style.display = "block";
    audio.load();
}
function updateFileName() {
    const file = document.getElementById("audio_file").files[0];
    document.getElementById("file_name").innerText = file ? file.name : "No file chosen";
}
