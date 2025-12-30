const API_BASE_URL = (window.location.protocol === "http:" || window.location.protocol === "https:")
  ? ""
  : "http://127.0.0.1:5000";

// Initialize file input change handler
document.addEventListener("DOMContentLoaded", function() {
  const fileInput = document.getElementById("file");
  const uploadLabel = document.querySelector(".upload-text");
  
  fileInput.addEventListener("change", function(e) {
    if (e.target.files.length > 0) {
      uploadLabel.textContent = `Selected: ${e.target.files[0].name}`;
    }
  });
});

async function uploadDocument() {
  const fileInput = document.getElementById("file");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadStatus = document.getElementById("uploadStatus");
  const fileInfo = document.getElementById("fileInfo");
  
  if (!fileInput.files || fileInput.files.length === 0) {
    showStatus(uploadStatus, "Please select a file first.", "error");
    return;
  }
  
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  
  // Disable button and show loading
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="loading"></span> Uploading...';
  uploadStatus.className = "status-message";
  fileInfo.className = "file-info";
  
  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData
    });
    
    const result = await response.json();
    
    if (response.ok) {
      showStatus(uploadStatus, result.message || "File uploaded successfully!", "success");
      fileInfo.className = "file-info show";
      fileInfo.innerHTML = `
        <strong>File:</strong> ${result.filename || fileInput.files[0].name}<br>
        <strong>Characters:</strong> ${result.char_count?.toLocaleString() || 'N/A'}<br>
        <strong>Chunks:</strong> ${result.chunk_count?.toLocaleString() || 'N/A'}
      `;
      
      // Clear previous summary
      document.getElementById("summary").textContent = "";
      document.getElementById("summaryStats").className = "stats";
      
      // Notify chat about new document
      addMessageToChat(`Document "${result.filename || fileInput.files[0].name}" uploaded successfully! I'm ready to answer questions about it.`, false);
    } else {
      showStatus(uploadStatus, result.error || "Upload failed. Please try again.", "error");
    }
  } catch (error) {
    showStatus(uploadStatus, `Error: ${error.message}. Make sure the backend server is running.`, "error");
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = '<span>Upload Document</span>';
  }
}

async function getSummary() {
  const summaryBtn = document.getElementById("summaryBtn");
  const summaryDiv = document.getElementById("summary");
  const summaryStats = document.getElementById("summaryStats");
  
  summaryBtn.disabled = true;
  summaryBtn.innerHTML = '<span class="loading"></span> Generating...';
  summaryDiv.textContent = "";
  summaryStats.className = "stats";
  
  try {
    const response = await fetch(`${API_BASE_URL}/summary`);
    const data = await response.json();
    
    if (response.ok) {
      summaryDiv.textContent = data.summary || "No summary available.";
      
      if (data.original_length && data.summary_length) {
        const compressionRatio = ((1 - data.summary_length / data.original_length) * 100).toFixed(1);
        summaryStats.className = "stats show";
        summaryStats.innerHTML = `
          <strong>Original:</strong> ${data.original_length.toLocaleString()} characters | 
          <strong>Summary:</strong> ${data.summary_length.toLocaleString()} characters | 
          <strong>Compressed:</strong> ${compressionRatio}%
        `;
      }
    } else {
      summaryDiv.textContent = data.error || "Failed to generate summary.";
    }
  } catch (error) {
    summaryDiv.textContent = `Error: ${error.message}. Make sure the backend server is running.`;
  } finally {
    summaryBtn.disabled = false;
    summaryBtn.innerHTML = '<span>Generate Summary</span>';
  }
}

// Chat functionality
let chatHistory = [];

function addMessageToChat(message, isUser = false) {
  const chatMessages = document.getElementById("chatMessages");
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user-message" : "bot-message"}`;
  
  const avatar = isUser ? "👤" : "🤖";
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  messageDiv.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-text">${escapeHtml(message)}</div>
      <div class="message-time">${time}</div>
    </div>
  `;
  
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  return messageDiv;
}

function showTypingIndicator() {
  const chatMessages = document.getElementById("chatMessages");
  const typingDiv = document.createElement("div");
  typingDiv.className = "message bot-message";
  typingDiv.id = "typingIndicator";
  typingDiv.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="message-text typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  `;
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const typingIndicator = document.getElementById("typingIndicator");
  if (typingIndicator) {
    typingIndicator.remove();
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatCitations(citations) {
  if (!Array.isArray(citations) || citations.length === 0) {
    return "";
  }

  const items = citations.map((c) => {
    const src = c.source ? escapeHtml(c.source) : "document";
    const pagePart = (c.page !== null && c.page !== undefined) ? ` • p.${escapeHtml(String(c.page))}` : "";
    const scorePart = (c.score !== null && c.score !== undefined) ? ` • score ${Number(c.score).toFixed(3)}` : "";
    const chunkPart = c.chunk_id ? ` • ${escapeHtml(String(c.chunk_id))}` : "";
    const excerpt = c.excerpt ? escapeHtml(c.excerpt) : "";

    return `
      <div class="citation-item">
        <div class="citation-meta">
          <strong>${src}</strong>${pagePart}${chunkPart}${scorePart}
        </div>
        <div class="citation-excerpt">${excerpt}</div>
      </div>
    `;
  }).join("");

  return `
    <div class="citations">
      <div class="citations-title">Sources</div>
      ${items}
    </div>
  `;
}

async function sendChatMessage() {
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");
  const question = chatInput.value.trim();
  
  if (!question) {
    return;
  }
  
  // Add user message to chat
  addMessageToChat(question, true);
  chatHistory.push({ role: "user", content: question });
  
  // Clear input and disable button
  chatInput.value = "";
  chatSendBtn.disabled = true;
  chatSendBtn.innerHTML = '<span class="loading"></span>';
  
  // Show typing indicator
  showTypingIndicator();
  
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ 
        question: question,
        history: chatHistory.slice(-5) // Send last 5 messages for context
      })
    });
    
    const data = await response.json();
    
    removeTypingIndicator();
    
    if (response.ok) {
      const answer = data.answer || "I couldn't find an answer to that question.";
      const citationsHtml = formatCitations(data.citations);
      const messageDiv = addMessageToChat(answer, false);
      if (citationsHtml) {
        const contentDiv = messageDiv.querySelector(".message-content");
        if (contentDiv) {
          contentDiv.insertAdjacentHTML("beforeend", citationsHtml);
        }
      }
      chatHistory.push({ role: "assistant", content: answer });
    } else {
      const errorMsg = data.error || "Failed to get response. Please make sure a document is uploaded.";
      addMessageToChat(errorMsg, false);
    }
  } catch (error) {
    removeTypingIndicator();
    const errorMsg = `Error: ${error.message}. Make sure the backend server is running.`;
    addMessageToChat(errorMsg, false);
  } finally {
    chatSendBtn.disabled = false;
    chatSendBtn.innerHTML = '<span>Send</span>';
    chatInput.focus();
  }
}

function handleChatKeyPress(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

function clearChat() {
  const chatMessages = document.getElementById("chatMessages");
  chatHistory = [];
  
  // Keep only the welcome message
  chatMessages.innerHTML = `
    <div class="message bot-message">
      <div class="message-avatar">🤖</div>
      <div class="message-content">
        <div class="message-text">
          Hello! I'm your AI documentation assistant. Upload a document and I'll help you understand it better. Ask me anything about the document!
        </div>
        <div class="message-time">Just now</div>
      </div>
    </div>
  `;
}

function showStatus(element, message, type) {
  element.textContent = message;
  element.className = `status-message ${type}`;
  
  // Auto-hide success messages after 5 seconds
  if (type === "success") {
    setTimeout(() => {
      element.className = "status-message";
    }, 5000);
  }
}
