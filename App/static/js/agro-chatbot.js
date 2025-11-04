// AgroBot Agriculture Chatbot with Claude API Integration
class AgroChatbot {
    constructor() {
        this.apiKey = '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';
        this.apiUrl = 'https://api.anthropic.com/v1/messages';
        this.isOpen = false;
        this.isTyping = false;
        this.messageHistory = [];
        this.sessionId = this.generateSessionId();
        this.conversationStarted = false;
        this.init();
    }

    init() {
        this.createChatbotHTML();
        this.setupEventListeners();
        this.loadMessageHistory();
        this.setupKeyboardShortcuts();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    createChatbotHTML() {
        // Check if chatbot already exists
        if (document.getElementById('agroChatbot')) return;

        const chatbotHTML = `
            <div id="agroChatbot" class="agro-chatbot">
                <!-- Chat Toggle Button -->
                <div id="chatbotToggle" class="chatbot-toggle">
                    <div class="chatbot-icon">
                        <i class="fas fa-comments"></i>
                        <span class="chat-badge">Online</span>
                    </div>
                </div>

                <!-- Chat Container -->
                <div id="chatbotContainer" class="chatbot-container">
                    <!-- Chat Header -->
                    <div class="chatbot-header">
                        <div class="chatbot-header-left">
                            <div class="chatbot-avatar">
                                <i class="fas fa-seedling"></i>
                            </div>
                            <div class="chatbot-info">
                                <h6 class="chatbot-name">AgroBot Assistant</h6>
                                <span class="chatbot-status">🌱 Agriculture Expert</span>
                            </div>
                        </div>
                        <div class="chatbot-header-right">
                            <button class="btn-icon" onclick="agroChatbot.clearChat()" title="Clear Chat">
                                <i class="fas fa-broom"></i>
                            </button>
                            <button class="btn-icon" onclick="agroChatbot.toggleChat()" title="Minimize">
                                <i class="fas fa-minus"></i>
                            </button>
                        </div>
                    </div>

                    <!-- Welcome Message -->
                    <div id="welcomeMessage" class="welcome-message">
                        <div class="welcome-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="welcome-content">
                            <h6>🌾 Welcome to AgroBot!</h6>
                            <p>I'm your agriculture assistant. Ask me about:</p>
                            <ul class="welcome-topics">
                                <li>🌱 Crop recommendations</li>
                                <li>🧪 Fertilizer advice</li>
                                <li>🍃 Plant diseases</li>
                                <li>💧 Irrigation tips</li>
                                <li>🌾 Farming best practices</li>
                            </ul>
                        </div>
                    </div>

                    <!-- Chat Messages -->
                    <div id="chatMessages" class="chat-messages"></div>

                    <!-- Typing Indicator -->
                    <div id="typingIndicator" class="typing-indicator" style="display: none;">
                        <div class="typing-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="typing-bubbles">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>

                    <!-- Quick Actions -->
                    <div id="quickActions" class="quick-actions">
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('What crops are best for my soil?')">
                            <i class="fas fa-seedling me-2"></i>
                            Crop Advice
                        </button>
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('How do I treat common plant diseases?')">
                            <i class="fas fa-microscope me-2"></i>
                            Disease Help
                        </button>
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('What fertilizer should I use?')">
                            <i class="fas fa-flask me-2"></i>
                            Fertilizer Tips
                        </button>
                        <button class="quick-action-btn" onclick="agroChatbot.quickQuestion('When should I plant crops?')">
                            <i class="fas fa-calendar me-2"></i>
                            Planting Time
                        </button>
                    </div>

                    <!-- Chat Input -->
                    <div class="chat-input-container">
                        <div class="chat-input-wrapper">
                            <textarea
                                id="chatInput"
                                class="chat-input"
                                placeholder="Ask me about farming..."
                                rows="1"
                                maxlength="500"
                            ></textarea>
                            <button id="sendButton" class="send-button" onclick="agroChatbot.sendMessage()">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        <div class="chat-input-footer">
                            <span class="input-hint">Press Enter to send, Shift+Enter for new line</span>
                            <span class="powered-by">Powered by Claude AI</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
        this.setupAnimations();
    }

    setupAnimations() {
        // Add CSS animations and transitions
        const style = document.createElement('style');
        style.textContent = `
            .agro-chatbot {
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 10000;
                font-family: 'Inter', sans-serif;
            }

            .chatbot-toggle {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(46, 125, 50, 0.3);
                transition: all 0.3s ease;
                position: relative;
            }

            .chatbot-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 30px rgba(46, 125, 50, 0.4);
            }

            .chatbot-toggle .chatbot-icon {
                font-size: 24px;
                color: white;
                transition: transform 0.3s ease;
            }

            .chatbot-toggle:hover .chatbot-icon {
                transform: rotate(10deg);
            }

            .chat-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: var(--accent-yellow);
                color: var(--dark-gray);
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: 600;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }

            .chatbot-container {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                opacity: 0;
                visibility: hidden;
                transform: translateY(20px) scale(0.9);
                transition: all 0.3s ease;
                overflow: hidden;
            }

            .chatbot-container.active {
                opacity: 1;
                visibility: visible;
                transform: translateY(0) scale(1);
            }

            .chatbot-header {
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-radius: 20px 20px 0 0;
            }

            .chatbot-header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .chatbot-avatar {
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            }

            .chatbot-name {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }

            .chatbot-status {
                font-size: 12px;
                opacity: 0.9;
            }

            .chatbot-header-right {
                display: flex;
                gap: 8px;
            }

            .btn-icon {
                width: 32px;
                height: 32px;
                background: rgba(255, 255, 255, 0.2);
                border: none;
                border-radius: 50%;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .btn-icon:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: scale(1.1);
            }

            .welcome-message {
                padding: 20px;
                background: linear-gradient(135deg, rgba(46, 125, 50, 0.1), rgba(76, 175, 80, 0.05));
                border-radius: 12px;
                margin: 20px;
                display: flex;
                gap: 15px;
                animation: slideInUp 0.5s ease;
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .welcome-avatar {
                width: 50px;
                height: 50px;
                background: var(--primary-green);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
                flex-shrink: 0;
            }

            .welcome-content h6 {
                margin: 0 0 10px 0;
                color: var(--primary-green);
                font-size: 16px;
                font-weight: 600;
            }

            .welcome-content p {
                margin: 0 0 10px 0;
                font-size: 14px;
                color: var(--medium-gray);
            }

            .welcome-topics {
                list-style: none;
                padding: 0;
                margin: 0;
                font-size: 13px;
                color: var(--medium-gray);
            }

            .welcome-topics li {
                margin-bottom: 3px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }

            .message {
                max-width: 85%;
                animation: messageSlide 0.3s ease;
            }

            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message.user {
                align-self: flex-end;
            }

            .message.bot {
                align-self: flex-start;
            }

            .message-bubble {
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
                position: relative;
            }

            .message.user .message-bubble {
                background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
                color: white;
                border-bottom-right-radius: 6px;
            }

            .message.bot .message-bubble {
                background: var(--light-gray);
                color: var(--dark-gray);
                border-bottom-left-radius: 6px;
            }

            .message-time {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.7);
                margin-top: 4px;
                text-align: right;
            }

            .message.bot .message-time {
                color: var(--medium-gray);
                text-align: left;
            }

            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                margin: 10px 20px;
            }

            .typing-avatar {
                width: 30px;
                height: 30px;
                background: var(--primary-green);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 14px;
            }

            .typing-bubbles {
                display: flex;
                gap: 4px;
            }

            .typing-bubbles span {
                width: 8px;
                height: 8px;
                background: var(--medium-gray);
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }

            .typing-bubbles span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .typing-bubbles span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }

            .quick-actions {
                padding: 10px 20px;
                border-top: 1px solid var(--light-gray);
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }

            .quick-action-btn {
                padding: 8px 12px;
                background: var(--white);
                border: 1px solid var(--light-gray);
                border-radius: 20px;
                font-size: 12px;
                color: var(--medium-gray);
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
                text-decoration: none;
            }

            .quick-action-btn:hover {
                background: var(--primary-green);
                color: white;
                border-color: var(--primary-green);
                transform: translateY(-2px);
            }

            .chat-input-container {
                padding: 20px;
                border-top: 1px solid var(--light-gray);
                background: var(--white);
                border-radius: 0 0 20px 20px;
            }

            .chat-input-wrapper {
                display: flex;
                gap: 10px;
                align-items: flex-end;
            }

            .chat-input {
                flex: 1;
                border: 2px solid var(--light-gray);
                border-radius: 20px;
                padding: 12px 16px;
                font-size: 14px;
                resize: none;
                max-height: 100px;
                transition: border-color 0.2s ease;
                font-family: inherit;
            }

            .chat-input:focus {
                outline: none;
                border-color: var(--primary-green);
            }

            .send-button {
                width: 40px;
                height: 40px;
                background: var(--primary-green);
                border: none;
                border-radius: 50%;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .send-button:hover:not(:disabled) {
                background: var(--secondary-green);
                transform: scale(1.1);
            }

            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .chat-input-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 8px;
                font-size: 11px;
                color: var(--medium-gray);
            }

            /* Mobile Responsive */
            @media (max-width: 480px) {
                .agro-chatbot {
                    bottom: 20px;
                    right: 20px;
                }

                .chatbot-container {
                    width: calc(100vw - 40px);
                    height: 70vh;
                    right: 20px;
                    bottom: 80px;
                }

                .chatbot-toggle {
                    width: 50px;
                    height: 50px;
                }

                .chatbot-toggle .chatbot-icon {
                    font-size: 20px;
                }

                .quick-actions {
                    grid-template-columns: 1fr;
                }
            }

            /* Loading and Error States */
            .error-message {
                background: rgba(244, 67, 54, 0.1);
                color: #d32f2f;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 14px;
                margin: 10px 20px;
                border-left: 3px solid #f44336;
            }

            .success-message {
                background: rgba(76, 175, 80, 0.1);
                color: #2e7d32;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 14px;
                margin: 10px 20px;
                border-left: 3px solid #4caf50;
            }
        `;

        document.head.appendChild(style);
    }

    setupEventListeners() {
        // Chat toggle
        const toggle = document.getElementById('chatbotToggle');
        if (toggle) {
            toggle.addEventListener('click', () => this.toggleChat());
        }

        // Send button
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendMessage());
        }

        // Enter key to send
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // Auto-resize textarea
            chatInput.addEventListener('input', () => {
                this.autoResizeTextarea(chatInput);
            });
        }

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.toggleChat();
            }
        });
    }

    setupKeyboardShortcuts() {
        // Ctrl/Cmd + / to open chat
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                if (!this.isOpen) {
                    this.toggleChat();
                }
                document.getElementById('chatInput')?.focus();
            }
        });
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
    }

    toggleChat() {
        const container = document.getElementById('chatbotContainer');
        const toggle = document.getElementById('chatbotToggle');

        this.isOpen = !this.isOpen;

        if (this.isOpen) {
            container.classList.add('active');
            toggle.style.display = 'none';
            document.getElementById('chatInput')?.focus();

            // Hide welcome message after first interaction
            const welcomeMsg = document.getElementById('welcomeMessage');
            if (welcomeMsg && this.conversationStarted) {
                welcomeMsg.style.display = 'none';
            }
        } else {
            container.classList.remove('active');
            toggle.style.display = 'flex';
        }
    }

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();

        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage(message, 'user');

        // Clear input
        input.value = '';
        this.autoResizeTextarea(input);

        // Mark conversation as started
        this.conversationStarted = true;

        // Hide welcome message
        const welcomeMsg = document.getElementById('welcomeMessage');
        if (welcomeMsg) {
            welcomeMsg.style.display = 'none';
        }

        // Hide quick actions
        const quickActions = document.getElementById('quickActions');
        if (quickActions) {
            quickActions.style.display = 'none';
        }

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await this.callClaudeAPI(message);
            this.hideTypingIndicator();
            this.addMessage(response, 'bot');
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again later.', 'bot', true);
            console.error('Chatbot API error:', error);
        }
    }

    async callClaudeAPI(message) {
        const systemPrompt = `You are AgroBot, a friendly and knowledgeable agriculture assistant. You help farmers and gardeners with:

        - Crop recommendations based on soil conditions, climate, and region
        - Fertilizer advice and soil management tips
        - Plant disease identification and treatment suggestions
        - Irrigation and water management guidance
        - Planting schedules and seasonal advice
        - Organic farming practices
        - Pest control methods
        - Modern farming technology guidance

        Guidelines:
        - Be helpful, encouraging, and practical
        - Ask clarifying questions if needed
        - Provide specific, actionable advice
        - Consider Indian agricultural context
        - Be concise but thorough
        - Use friendly, conversational tone
        - Include relevant emojis when appropriate 🌾🌱🧪💧

        Always respond in the same language as the user's message. If the message is in English, respond in English.`;

        const conversation = [
            {
                role: 'user',
                content: message
            }
        ];

        // Add some context if this is a new conversation
        if (this.messageHistory.length === 0) {
            conversation.unshift({
                role: 'user',
                content: 'Hello! I need help with farming advice.'
            });
            conversation.unshift({
                role: 'assistant',
                content: 'Hello! I\'m AgroBot, your agriculture assistant. I\'d be happy to help you with crop recommendations, fertilizer advice, plant disease management, irrigation tips, and any other farming-related questions. What would you like to know about? 🌾'
            });
        }

        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.apiKey,
                    'anthropic-version': '2023-06-01'
                },
                body: JSON.stringify({
                    model: 'claude-3-haiku-20240307',
                    max_tokens: 500,
                    system: systemPrompt,
                    messages: conversation,
                    temperature: 0.7
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.content[0].text;
        } catch (error) {
            console.error('Claude API Error:', error);
            throw error;
        }
    }

    addMessage(text, sender, isError = false) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const timestamp = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        const errorClass = isError ? ' error-message' : '';

        messageDiv.innerHTML = `
            <div class="message-bubble${errorClass}">
                ${text}
                <div class="message-time">${timestamp}</div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store in history
        this.messageHistory.push({
            text: text,
            sender: sender,
            timestamp: timestamp,
            isError: isError
        });

        // Save to localStorage
        this.saveMessageHistory();
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'flex';
            this.isTyping = true;
        }
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'none';
            this.isTyping = false;
        }
    }

    quickQuestion(question) {
        if (!this.isOpen) {
            this.toggleChat();
        }

        const input = document.getElementById('chatInput');
        if (input) {
            input.value = question;
            this.sendMessage();
        }
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            const messagesContainer = document.getElementById('chatMessages');
            if (messagesContainer) {
                messagesContainer.innerHTML = '';
            }

            this.messageHistory = [];
            this.conversationStarted = false;
            this.sessionId = this.generateSessionId();

            // Show welcome message again
            const welcomeMsg = document.getElementById('welcomeMessage');
            if (welcomeMsg) {
                welcomeMsg.style.display = 'flex';
            }

            // Show quick actions
            const quickActions = document.getElementById('quickActions');
            if (quickActions) {
                quickActions.style.display = 'grid';
            }

            // Show success message
            this.addMessage('Chat history cleared! How can I help you today?', 'bot');

            this.saveMessageHistory();
        }
    }

    saveMessageHistory() {
        try {
            localStorage.setItem('agroChatbotHistory', JSON.stringify({
                messages: this.messageHistory,
                sessionId: this.sessionId,
                timestamp: Date.now()
            }));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }

    loadMessageHistory() {
        try {
            const saved = localStorage.getItem('agroChatbotHistory');
            if (saved) {
                const data = JSON.parse(saved);

                // Check if session is recent (within 24 hours)
                const isRecent = Date.now() - data.timestamp < 24 * 60 * 60 * 1000;

                if (isRecent && data.messages && data.messages.length > 0) {
                    this.messageHistory = data.messages;
                    this.sessionId = data.sessionId;

                    // Restore messages (except the last bot message to show freshness)
                    const messagesToRestore = data.messages.slice(0, -1);
                    messagesToRestore.forEach(msg => {
                        this.addMessage(msg.text, msg.sender, msg.isError);
                    });

                    if (messagesToRestore.length > 0) {
                        this.conversationStarted = true;

                        // Hide welcome message if we have history
                        const welcomeMsg = document.getElementById('welcomeMessage');
                        if (welcomeMsg) {
                            welcomeMsg.style.display = 'none';
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
        }
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agroChatbot = new AgroChatbot();
});