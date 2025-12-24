import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import styles from './LiveChatWidget.module.css';

const LiveChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! How can I help you today?', sender: 'agent', timestamp: new Date() },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isAgentTyping, setIsAgentTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isOpen]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() === '') return;

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    // Simulate agent typing
    setIsAgentTyping(true);
    setTimeout(() => {
      const agentMessage = {
        id: messages.length + 2,
        text: 'Thanks for your message. This is a simulated response.',
        sender: 'agent',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, agentMessage]);
      setIsAgentTyping(false);
    }, 1500);
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={styles.liveChatContainer}>
      {/* Chat button */}
      {!isOpen && (
        <button
          className={clsx(styles.chatButton, styles.floating)}
          onClick={toggleChat}
          aria-label="Open chat"
          id="live-chat-button"
        >
          💬
        </button>
      )}

      {/* Chat window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <div className={styles.chatHeaderContent}>
              <h4>Live Chat</h4>
              <button
                className={styles.closeButton}
                onClick={toggleChat}
                aria-label="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className={styles.chatMessages}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={clsx(
                  styles.message,
                  styles[message.sender],
                  message.sender === 'user' ? styles.userMessage : styles.agentMessage
                )}
              >
                <div className={styles.messageContent}>
                  {message.text}
                </div>
                <div className={styles.messageTime}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}

            {isAgentTyping && (
              <div className={clsx(styles.message, styles.agentMessage)}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSendMessage} className={styles.chatInputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              className={styles.chatInput}
              disabled={isAgentTyping}
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={inputValue.trim() === '' || isAgentTyping}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default LiveChatWidget;