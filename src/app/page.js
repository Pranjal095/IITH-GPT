'use client';
import { useState, useEffect } from 'react';
import { IoSend } from 'react-icons/io5';
import { RiRobot2Line } from 'react-icons/ri';
import { BsPersonFill } from 'react-icons/bs';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [titleEffect, setTitleEffect] = useState(false);

  // Add title animation effect
  useEffect(() => {
    // Initialize title animation
    if (typeof document !== 'undefined') {
      const style = document.createElement('style');
      style.textContent = `
        @keyframes pulse {
          0% { opacity: 0.85; transform: scale(0.98); }
          50% { opacity: 1; transform: scale(1.02); }
          100% { opacity: 0.85; transform: scale(0.98); }
        }
        
        @keyframes shimmer {
          0% { background-position: -100% 0; }
          100% { background-position: 200% 0; }
        }
        
        .title-glow {
          text-shadow: 0 0 10px rgba(156, 156, 255, 0.5), 
                      0 0 20px rgba(156, 156, 255, 0.3), 
                      0 0 30px rgba(156, 156, 255, 0.1);
          animation: pulse 3s infinite ease-in-out;
        }
        
        .gradient-text {
          background: linear-gradient(90deg, #9d9dff, #6e7efa, #9d9dff, #6e7efa);
          background-size: 300% 100%;
          animation: shimmer 6s infinite linear;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
      `;
      document.head.appendChild(style);
    }
    
    // Toggle title effect every 10 seconds
    const interval = setInterval(() => {
      setTitleEffect(prev => !prev);
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  // Rest of your existing code...
  const handleSend = async () => {
    if (userInput.trim() === "") return;

    // Add the user input to the chat
    const newMessages = [...messages, { text: userInput, sender: "user" }];
    setMessages(newMessages);
    setUserInput("");

    // Fetch the response from the server
    setLoading(true);
    try {
      const botResponse = await getBotResponse(userInput);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: botResponse, sender: "bot" },
      ]);
    } catch (error) {
      console.error("Error fetching bot response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Oops! Something went wrong. Please try again later.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getBotResponse = async (input) => {
    const response = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: input }),
    });

    if (!response.ok) {
      throw new Error("Failed to get response from server");
    }
    console.log(response);

    const data = await response.json();
    return data.response;
  };

  useEffect(() => {
    const chatBox = document.getElementById("chat-messages");
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);

  return (
    <div style={styles.container}>
      <div style={styles.chatContainer}>
        <div style={styles.header}>
          <div style={styles.logoContainer}>
            <RiRobot2Line style={styles.headerIcon} />
          </div>
          <h1 className={titleEffect ? 'title-glow gradient-text' : 'gradient-text'} style={styles.title}>
            IITH GPT
          </h1>
        </div>
        
        {/* Rest of your existing JSX code... */}
        <div id="chat-messages" style={styles.chatBox}>
          {messages.map((message, index) => (
            <div
              key={index}
              style={{
                ...styles.messageWrapper,
                justifyContent: message.sender === "user" ? "flex-end" : "flex-start",
              }}
            >
              {message.sender === "bot" && <div style={styles.botAvatar}><RiRobot2Line /></div>}
              <div
                style={{
                  ...styles.message,
                  ...styles[message.sender === "user" ? "userMessage" : "botMessage"],
                }}
              >
                {message.text}
              </div>
              {message.sender === "user" && <div style={styles.userAvatar}><BsPersonFill /></div>}
            </div>
          ))}
          {loading && (
            <div style={styles.messageWrapper}>
              <div style={styles.botAvatar}><RiRobot2Line /></div>
              <div style={{...styles.message, ...styles.botMessage, ...styles.typingIndicator}}>
                <div style={styles.dot}></div>
                <div style={styles.dot}></div>
                <div style={styles.dot}></div>
              </div>
            </div>
          )}
        </div>
        
        <div style={styles.inputContainer}>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Type your message..."
            style={styles.input}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={loading}
          />
          <button 
            onClick={handleSend} 
            style={{
              ...styles.button, 
              ...(loading || userInput.trim() === "" ? styles.buttonDisabled : {}),
            }} 
            disabled={loading || userInput.trim() === ""}
          >
            <IoSend size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    width: "100vw",
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    backgroundColor: "#0f0f13",
    color: "#f0f0f0",
    padding: "0",
    margin: "0",
  },
  chatContainer: {
    display: "flex",
    flexDirection: "column",
    width: "85%",
    maxWidth: "1000px",
    height: "90vh",
    borderRadius: "16px",
    overflow: "hidden",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.4)",
    background: "linear-gradient(145deg, #1a1a24 0%, #141419 100%)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    padding: "16px 20px",
    background: "linear-gradient(90deg, #1d1d2b, #18181f)",
    borderBottom: "1px solid #2e2e3a",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.1)",
  },
  logoContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "36px",
    height: "36px",
    borderRadius: "50%",
    background: "linear-gradient(135deg, #8c8dfc, #6e7efa)",
    boxShadow: "0 2px 8px rgba(140, 141, 252, 0.4)",
    marginRight: "12px",
    transition: "all 0.3s ease",
  },
  headerIcon: {
    fontSize: "20px",
    color: "#ffffff",
  },
  title: {
    margin: "0",
    fontSize: "22px",
    fontWeight: "700",
    letterSpacing: "0.5px",
    transition: "all 0.3s ease",
  },
  chatBox: {
    flex: 1,
    padding: "20px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
    scrollBehavior: "smooth",
    background: "linear-gradient(160deg, #181822 0%, #12121a 100%)",
  },
  messageWrapper: {
    display: "flex",
    alignItems: "flex-end",
    gap: "8px", 
    width: "100%",
  },
  message: {
    maxWidth: "70%",
    padding: "14px 18px",
    borderRadius: "18px",
    fontSize: "15px",
    lineHeight: "1.5",
    wordWrap: "break-word",
    position: "relative",
  },
  botMessage: {
    backgroundColor: "#2a2a38",
    color: "#f0f0f0",
    borderTopLeftRadius: "4px",
    boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
  },
  userMessage: {
    backgroundColor: "#5865f2",
    color: "#ffffff",
    borderTopRightRadius: "4px",
    boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
  },
  botAvatar: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    backgroundColor: "#2a2a38",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#8c8dfc",
    fontSize: "18px",
  },
  userAvatar: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    backgroundColor: "#5865f2",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#ffffff",
    fontSize: "16px",
  },
  inputContainer: {
    display: "flex",
    padding: "16px",
    background: "#1a1a24",
    borderTop: "1px solid #2e2e3a",
  },
  input: {
    flex: 1,
    padding: "12px 20px",
    border: "1px solid #2e2e3a",
    borderRadius: "24px",
    outline: "none",
    fontSize: "15px",
    backgroundColor: "#23232f",
    color: "#f0f0f0",
    transition: "border-color 0.3s",
    marginRight: "8px",
    boxShadow: "inset 0 1px 3px rgba(0, 0, 0, 0.1)",
  },
  button: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "44px",
    height: "44px",
    backgroundColor: "#5865f2",
    color: "white",
    border: "none",
    borderRadius: "50%",
    cursor: "pointer",
    transition: "all 0.2s ease",
    boxShadow: "0 2px 5px rgba(0, 0, 0, 0.2)",
  },
  buttonDisabled: {
    backgroundColor: "#3b3b4f",
    cursor: "not-allowed",
    opacity: 0.7,
  },
  typingIndicator: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    padding: "12px 18px",
    minWidth: "60px",
    minHeight: "20px",
  },
  dot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    backgroundColor: "#a0a0a0",
    animation: "bounce 1.5s infinite ease-in-out",
  },
};

if (typeof document !== "undefined") {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes bounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-6px); }
    }
  `;
  document.head.appendChild(style);
}
