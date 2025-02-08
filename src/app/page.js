'use client';
import { useState, useEffect } from 'react';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);

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
    const chatBox = document.querySelector("[style*='overflowY: scroll']");
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);

  return (
    <div style={styles.container}>
      <div style={styles.chatBox}>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.message,
              alignSelf: message.sender === "user" ? "flex-end" : "flex-start",
              backgroundColor: message.sender === "user" ? "#d1f7d6" : "#f0f0f0",
            }}
          >
            {message.text}
          </div>
        ))}
        {loading && (
          <div style={{ ...styles.message, alignSelf: "flex-start" }}>
            Typing...
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
        <button onClick={handleSend} style={styles.button} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}

// const styles = { /* Unchanged */ };


const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    width: "100vw",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#f9f9f9",
    padding: "0",
    margin: "0",
  },
  chatBox: {
    width: "80vw",
    height: "80vh",
    border: "1px solid #ccc",
    borderRadius: "8px",
    padding: "10px",
    overflowY: "scroll",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    backgroundColor: "#ffffff",
  },
  message: {
    maxWidth: "70%",
    padding: "10px",
    borderRadius: "10px",
    fontSize: "16px",
    wordWrap: "break-word",
  },
  inputContainer: {
    display: "flex",
    width: "80vw",
    marginTop: "10px",
  },
  input: {
    flex: 1,
    padding: "15px",
    border: "1px solid #ccc",
    borderRadius: "8px 0 0 8px",
    outline: "none",
    fontSize: "16px",
  },
  button: {
    padding: "15px 20px",
    backgroundColor: "#0070f3",
    color: "white",
    border: "none",
    borderRadius: "0 8px 8px 0",
    cursor: "pointer",
    fontSize: "16px",
  },
};
