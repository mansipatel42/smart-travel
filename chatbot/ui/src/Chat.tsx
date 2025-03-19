import { useState } from "react";
import { TextField, Button, Container, Paper, Typography } from "@mui/material";

interface Message {
  sender: "User" | "Bot";
  text: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { sender: "User", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://192.168.1.64:8000/chat/v1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input, stream: true }) // Enable streaming
      });

      if (!response.ok) throw new Error(`Error: ${response.statusText}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      let botMessage: Message = { sender: "Bot", text: "" };
      setMessages((prev) => [...prev, botMessage]); // Start bot message

      const decoder = new TextDecoder();
      let newText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        newText += decoder.decode(value, { stream: true });
        botMessage = { sender: "Bot", text: newText };
        setMessages((prev) => [...prev.slice(0, -1), botMessage]); // Update last message
      }
    } catch (error) {
      console.error("Error fetching response:", error);
    }
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} sx={{ p: 2, height: "80vh", display: "flex", flexDirection: "column" }}>
        <Typography variant="h5" gutterBottom>Chat</Typography>
        <div style={{ flexGrow: 1, overflowY: "auto", padding: "10px" }}>
          {messages.map((msg, index) => (
            <Typography key={index} align={msg.sender === "User" ? "right" : "left"}>
              <b>{msg.sender}:</b> {msg.text}
            </Typography>
          ))}
        </div>
        <TextField
          variant="outlined"
          fullWidth
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          sx={{ mt: 1 }}
        />
        <Button variant="contained" onClick={sendMessage} sx={{ mt: 1 }}>Send</Button>
      </Paper>
    </Container>
  );
}