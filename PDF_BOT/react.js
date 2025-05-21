import React, { useState } from "react";
import { Container, TextField, Button, Box, Typography, Paper } from "@mui/material";

const ChatInterface = () => {
  const [chat, setChat] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Append user message
    setChat(prev => [...prev, { sender: "User", message: input }]);
    
    // Call your backend API for chatbot response (simulate with timeout)
    setTimeout(() => {
      const botResponse = "This is a simulated response from the chatbot.";
      setChat(prev => [...prev, { sender: "Chatbot", message: botResponse }]);
    }, 1000);
    
    setInput("");
  };

  return (
    <Container maxWidth="sm">
      <Typography variant="h4" align="center" gutterBottom>
        PDF Chatbot
      </Typography>
      <Paper style={{ height: 300, overflowY: "scroll", padding: 10, marginBottom: 20 }}>
        {chat.map((msg, index) => (
          <Box key={index} marginY={1}>
            <Typography variant="body1" color={msg.sender === "User" ? "primary" : "secondary"}>
              {msg.sender}: {msg.message}
            </Typography>
          </Box>
        ))}
      </Paper>
      <TextField
        fullWidth
        variant="outlined"
        placeholder="Enter your question..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <Button variant="contained" color="primary" onClick={sendMessage} style={{ marginTop: 10 }}>
        Send
      </Button>
    </Container>
  );
};

export default ChatInterface;
