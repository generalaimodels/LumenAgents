
# 🚀 Advanced AI Assistant CLI  

## 🧠 Overview  
**Advanced AI Assistant CLI** is a powerful tool built using cutting-edge language models (LLMs). It provides a rich suite of features for interactive AI-driven tasks such as text generation, code generation, Linux interactions, API searches, and web search synthesis.

This project utilizes a modular architecture, allowing for seamless integration and scalability while enhancing usability through a command-line interface (CLI).

## ✨ Key Features  
- **Text Generation Pipeline:** Intelligent responses using contextual history.  
- **Code Generation & Execution:** Generate and execute Python code dynamically.  
- **Linux System Interaction:** Perform OS-level commands and operations.  
- **API Search Agent:** Search and process data from external APIs.  
- **Web Search & Synthesis:** Intelligent synthesis of web search results.  
- **Self-Prompting:** Generate fine-tuned prompts from user input.  
- **Command History Management:** View, export, and import command history.  

## 📂 Project Structure  
- `history_manager.py`: Manages interaction history.  
- `Agent_tool.py`: Implements code generation, web search, and API search.  
- `rag_pipeline.py`: Handles retrieval-augmented generation tasks.  
- `main.py`: Main entry point for the interactive CLI.

## 📦 Installation  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/generalaimodels/LumenAgents.git
   cd LumenAgents
   ```  



3. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  

## 🛠 Usage  

### Running the Assistant  
Launch the CLI tool:  
```bash
python chatbot_copy.py
```  

### Available Commands  
| Command           | Description                                   |  
|--------------------|-----------------------------------------------|  
| `generate`         | Generate text based on a query.              |  
| `history`          | View command generation history.             |  
| `export`           | Export history to a file.                   |  
| `import`           | Import history from a file.                 |  
| `api_search`       | Perform an API search.                       |  
| `os_interact`      | Interact with the operating system.          |  
| `code_gen`         | Generate and execute Python code.            |  
| `web_search`       | Perform a web search and summarize results.  |  
| `selfprompting`    | Generate fine-tuned prompts.                 |  
| `help`             | Show available commands.                    |  
| `exit`             | Exit the assistant.                         |  

### Example Commands  
1. **Text Generation:**  
   ```bash
   Enter your query: Explain how neural networks work.
   ```  

2. **Code Generation:**  
   ```bash
   Enter code generation prompt: Write a function to compute Fibonacci sequence.
   ```  

3. **Web Search:**  
   ```bash
   Enter web search query: Latest trends in AI research.
   ```  

## 📝 Logs & History  
- Interaction history is automatically saved as a JSON file (`final_state.json`).  
- Use `export` and `import` commands to manage history.  

## 🤝 Contributing  
Contributions are welcome! To contribute:  



## 🌟 Acknowledgements  
Special thanks to [OpenAI](https://openai.com) for the foundational technologies that power this project.  

NOTE: "gpt4free" serves as a PoC (proof of concept), demonstrating the development of an API package with multi-provider requests, with features like timeouts, load balance and flow control. more info go to  [gpt4free](https://github.com/xtekky/gpt4free.git)
---  
Built with ❤️ and advanced LLMs.
```  