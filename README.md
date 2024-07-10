# LLM Word Game Arena
 
### Prerequisites
1. [Python](https://www.python.org/downloads/)
2. [Ollama](https://github.com/ollama/ollama/tree/main)

### Instructions
1. Clone the repository
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Download the models
```bash
ollama pull llama3:8b-instruct-q8_0
ollama pull gemma2:9b-instruct-q8_0 
```
4. Run the script
```bash
python main.py
```