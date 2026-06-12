from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="gemma:2b", request_timeout=360.0)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


def initialize_chatbot():
    
    print("Loading business documents...")
    
    documents = SimpleDirectoryReader(input_files=["C:\\Users\\gstng\\Downloads\\ML_notes.pdf"]).load_data()
    
    print("Indexing data, please wait...")
    index = VectorStoreIndex.from_documents(documents)

    return index.as_query_engine()

if __name__ == "__main__":
    query_engine = initialize_chatbot()
    
    print("\n--- Chatbot Ready! (Type 'exit' to stop) ---")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            break
            
        response = query_engine.query(prompt)
        print(f"\nAI: {response}\n")