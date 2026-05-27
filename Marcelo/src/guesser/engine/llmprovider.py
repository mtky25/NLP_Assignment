import requests
from llama_index.llms.ollama import Ollama


def get_llm(model_name="llama3.2", temperature=0.1, timeout=120.0, num_ctx=4096, num_predict=128, stop=["\n", "\n\n"], keep_alive=-1):

    return Ollama(
        model=model_name, 
        request_timeout=timeout,
        temperature=temperature,
        context_window=num_ctx, 
        keep_alive=keep_alive,
        additional_kwargs={
            "num_ctx": num_ctx, 
            "num_predict": num_predict,
            "stop": stop
        } 
    )

def unload_model(model_name: str):
    """
    Explicitly unloads a model from Ollama's memory to free up RAM.
    """
    if not model_name:
        return
        
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "keep_alive": 0}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print(f" [Ollama] Successfully unloaded model: {model_name}")
        else:
            print(f" [Ollama] Failed to unload model: {model_name} (Status: {response.status_code})")
    except Exception as e:
        print(f" [Ollama] Error unloading model {model_name}: {e}")