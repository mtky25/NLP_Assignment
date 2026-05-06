from llama_index.llms.ollama import Ollama


def get_llm(model_name="llama3.2", temperature=0.1,timeout=30.0):

    return Ollama(
        model=model_name, 
        request_timeout=timeout,
        temperature=temperature,
        context_window=4096, 
        additional_kwargs={"num_ctx": 4096} 
    )