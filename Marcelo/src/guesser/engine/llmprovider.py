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
