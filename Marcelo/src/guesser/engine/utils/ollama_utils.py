import subprocess

def get_ollama_model_sizes():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return {}

        model_sizes = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                size = parts[2] + " " + parts[3] if len(parts) > 3 and parts[3] in ["GB", "MB", "KB"] else parts[2]
                model_sizes[name] = size
        return model_sizes
    except Exception as e:
        print(f"Warning: Could not fetch model sizes from Ollama: {e}")
        return {}

def populate_experiment_config_sizes(config):
    sizes = get_ollama_model_sizes()

    if config.inference_model and config.inference_model in sizes:
        config.inference_model_size = sizes[config.inference_model]
    elif config.inference_model:
        if ":" not in config.inference_model:
            latest_name = f"{config.inference_model}:latest"
            if latest_name in sizes:
                config.inference_model_size = sizes[latest_name]

    if config.embedding_model and config.embedding_model in sizes:
        config.embedding_model_size = sizes[config.embedding_model]
    elif config.embedding_model:
        if ":" not in config.embedding_model:
            latest_name = f"{config.embedding_model}:latest"
            if latest_name in sizes:
                config.embedding_model_size = sizes[latest_name]

    return config
