import subprocess
import time

models = ["slca"]  # Replace with your actual model names
# models = ["moe_adapters", "coda_prompt", "l2p", "dualprompt"]
datasets = ["seq-cifar100"]  # Replace with your actual dataset names
for model in models:
    for dataset in datasets:
        print(f"Running model: {model}")

        start_time = time.time()

        subprocess.run([
            "python", "main.py",
            "--dataset", dataset,
            "--model", model,
            "--model_config", "best",
            "--wandb_project", "Mammoth CIL",
            "--wandb_entity", "es22btech11013-iit-hyderabad",
            "--seed", "0",
            # "--debug_mode", "true",
            "--device", "3"
        ])

        end_time = time.time()
        print(f"Time taken for model {model}: {end_time - start_time:.2f} seconds")
