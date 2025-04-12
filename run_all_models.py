import subprocess

models = ["l2p", "dualprompt", "coda_prompt", "slca", "moe_adapters"]  # Replace with your actual model names

for model in models:
    print(f"Running model: {model}")
    subprocess.run([
        "python", "main.py",
        "--dataset", "seq-cifar100",
        "--model", model,
        "--model_config", "best",
        "--wandb_project", "Mammoth CIL",
        "--wandb_entity", "es22btech11013-iit-hyderabad",
        "--seed", "0"
        # "--debug_mode", "true",
    ])
