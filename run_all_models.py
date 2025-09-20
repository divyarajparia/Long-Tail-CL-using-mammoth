import subprocess
import time
import argparse

def run_models(enable_logging=True):
    models = ["moe_adapters"]  # Replace with your actual model names
    # models = ["moe_adapters", "coda_prompt", "l2p", "dualprompt"]
    datasets = ["seq-cifar100"]  # Replace with your actual dataset names
    imb_factors = [0.01]  # Define your imbalance factors here

    for model in models:
        for dataset in datasets:
            for imb_factor in imb_factors:
                log_status = "with logging" if enable_logging else "without logging"
                print(f"Running model: {model}, dataset: {dataset}, imb_factor: {imb_factor} ({log_status})")

                start_time = time.time()

                # Build command arguments
                cmd_args = [
                    "python", "main.py",
                    "--dataset", dataset,
                    "--model", model,
                    "--model_config", "best",
                    "--wandb_project", "Mammoth CIL",
                    "--wandb_entity", "es22btech11013-iit-hyderabad",
                    "--seed", "0",
                    "--debug_mode", "true",
                    "--device", "3",
                    "--imb_factor", str(imb_factor),
                    "--enable_logging", str(enable_logging).lower()
                ]

                subprocess.run(cmd_args)

                end_time = time.time()
                print(f"Time taken for model {model} with imb_factor {imb_factor}: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all models with optional logging')
    parser.add_argument('--log', action='store_true', default=True,
                       help='Enable logging to dmr_logs folder (default: True)')
    parser.add_argument('--no-log', dest='log', action='store_false',
                       help='Disable logging to dmr_logs folder')

    args = parser.parse_args()
    run_models(args.log)
