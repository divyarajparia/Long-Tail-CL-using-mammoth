# TinyImageNet Dataset Configurations

This document shows how to use the new task configurations for the TinyImageNet dataset.

## Available Configurations

### 5tasks.yaml
- **N_TASKS**: 5
- **N_CLASSES_PER_TASK**: 20
- **N_CLASSES**: 100

### 10tasks.yaml
- **N_TASKS**: 10
- **N_CLASSES_PER_TASK**: 10
- **N_CLASSES**: 100
- **imb_factor**: 0.1

## Usage Examples

### Using 5-task configuration:
```bash
python main.py --model <model_name> --dataset seq-tinyimg --dataset_config 5tasks
```

### Using 10-task configuration:
```bash
python main.py --model <model_name> --dataset seq-tinyimg --dataset_config 10tasks
```

### Using default configuration (5 tasks):
```bash
python main.py --model <model_name> --dataset seq-tinyimg
```

## Configuration Details

The configurations follow the same pattern as CIFAR-100:
- **5tasks**: Splits 100 classes into 5 tasks with 20 classes each
- **10tasks**: Splits 100 classes into 10 tasks with 10 classes each, with imbalance factor
- **default**: Same as 5tasks configuration

These configurations will be automatically picked up by the pipeline and applied to the SequentialTinyImagenet dataset.