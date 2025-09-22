"""
Parameter counting utility for all continual learning models.
This module provides functions to count total and trainable parameters
across different model architectures without missing any neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging


def count_model_parameters(model):
    """
    Count total and trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model or custom model object
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0
    
    try:
        # Handle standard PyTorch models
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
        # Handle SLCA_Model which has _network attribute
        elif hasattr(model, '_network') and hasattr(model._network, 'parameters'):
            for param in model._network.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
        # Handle custom model structures that might have get_parameters method
        elif hasattr(model, 'get_parameters'):
            try:
                params = model.get_parameters()
                for param in params:
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
            except:
                pass
                
        # If no parameters method found, try to find neural network components
        else:
            # Look for common neural network attributes
            for attr_name in dir(model):
                if not attr_name.startswith('_'):
                    attr = getattr(model, attr_name)
                    if hasattr(attr, 'parameters'):
                        try:
                            for param in attr.parameters():
                                total_params += param.numel()
                                if param.requires_grad:
                                    trainable_params += param.numel()
                        except:
                            continue
                            
    except Exception as e:
        logging.warning(f"Error counting parameters: {e}")
        
    return total_params, trainable_params


def count_continual_model_parameters(continual_model) -> Dict[str, Any]:
    """
    Comprehensive parameter counting for continual learning models.
    Handles different model architectures and multiple neural networks.
    
    Args:
        continual_model: The continual learning model instance
        
    Returns:
        Dictionary with parameter counts and model information
    """
    model_name = continual_model.__class__.__name__
    result = {
        'model_name': model_name,
        'total_params': 0,
        'trainable_params': 0,
        'components': {}
    }
    
    try:
        # Strategy: Count all parameters from the main model object
        # This ensures we don't miss any neural networks
        
        if hasattr(continual_model, 'net'):
            # Most models have their main network in 'net'
            main_net = continual_model.net
            total, trainable = count_model_parameters(main_net)
            result['total_params'] += total
            result['trainable_params'] += trainable
            result['components']['net'] = {'total': total, 'trainable': trainable}
            
            # For models like MoE-Adapters that have CLIP model inside net
            if hasattr(main_net, 'model'):
                # Check if this is a different model (like CLIP inside MoE-Adapters)
                if hasattr(main_net.model, 'named_parameters'):
                    clip_total, clip_trainable = count_model_parameters(main_net.model)
                    # Only add if it's different from the parent count (avoid double counting)
                    if clip_total != total:
                        result['components']['net.model'] = {'total': clip_total, 'trainable': clip_trainable}
        
        # For L2P which uses different structure
        if hasattr(continual_model, 'net') and hasattr(continual_model.net, 'model'):
            if model_name == 'L2P':
                l2p_total, l2p_trainable = count_model_parameters(continual_model.net.model)
                if 'net.model' not in result['components']:
                    result['components']['l2p_model'] = {'total': l2p_total, 'trainable': l2p_trainable}
        
        # For SLCA which uses _network
        if hasattr(continual_model, 'net') and hasattr(continual_model.net, '_network'):
            slca_total, slca_trainable = count_model_parameters(continual_model.net._network)
            result['total_params'] += slca_total
            result['trainable_params'] += slca_trainable
            result['components']['slca_network'] = {'total': slca_total, 'trainable': slca_trainable}
        
        # Safety check: if we didn't count anything, try counting the whole continual_model
        if result['total_params'] == 0:
            logging.warning(f"No parameters found in standard locations for {model_name}, counting entire model")
            total, trainable = count_model_parameters(continual_model)
            result['total_params'] = total
            result['trainable_params'] = trainable
            result['components']['full_model'] = {'total': total, 'trainable': trainable}
        
        # Additional specific checks for complex models
        if model_name == 'MoEAdapters':
            # Ensure we count CLIP model parameters correctly
            if hasattr(continual_model.net, 'model'):
                clip_model = continual_model.net.model
                clip_total, clip_trainable = count_model_parameters(clip_model)
                result['components']['clip_model'] = {'total': clip_total, 'trainable': clip_trainable}
                # Use the CLIP model as the main count
                result['total_params'] = clip_total
                result['trainable_params'] = clip_trainable
    
    except Exception as e:
        logging.error(f"Error counting parameters for {model_name}: {e}")
        # Fallback: count the entire model
        try:
            total, trainable = count_model_parameters(continual_model)
            result['total_params'] = total
            result['trainable_params'] = trainable
            result['components']['fallback'] = {'total': total, 'trainable': trainable}
        except Exception as e2:
            logging.error(f"Fallback parameter counting also failed for {model_name}: {e2}")
    
    return result


def log_parameter_count(continual_model, args, log_file_path: str = None):
    """
    Count and log parameters for a continual learning model.
    
    Args:
        continual_model: The continual learning model instance
        args: Arguments object containing model and dataset info
        log_file_path: Optional custom log file path
    """
    import os
    from datetime import datetime
    
    # Create log file path if not provided
    if log_file_path is None:
        log_folder = "parameter_logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(log_folder, f"parameters_{args.model}_{args.dataset}_{timestamp}.log")
    
    # Count parameters
    param_info = count_continual_model_parameters(continual_model)
    
    # Create log entry
    log_entry = f"""
================================================================================
Parameter Count Report
================================================================================
Model: {param_info['model_name']}
Dataset: {args.dataset}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY:
Total Parameters: {param_info['total_params']:,}
Trainable Parameters: {param_info['trainable_params']:,}
Non-trainable Parameters: {param_info['total_params'] - param_info['trainable_params']:,}

COMPONENTS:
"""
    
    for component, counts in param_info['components'].items():
        log_entry += f"  {component}: {counts['total']:,} total, {counts['trainable']:,} trainable\n"
    
    log_entry += "================================================================================\n"
    
    # Write to file
    with open(log_file_path, 'w') as f:
        f.write(log_entry)
    
    # Also log to console
    print(f"\nParameter Count for {param_info['model_name']}:")
    print(f"Total Parameters: {param_info['total_params']:,}")
    print(f"Trainable Parameters: {param_info['trainable_params']:,}")
    print(f"Non-trainable Parameters: {param_info['total_params'] - param_info['trainable_params']:,}")
    print(f"Details saved to: {log_file_path}")
    
    return param_info