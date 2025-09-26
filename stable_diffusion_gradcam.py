#!/usr/bin/env python3

import os
import cv2
import numpy as np
import warnings
import argparse
import math
import random
import re
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# Diffusion model imports
from diffusers import DPMSolverMultistepScheduler

# Note: You may need to adjust this import based on your environment
# from models.stablediffusion import StableDiffusionPipeline
try:
    from models.stablediffusion import StableDiffusionPipeline
except ImportError:
    from diffusers import StableDiffusionPipeline
    print("Warning: Using standard diffusers StableDiffusionPipeline")

warnings.filterwarnings("ignore")


class StableDiffusionGradCAM:
    """
    Stable Diffusion Grad-CAM implementation for visualizing attention patterns.
    """
    
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: str = "cuda"):
        """
        Initialize the Grad-CAM analyzer.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_id = model_id
        self.gradients = {}
        self.activations = {}
        
        # Load the model
        self._load_model()
        
        # U-Net configuration for layer selection
        self.unet_config = {
            "up": [
                {"resnets": [1, 2, 3], "upsamplers": [1]},
                {"attentions": [1, 2, 3], "resnets": [1, 2, 3], "upsamplers": [1]},
                {"attentions": [1, 2, 3], "resnets": [1, 2, 3], "upsamplers": [1]},
                {"attentions": [1, 2, 3], "resnets": [1, 2, 3]},
            ],
            "down": [
                {"attentions": [1, 2], "resnets": [1, 2], "downsamplers": [1]},
                {"attentions": [1, 2], "resnets": [1, 2], "downsamplers": [1]},
                {"attentions": [1, 2], "resnets": [1, 2], "downsamplers": [1]},
                {"resnets": [1, 2]},
            ],
            "mid": [{"resnets": [1, 2]}]
        }
    
    def _load_model(self):
        """Load and configure the Stable Diffusion model."""
        print(f"Loading model: {self.model_id}")
        
        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Use DPM-Solver++ scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Configure for multi-GPU if available
        if torch.cuda.device_count() > 1:
            self.pipe = torch.nn.DataParallel(self.pipe).to(self.device)
            self.pipe = self.pipe.module
            
        self.pipe.unet.config.sample_size = 64
        
        print(f'Current cuda device: {torch.cuda.current_device()}')
        print(f'Count of using GPUs: {torch.cuda.device_count()}')
    
    def auto_device(self, obj=torch.device('cpu')):
        """Automatically move object to appropriate device."""
        if isinstance(obj, torch.device):
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            return obj.to('cuda')
        
        return obj
    
    def set_seed(self, seed: int) -> torch.Generator:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        gen = torch.Generator(device=self.auto_device())
        gen.manual_seed(seed)
        return gen
    
    def backward_hook(self, module, grad_input, grad_output):
        """Hook function to capture gradients during backward pass."""
        print("Capturing gradients...")
        self.gradients['value'] = grad_output[0]
    
    def forward_hook(self, module, input, output):
        """Hook function to capture activations during forward pass."""
        self.activations['value'] = output
    
    def register_hooks(self, block: str, layer: str, block_num: int, layer_num: int):
        """
        Register hooks on specified U-Net layer.
        
        Args:
            block: Block type ('down', 'up', 'mid')
            layer: Layer type ('attentions', 'resnets', 'downsamplers', 'upsamplers')
            block_num: Block number (0-indexed for down/up, ignored for mid)
            layer_num: Layer number within block (1-indexed)
        
        Returns:
            str: Layer message for identification
        """
        if block == "down":
            if layer == "attentions":
                self.pipe.unet.down_blocks[block_num].attentions[layer_num-1].register_forward_hook(self.forward_hook)
                self.pipe.unet.down_blocks[block_num].attentions[layer_num-1].register_backward_hook(self.backward_hook)
            elif layer == "resnets":
                self.pipe.unet.down_blocks[block_num].resnets[layer_num-1].register_forward_hook(self.forward_hook)
                self.pipe.unet.down_blocks[block_num].resnets[layer_num-1].register_backward_hook(self.backward_hook)
            elif layer == "downsamplers":
                self.pipe.unet.down_blocks[block_num].downsamplers[0].register_forward_hook(self.forward_hook)
                self.pipe.unet.down_blocks[block_num].downsamplers[0].register_backward_hook(self.backward_hook)
            else:
                raise ValueError("layer should be 'attentions', 'resnets', or 'downsamplers'")
        
        elif block == "up":
            if layer == "attentions":
                self.pipe.unet.up_blocks[block_num].attentions[layer_num-1].register_forward_hook(self.forward_hook)
                self.pipe.unet.up_blocks[block_num].attentions[layer_num-1].register_backward_hook(self.backward_hook)
            elif layer == "resnets":
                self.pipe.unet.up_blocks[block_num].resnets[layer_num-1].register_forward_hook(self.forward_hook)
                self.pipe.unet.up_blocks[block_num].resnets[layer_num-1].register_backward_hook(self.backward_hook)
            elif layer == "upsamplers":
                self.pipe.unet.up_blocks[block_num].upsamplers[0].register_forward_hook(self.forward_hook)
                self.pipe.unet.up_blocks[block_num].upsamplers[0].register_backward_hook(self.backward_hook)
            else:
                raise ValueError("layer should be 'attentions', 'resnets', or 'upsamplers'")
        
        elif block == "mid":
            if layer == "attentions":
                self.pipe.unet.mid_block.attentions[0].register_forward_hook(self.forward_hook)
                self.pipe.unet.mid_block.attentions[0].register_full_backward_hook(self.backward_hook)
            elif layer == "resnets":
                self.pipe.unet.mid_block.resnets[layer_num-1].register_forward_hook(self.forward_hook)
                self.pipe.unet.mid_block.resnets[layer_num-1].register_full_backward_hook(self.backward_hook)
            else:
                raise ValueError("layer should be 'attentions' or 'resnets'")
        else:
            raise ValueError("block should be 'down', 'up', or 'mid'")
        
        layer_msg = f"{block}Block-{block_num}_{layer}-{layer_num}"
        return layer_msg
    
    def generate_exponential_timesteps(self, num_steps: int, semantic_or_delicate: str = "semantic", 
                                     lamda: int = 30) -> torch.Tensor:
        """
        Generate exponential scheduler timesteps.
        
        Args:
            num_steps: Number of inference steps
            semantic_or_delicate: 'semantic' for early steps, 'delicate' for later steps
            lamda: Lambda parameter for exponential scheduling
        
        Returns:
            torch.Tensor: Timesteps tensor
        """
        alpha = math.e ** (math.log(1000) / (num_steps + lamda))
        
        if semantic_or_delicate == "semantic":
            time_steps = [1000 - int(alpha ** (i + 1 + lamda)) for i in range(num_steps)]
            time_steps.insert(0, 999)
            time_steps.pop(-1)
        elif semantic_or_delicate == "delicate":
            time_steps = [int(alpha ** (i + 1 + lamda)) for i in range(num_steps)]
            time_steps.sort(reverse=True)
            if time_steps[0] == 1000:
                time_steps[0] = 999
        else:
            raise ValueError("semantic_or_delicate should be 'semantic' or 'delicate'")
        
        time_steps = np.array(time_steps)
        time_steps = torch.from_numpy(time_steps).to(self.device)
        return time_steps
    
    def diffusion_gradcam(self, prompt_txt: str, seed: int = 42, num_steps: int = 30, 
                         step_vis: int = 15, layer_message: str = "", save: bool = False, 
                         time_steps: Optional[torch.Tensor] = None, 
                         output_dir: str = "./outputs") -> Tuple[List, torch.Tensor]:
        """
        Generate Grad-CAM visualization for diffusion process.
        
        Args:
            prompt_txt: Text prompt for generation
            seed: Random seed
            num_steps: Number of inference steps
            step_vis: Step number to visualize
            layer_message: Layer identification message
            save: Whether to save the visualization
            time_steps: Custom timesteps (optional)
            output_dir: Output directory for saved images
        
        Returns:
            Tuple of (all_images, saliency_map)
        """
        # Generate image with custom pipeline call
        # Note: This assumes a custom pipeline that supports the required parameters
        # You may need to modify this based on your specific pipeline implementation
        
        generator = self.set_seed(seed)
        
        # For standard diffusers pipeline, we'll use a simplified approach
        try:
            # Try custom pipeline call if available
            out, logit, all_images = self.pipe(
                prompt_txt,
                num_inference_steps=num_steps,
                generator=generator,
                get_images_for_all_inference_steps=True,
                output_type=None,
                step_visualization_num=step_vis,
                time_steps=time_steps,
                visualization_mode={'mode': "cam", 'mask': None, 'layer_vis': False}
            )
            logit.sample.sum().backward()
        except TypeError:
            # Fallback for standard pipeline
            print("Using standard pipeline - some features may be limited")
            out = self.pipe(
                prompt_txt,
                num_inference_steps=num_steps,
                generator=generator
            )
            # Create dummy logit for backward pass
            logit = torch.randn(1, requires_grad=True, device=self.device)
            logit.sum().backward()
            all_images = [out.images[0]] * num_steps
        
        guidance_scale = 7.5
        
        # Process gradients and activations
        if 'value' in self.gradients and 'value' in self.activations:
            grad_pred_uncond, grad_pred_text = self.gradients['value'].data.chunk(2)
            gradients_ = grad_pred_uncond + guidance_scale * (grad_pred_text - grad_pred_uncond)
            
            if "attentions" in layer_message:
                actv_pred_uncond, actv_pred_text = self.activations['value'].sample.chunk(2)
                activations_ = actv_pred_uncond + guidance_scale * (actv_pred_text - actv_pred_uncond)
            else:
                actv_pred_uncond, actv_pred_text = self.activations['value'].data.chunk(2)
                activations_ = actv_pred_uncond + guidance_scale * (actv_pred_text - actv_pred_uncond)
            
            b, k, u, v = activations_.size()
            
            # Compute Grad-CAM
            alpha = gradients_.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations_).sum(1, keepdim=True)
            
            h = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
            w = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
            
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            
            # Normalize saliency map
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
            
            mask = saliency_map.cpu().data
            mask = (mask - mask.min()).div(mask.max() - mask.min()).data
            heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
            
            # Process images
            if len(all_images) > step_vis - 1:
                noise_img = all_images[step_vis - 1]
            else:
                noise_img = all_images[0]
            
            org_img = out.images[0]
            
            # Convert to numpy arrays if they're PIL images
            if hasattr(noise_img, 'convert'):
                noise_img = np.array(noise_img.convert('RGB'))
            if hasattr(org_img, 'convert'):
                org_img = np.array(org_img.convert('RGB'))
            
            raw_noise_img = cv2.resize(noise_img, (h, w), interpolation=cv2.INTER_LINEAR)
            raw_org_img = cv2.resize(org_img, (h, w), interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensors
            noise_img = torch.from_numpy(np.ascontiguousarray(np.transpose(raw_noise_img, (2, 0, 1))))
            org_img = torch.from_numpy(np.ascontiguousarray(np.transpose(raw_org_img, (2, 0, 1))))
            
            # Normalize images
            transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            norm_noise_img = transform_norm(noise_img.float() / 255.0).unsqueeze(0)
            norm_org_img = transform_norm(org_img.float() / 255.0).unsqueeze(0)
            
            # Process heatmap
            b, g, r = heatmap.split(1)
            heatmap = torch.cat([r, g, b])
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Create CAM overlays
            noise_cam = 1 * (1 - mask ** 0.8) * noise_img.float() / 255.0 + (mask ** 0.8) * heatmap
            noise_gradcam = noise_cam.cpu().detach().squeeze(0)
            original_cam = 1 * (1 - mask ** 0.8) * org_img.float() / 255.0 + (mask ** 0.8) * heatmap
            original_gradcam = original_cam.cpu().detach().squeeze(0)
            
            # Visualization
            plt.figure('Grad CAM', figsize=(15, 5))
            plt.suptitle(f"{layer_message} | total step:{num_steps} | visualization:{step_vis}")
            
            plt.subplot(1, 3, 1)
            plt.imshow(noise_gradcam.permute(1, 2, 0).clamp(0, 1))
            plt.title("Noise Image + CAM")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(org_img.permute(1, 2, 0).float() / 255.0)
            plt.title("Generated Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(original_gradcam.permute(1, 2, 0).clamp(0, 1))
            plt.title("Generated Image + CAM")
            plt.axis("off")
            
            if save:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{layer_message}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to: {save_path}")
            
            plt.show()
            
            return all_images, saliency_map
        else:
            print("Warning: No gradients or activations captured")
            return [], torch.zeros(1, 1, 64, 64)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Stable Diffusion Grad-CAM Visualization")
    
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse",
                       help="Text prompt for image generation")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4",
                       help="HuggingFace model identifier")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--step_vis", type=int, default=15,
                       help="Step number to visualize")
    parser.add_argument("--block", type=str, default="mid", choices=["down", "up", "mid"],
                       help="U-Net block to analyze")
    parser.add_argument("--layer", type=str, default="resnets", 
                       choices=["attentions", "resnets", "downsamplers", "upsamplers"],
                       help="Layer type to analyze")
    parser.add_argument("--block_num", type=int, default=0,
                       help="Block number (0-indexed, ignored for mid block)")
    parser.add_argument("--layer_num", type=int, default=1,
                       help="Layer number within block (1-indexed)")
    parser.add_argument("--save", action="store_true",
                       help="Save visualization to file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for saved images")
    parser.add_argument("--exponential_schedule", action="store_true",
                       help="Use exponential timestep scheduling")
    parser.add_argument("--schedule_type", type=str, default="semantic", 
                       choices=["semantic", "delicate"],
                       help="Type of exponential scheduling")
    
    args = parser.parse_args()
    
    # Initialize Grad-CAM analyzer
    print("Initializing Stable Diffusion Grad-CAM...")
    gradcam = StableDiffusionGradCAM(model_id=args.model_id, device=args.device)
    
    # Register hooks on specified layer
    print(f"Registering hooks on {args.block} block, {args.layer} layer...")
    layer_message = gradcam.register_hooks(args.block, args.layer, args.block_num, args.layer_num)
    
    # Generate timesteps if using exponential scheduling
    time_steps = None
    if args.exponential_schedule:
        print("Using exponential timestep scheduling...")
        time_steps = gradcam.generate_exponential_timesteps(
            args.num_steps, args.schedule_type
        )
    
    # Run Grad-CAM analysis
    print(f"Generating Grad-CAM for prompt: '{args.prompt}'")
    all_images, saliency_map = gradcam.diffusion_gradcam(
        prompt_txt=args.prompt,
        seed=args.seed,
        num_steps=args.num_steps,
        step_vis=args.step_vis,
        layer_message=layer_message,
        save=args.save,
        time_steps=time_steps,
        output_dir=args.output_dir
    )
    
    print("Grad-CAM analysis completed!")
    print(f"Saliency map shape: {saliency_map.shape}")


if __name__ == "__main__":
    main()