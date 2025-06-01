from diffusers import LCMScheduler, DiffusionPipeline, StableDiffusionPipeline
import torch
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.attention_processor import Attention
from einops import rearrange

# 如果不能从 Huggingface 上下载检查点，手动下载到本地，然后更改 model_id_or_path 为本地目录
def load_model(model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7", device='cuda:0'):
    scheduler = LCMScheduler.from_config(model_id_or_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to(device)
    tokenizer, text_encoder, vae, unet = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet
    return pipe.scheduler, pipe, (tokenizer, text_encoder, vae, unet)

# Save the attention map of a specific layer of Unet
class AttnStore:
    def __init__(self, unet_attn_abbr_name=None):
        self.unet_attn_abbr_name = unet_attn_abbr_name
        self.unet_attns = []
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # store attention maps
        self.unet_attns.append(attention_probs)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def reset(self):
        self.unet_attns = []
     
    def visualize_cross_attn_maps(self, word_idx):
        attn_type = self.unet_attn_abbr_name.split('-')[-1]
        assert attn_type == 'cross'
        attn_maps = []
        timesteps = len(self.unet_attns)
        for attn_map in self.unet_attns:
            attn_map = rearrange(attn_map, '(b h) i t -> b h i t', h=8)
            attn_map = torch.mean(attn_map, dim=1)
            attn_map = torch.stack([attn_map[i, :, word_idx[i]] for i in range(len(word_idx))])
            bs, tokens = attn_map.shape
            height = width = int(tokens ** 0.5)
            attn_map = rearrange(attn_map, 'b (h w) -> b h w', h=height, w=width)
            attn_maps.append(attn_map)  # [t, b, h, w]
        rearange_attn_maps = []
        for i in range(bs):
            maps = torch.stack([attn_maps[j][i] for j in range(timesteps)])
            rearange_attn_maps.append(maps)
        rearange_attn_maps = torch.cat(rearange_attn_maps).unsqueeze(1)
        save_image(rearange_attn_maps, "attn_maps.png", nrow=timesteps, normalize=True, cmap='gray')
                   
class LCMSampler():
    def __init__(self, scheduler, pipe, tokenizer, text_encoder, vae, unet):
        self.scheduler = scheduler
        self.pipe = pipe
        self.tokenize = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
    
    def set_attn_processors(self, place_in_unet=None):
        unet_attn_names_abbr = [
            "1-self", "1-cross", "2-self", "2-cross", "4-self", "4-cross", "5-self", "5-cross", "7-self", "7-cross", "8-self", "8-cross",
            "16-self", "16-cross", "17-self", "17-cross", "18-self", "18-cross", "19-self", "19-cross", "20-self", "20-cross", "21-self", "21-cross", "22-self", "22-cross", "23-self", "23-cross", "24-self", "24-cross",
            "12-self", "12-cross"
            ]
        unet_attn_names = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
            "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",
            "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor",
            "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor",
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",
            "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor",
            "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor",
            "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor",
            "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor",
            "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor",
            "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor",
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
        ]
        unet_attn_map = dict(zip(unet_attn_names, unet_attn_names_abbr))
        
        self.attn_processors = {}
        setted_attn_processors = {}
        for k in self.unet.attn_processors.keys():
            unet_attn_abbr = unet_attn_map[k] 
            if unet_attn_abbr in place_in_unet:
                setted_attn_processors[k] = AttnStore(unet_attn_abbr)
                self.attn_processors[unet_attn_abbr] = setted_attn_processors[k]
            else:
                setted_attn_processors[k] = AttnProcessor()
                
        self.unet.set_attn_processor(setted_attn_processors)
    
    def reset_attn_processors(self):
        for ap in self.attn_processors.values():
            ap.reset()
    
    def get_attn_processors(self):
        return self.attn_processors
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__call__()
    @torch.no_grad()
    def lcm_sample(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * vae_scale_factor
        width = width or self.unet.config.sample_size * vae_scale_factor
            
        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        do_classifier_free_guidance = guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        
        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip
        )
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        for i, t in enumerate(tqdm(timesteps, desc='LCM Sampling')):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)
        
        if output_type == 'latent':
            return latents
        else:
            img_tensor = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
            do_denormalize = [True] * img_tensor.shape[0]
            img_output = self.pipe.image_processor.postprocess(img_tensor, output_type=output_type, do_denormalize=do_denormalize)
            if output_type == 'pt':
                save_image(img_output, "lcm_result.png", nrow=batch_size)
            elif output_type == 'pil':
                for i in range(len(img_output)):
                    img_output[i].save(f"lcm_result_{i}.png")
                
if __name__ == '__main__':
    scheduler, pipe, (tokenizer, text_encoder, vae, unet) = load_model()
    lcm = LCMSampler(scheduler, pipe, tokenizer, text_encoder, vae, unet)
    prompt = ['a photo of a corgi', 'a teddy bear sitting near a bird']
    negative_prompt = ['']
    place_in_unet = ['7-cross']
    lcm.set_attn_processors(place_in_unet)
    
    pils = lcm.lcm_sample(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=4,
        output_type='pt'
    )
    
    attn_processor = lcm.get_attn_processors()[place_in_unet[0]]
    attn_processor.visualize_cross_attn_maps([5, 2])