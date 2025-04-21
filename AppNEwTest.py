import os
import time
import random
import torch
from PIL import Image
import numpy as np
import gradio as gr
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler
)
#from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer



# Optional upscalers
try:
    from realesrgan.utils import RealESRGANer 
    from gfpgan.utils import GFPGANer
    UPSCALERS_AVAILABLE = True
except ImportError:
    UPSCALERS_AVAILABLE = True

try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False

# -------- Configuration --------
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
WEIGHTS_DIR = "weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

dtype = 'torch.float16'

# -------- Utilities --------

def get_best_device(device_choice):
    if device_choice.lower() == "cuda" and torch.cuda.is_available():
        return "cuda", torch.float16
    elif device_choice.lower() == "directml":
        if not DML_AVAILABLE:
            raise RuntimeError("DirectML selected but torch-directml not installed.")
        return torch_directml.device(), torch.float16
    else:
        return "cpu", torch.float32

def preprocess_image(img, size):
    return img.resize(size)

def scan_model_folder():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith((".safetensors", ".ckpt"))]

def apply_upscaling(image, device="cpu", method="realesrgan"):
    if method == "lanczos":
        new_size = (image.width * 4, image.height * 4)
        return image.resize(new_size, Image.LANCZOS)

    if not UPSCALERS_AVAILABLE:
        raise RuntimeError("RealESRGAN not installed.")
    model = RealESRGANer(device, scale=4)
    model.load_weights(os.path.join(WEIGHTS_DIR, "RealESRGAN_x4.pth"))
    return model.predict(np.array(image))

def apply_face_enhancement(image, device="cpu"):
    if not UPSCALERS_AVAILABLE:
        raise RuntimeError("GFPGAN not installed.")

    device_str = "cuda" if "cuda" in str(device).lower() else "cpu"

    gfpgan = GFPGANer(
        model_path=os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth"),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        device=device_str
    )

    _, _, output = gfpgan.enhance(
        np.array(image),
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    return Image.fromarray(output)

# -------- Pipeline Management --------

pipeline_cache = {}

class UnifiedPipeline:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_path, task, scheduler_name="ddpm", device="device"):
        pipe_params = {
            "pretrained_model_link_or_path": model_path,
            "torch_dtype": torch.float16,
            "safety_checker": None,
            "use_safetensors": model_path.endswith(".safetensors")
        }

        if task == "img2img":
            self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(**pipe_params)
        else:
            self.pipe = StableDiffusionPipeline.from_single_file(**pipe_params)

        self._configure_scheduler(scheduler_name)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _configure_scheduler(self, scheduler_name):
        scheduler_map = {
            'ddpm': DDPMScheduler,
            'pndm': PNDMScheduler,
            'lms': LMSDiscreteScheduler,
            'euler': EulerDiscreteScheduler,
            'ddim': DDIMScheduler
        }
        self.pipe.scheduler = scheduler_map[scheduler_name].from_config(
            self.pipe.scheduler.config,
            prediction_type="epsilon"
        )

# -------- Core Logic --------

def generate_image(
    prompt,
    uploaded_image,
    strength,
    guidance_scale,
    seed,
    model_dropdown,
    device_choice,
    task,
    scheduler_type,
    steps,
    width,
    height,
    negative_prompt,
    batch_count,
    lock_aspect,
    upscale_method,
    enable_facefix,
    progress=gr.Progress(track_tqdm=True)
):
    model_path = os.path.join(MODEL_DIR, model_dropdown)
    if not os.path.exists(model_path):
        return None, "❌ Model not found."

    try:
        cache_key = (model_path, task, str(dtype), scheduler_type)
        device, _ = get_best_device(device_choice)

        if cache_key not in pipeline_cache:
            pipeline = UnifiedPipeline()
            pipeline.load_model(
                model_path=model_path,
                task=task,
                scheduler_name=scheduler_type,
                device=device
            )
            pipeline_cache[cache_key] = pipeline
        pipe = pipeline_cache[cache_key]

        generator = torch.Generator(device=device)
        if seed is None:
            seed = random.randint(0, 2**32)
        generator.manual_seed(seed)

        images = []
        for i in tqdm(range(batch_count), desc="Generating"):
            current_seed = seed + i
            generator.manual_seed(current_seed)

            init_image = None
            if task == "img2img":
                if not uploaded_image:
                    raise gr.Error("Please upload an image for img2img mode")
                size = (width, int(width) if lock_aspect else height)
                init_image = preprocess_image(uploaded_image, size)

            generation_args = {
                "prompt": prompt,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "num_inference_steps": steps,
                "negative_prompt": negative_prompt or None
            }

            if task == "img2img":
                generation_args.update({
                    "image": init_image,
                    "strength": strength
                })
            else:
                generation_args.update({
                    "height": height,
                    "width": width
                })

            result = pipe.pipe(**generation_args).images[0]

            if enable_facefix:
                result = apply_face_enhancement(result, device)
            if upscale_method != "none":
                result = apply_upscaling(result, device, method=upscale_method)

            timestamp = int(time.time())
            filename = f"output_{timestamp}_{current_seed}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            result.save(save_path)
            images.append(result)

        return images if batch_count > 1 else images[0], f"✅ {batch_count} image(s) saved to: {OUTPUT_DIR}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# -------- UI --------

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## 🎨 Local Stable Diffusion Generator")

        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(choices=scan_model_folder(), label="Model File (from /models)")
                prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Enter a description...")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="What to avoid...")
                task = gr.Radio(choices=["text2img", "img2img"], value="text2img", label="Generation Mode")
                uploaded_image = gr.Image(label="Input Image (for img2img)", type="pil", visible=False)

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    strength = gr.Slider(0.0, 1.0, 0.75, label="Denoising Strength (img2img)")
                    guidance = gr.Slider(1.0, 20.0, 7.5, label="Guidance Scale")
                    steps = gr.Slider(10, 100, 50, step=5, label="Inference Steps")
                    width = gr.Slider(256, 1024, 512, step=8, label="Width")
                    height = gr.Slider(256, 1024, 512, step=8, label="Height")
                    lock_aspect = gr.Checkbox(label="Lock height to match width", value=True)
                    seed = gr.Number(label="Seed (random if blank)", value=None, precision=0)
                    batch_count = gr.Slider(1, 5, 1, step=1, label="Batch Count")
                    scheduler = gr.Dropdown(["ddpm", "pndm", "lms", "euler", "ddim"], value="ddim", label="Scheduler")
                    device = gr.Radio(["CPU", "CUDA", "DirectML"], value="CPU", label="Device")
                    
                    enable_facefix = gr.Checkbox(
                        label="Enhance Faces (GFPGAN)",
                        value=False,
                        interactive=UPSCALERS_AVAILABLE,
                        visible=UPSCALERS_AVAILABLE
                    )
                    
                    upscale_method = gr.Radio(
                        choices=["none", "realesrgan", "lanczos"],
                        value="none",
                        label="Upscale Method"
                    )

                generate_btn = gr.Button("🚀 Generate Image")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", show_label=True)
                status = gr.Textbox(label="Status", lines=2)

        # Auto toggle upload box
        task.change(
            fn=lambda mode: gr.update(visible=(mode == "img2img")),
            inputs=task,
            outputs=uploaded_image
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, uploaded_image, strength, guidance, seed,
                model_dropdown, device, task, scheduler, steps,
                width, height, negative_prompt, batch_count,
                lock_aspect, upscale_method, enable_facefix
            ],
            outputs=[output_image, status]
        )

    return interface

# -------- App Entry --------

if __name__ == "__main__":
    if not DML_AVAILABLE:
        print("ℹ️ DirectML support not available.")
    interface = create_interface()
    interface.launch()
