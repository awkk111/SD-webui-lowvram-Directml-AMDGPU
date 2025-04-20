import torch
import time
import random
import logging
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import tqdm  # Added tqdm import

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)

# Try DirectML for AMD GPUs
try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImageGen")

# Device setup
if dml_available:
    device = torch_directml.device()
    default_dtype = torch.float16
    logger.info("Using DirectML (AMD GPU)")
else:
    device = "cpu"
    default_dtype = torch.float32
    logger.info("Using CPU")

MAX_SEED = np.iinfo(np.int32).max

SCHEDULER_MAP = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "LMS": LMSDiscreteScheduler
}

SUPPORTED_RESOLUTIONS = [
    "128x128", "128x256", "256x256", "360x480", "320x400", 
    "512x512", "640x480", "768x1024"
]

pipeline = None


class GFPGANer:
    def __init__(self, model_path: str, upscale: int = 2):
        self.model_path = model_path
        self.upscale = upscale
        logger.info(f"Loaded GFPGAN model from {model_path}")
    
    def enhance(self, image_cv, has_aligned=False, only_face=True):
        logger.info("Restoring faces using GFPGAN...")
        return image_cv, None, None  # Returning image as-is for this example


class UnifiedPipeline:
    def __init__(self, face_restoration_model_path=None):
        self.pipe = None
        self.face_restoration_model = None
        if face_restoration_model_path:
            self.load_face_restoration_model(face_restoration_model_path)
    
    def load_model(self, model_path: str, model_type: str, low_vram: bool, 
                   use_float16: bool, scheduler_name: str):
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading {model_type} model from {model_path}")
        try:
            dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32

            pipe_params = {
                "pretrained_model_link_or_path": str(model_path_obj.resolve()),
                "torch_dtype": dtype,
                "safety_checker": None,
                "use_safetensors": True,
            }

            if model_type == "sd":
                self.pipe = StableDiffusionPipeline.from_single_file(**pipe_params)
            elif model_type == "sdxl":
                self.pipe = StableDiffusionXLPipeline.from_single_file(**pipe_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            scheduler_class = SCHEDULER_MAP[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

            self.pipe.text_encoder.to(cpu)
            self.pipe.unet.to(device)
            self.pipe.vae.to(cpu)

            if low_vram:
                if hasattr(self.pipe, "enable_model_cpu_offload"):
                    self.pipe.enable_model_cpu_offload()
                elif hasattr(self.pipe, "enable_sequential_cpu_offload"):
                    self.pipe.enable_sequential_cpu_offload()
                else:
                    logger.warning("Low VRAM mode is not supported in this model.")
            else:
                self.pipe = self.pipe.to(device)

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")

            self.pipe.set_progress_bar_config(disable=True)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt, width, height, steps, seed, 
                 face_restoration=False, gr_progress=None):
        generator = torch.Generator(device=device).manual_seed(seed)
        
        try:
            # Initialize tqdm progress bar
            pbar = tqdm(total=steps, desc="Generating Image", unit="step")
            
            # Define callback function for progress updates
            def update_progress(step, timestep, latents):
                pbar.update(1)
                if gr_progress is not None:
                    gr_progress.update(step + 1)  # Update Gradio progress

            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                output_type="pil",
                #callback=update_progress,
                #callback_steps=1
            )
            pbar.close()  # Close tqdm progress bar

            image = result.images[0]

            if face_restoration:
                image = self.apply_face_restoration(image)

        except Exception as e:
            pbar.close()  # Ensure progress bar closes on error
            raise

        return image

    def upscale(self, image, scale_factor):
        width, height = image.size
        return image.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)

    def load_face_restoration_model(self, model_path: str):
        self.face_restoration_model = GFPGANer(model_path=model_path, upscale=2)

    def apply_face_restoration(self, image: Image.Image):
        image_cv = np.array(image)
        restored_img, _, _ = self.face_restoration_model.enhance(
            image_cv, has_aligned=False, only_face=True
        )
        restored_image = Image.fromarray(restored_img)
        return restored_image


def generate_image(model_type, prompt, resolution, model_file, seed, randomize_seed,
                   steps, save_path, low_vram, use_float16, scheduler, upscale, 
                   face_restoration, face_restoration_model_path):
    global pipeline

    try:
        if pipeline is None:
            logger.info("Pipeline not initialized. Loading model dynamically...")
            pipeline = UnifiedPipeline(
                face_restoration_model_path if face_restoration else None
            )
            pipeline.load_model(model_file, model_type, low_vram, 
                               use_float16, scheduler)

        if randomize_seed or seed < 0:
            seed = random.randint(0, MAX_SEED)
        torch.manual_seed(seed)

        width, height = map(int, resolution.split('x'))

        start_time = time.time()
        
        # Create Gradio progress bar
        gr_progress = gr.Progress()

        image = pipeline.generate(
            prompt, width, height, steps, seed, 
            face_restoration, gr_progress
        )

        if upscale == "2x":
            image = pipeline.upscale(image, 2)
        elif upscale == "4x":
            image = pipeline.upscale(image, 4)

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"generated_{seed}_{int(time.time())}.png"
        image.save(image_path)

        latency = time.time() - start_time
        return image, seed, f"Saved to {image_path}, generated in {latency:.2f}s"

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return None, seed, f"Error: {str(e)}"


def scan_model_files(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    return [
        str(f.resolve()) for f in model_dir.glob("*")
        if f.suffix.lower() in [".safetensors", ".ckpt"]
    ]


# Gradio UI
with gr.Blocks(title="Offline Image Generator") as demo:
    gr.Markdown("## 🧠 Offline Stable Diffusion Generator (DirectML / CPU)")

    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(["sd", "sdxl"], value="sd", label="Model Type")
            model_dir = gr.Textbox(label="Model Directory", value="./models")
            model_file = gr.Dropdown(
                choices=scan_model_files("./models"), 
                label="Model File"
            )
            model_dir.change(
                lambda x: gr.Dropdown.update(choices=scan_model_files(x)), 
                inputs=model_dir, 
                outputs=model_file
            )

            prompt = gr.Textbox(label="Prompt", lines=3, 
                              placeholder="e.g. A fantasy village in the sky")
            resolution = gr.Dropdown(SUPPORTED_RESOLUTIONS, value="512x512", 
                                   label="Resolution")
            scheduler = gr.Dropdown(list(SCHEDULER_MAP.keys()), value="DPM++ 2M", 
                                   label="Scheduler")
            steps = gr.Slider(1, 50, value=20, step=1, label="Inference Steps")
            seed = gr.Number(value=-1, label="Seed (-1 for random)")
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            low_vram = gr.Checkbox(label="Low VRAM Mode", value=False)
            use_float16 = gr.Checkbox(
                label="Use float16 (if supported)", 
                value=(default_dtype == torch.float16)
            )
            save_path = gr.Textbox(label="Output Directory", value="./outputs")
            upscale = gr.Dropdown(["None", "2x", "4x"], value="None", 
                                label="Upscale Image")
            face_restoration = gr.Checkbox(label="Enable Face Restoration", value=False)
            face_restoration_model_path = gr.Textbox(
                label="GFPGAN Model Path", 
                value="./models/GFPGANv1.4.pth"
            )

            generate_button = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", height=512)
            used_seed = gr.Number(label="Used Seed", interactive=False)
            generation_info = gr.Textbox(label="Info", interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[
            model_type, prompt, resolution, model_file, seed, randomize_seed,
            steps, save_path, low_vram, use_float16, scheduler, upscale,
            face_restoration, face_restoration_model_path
        ],
        outputs=[output_image, used_seed, generation_info]
    )

    demo.queue().launch(share=False)
