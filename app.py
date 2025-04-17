import gradio as gr
import torch
import time
import random
import logging
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)
import numpy as np

# Try to import DirectML for AMD GPU support on Windows
try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImageGen")

# Device configuration
device = None
if torch.cuda.is_available():
    device = "cuda"
    default_dtype = torch.float16
elif dml_available:
    device = torch_directml.device()
    default_dtype = torch.float32
else:
    device = "cpu"
    default_dtype = torch.float32

logger.info(f"Using device: {device}")

# Constants
MAX_SEED = np.iinfo(np.int32).max
SUPPORTED_RESOLUTIONS = {
    "sd": ["360x480", "512x512", "768x768"],
    "sdxl": ["1024x1024"],
}
SCHEDULER_MAP = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "LMS": LMSDiscreteScheduler
}

class UnifiedPipeline:
    def __init__(self):
        self.pipe = None
        self.model_type = None

    def load_model(self, model_path: str, model_type: str, low_vram: bool, use_float16: bool, scheduler_name: str):
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_type = model_type
        logger.info(f"Loading {model_type} model from {model_path}")

        try:
            model_path_str = str(model_path_obj.resolve())
            dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32

            # Common pipeline parameters
            pipe_params = {
                "pretrained_model_link_or_path": model_path_str,
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

            # Set scheduler
            scheduler_class = SCHEDULER_MAP[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

            # Apply Low VRAM optimizations
            if low_vram:
                logger.info("Applying Low VRAM optimizations")
                if hasattr(self.pipe, "enable_model_cpu_offload"):
                    self.pipe.enable_model_cpu_offload()
                elif hasattr(self.pipe, "enable_sequential_cpu_offload"):
                    self.pipe.enable_sequential_cpu_offload()
                
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision('medium')
            else:
                self.pipe = self.pipe.to(device)

            self.pipe.set_progress_bar_config(disable=True)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt, width, height, steps, seed):
        # Fixed generator handling
        if device == "cpu":
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        return self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            output_type="pil"
        ).images[0]

def generate_image(model_type, prompt, resolution, model_path, seed, randomize_seed, 
                  steps, save_path, low_vram, use_float16, scheduler):
    try:
        if model_type not in SUPPORTED_RESOLUTIONS:
            return None, seed, "Invalid model type"
        
        if resolution not in SUPPORTED_RESOLUTIONS[model_type]:
            return None, seed, "Unsupported resolution"

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        torch.manual_seed(seed)

        width, height = map(int, resolution.split('x'))

        pipeline = UnifiedPipeline()
        pipeline.load_model(model_path, model_type, low_vram, use_float16, scheduler)

        start_time = time.time()
        image = pipeline.generate(prompt, width, height, steps, seed)

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        image_path = save_path / f"generated_{seed}_{int(time.time())}.png"
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
        if f.is_file() and f.suffix.lower() in [".safetensors", ".ckpt"]
    ]

with gr.Blocks(title="Enhanced Image Generator") as demo:
    gr.Markdown("# 🖼️ Enhanced Image Generator")
    
    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(
                label="Model Type",
                choices=["sd", "sdxl"],
                value="sd"
            )
            model_dir = gr.Textbox(
                label="Model Directory",
                value="./models",
                placeholder="Path to model directory"
            )
            model_file = gr.Dropdown(
                label="Model File",
                choices=scan_model_files("./models"),
                interactive=True
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A futuristic cityscape at sunset",
                lines=3
            )
            resolution = gr.Dropdown(
                label="Resolution", 
                choices=SUPPORTED_RESOLUTIONS["sd"],
                value="512x512"
            )
            scheduler = gr.Dropdown(
                label="Sampling Method",
                choices=list(SCHEDULER_MAP.keys()),
                value="DPM++ 2M"
            )
            steps = gr.Slider(1, 50, value=20, label="Inference Steps")
            seed = gr.Number(42, label="Seed")
            randomize_seed = gr.Checkbox(True, label="Randomize Seed")
            low_vram = gr.Checkbox(False, label="Low VRAM Mode")
            use_float16 = gr.Checkbox(
                value=default_dtype == torch.float16,
                label="Use Float16 (faster, less memory)",
                interactive=device != "cpu"
            )
            save_path = gr.Textbox(
                label="Save Path",
                value="./outputs",
                placeholder="Path to save generated images"
            )
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Generated Image", height=512)
            info_seed = gr.Number(label="Used Seed", interactive=False)
            info_latency = gr.Textbox(label="Generation Info", interactive=False)

    model_dir.change(
        fn=lambda x: gr.Dropdown.update(choices=scan_model_files(x)),
        inputs=model_dir,
        outputs=model_file
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[
            model_type,
            prompt,
            resolution,
            model_file,
            seed,
            randomize_seed,
            steps,
            save_path,
            low_vram,
            use_float16,
            scheduler
        ],
        outputs=[output_image, info_seed, info_latency]
    )

if __name__ == "__main__":
    demo.launch()
