import gradio as gr
import torch
import time
import random
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DirectMLImageGen")

# DirectML device setup
try:
    import torch_directml
    dml = torch_directml.device(torch_directml.default_device())
    logger.info(f"Using DirectML device: privateuseone")
except ImportError:
    dml = None
    logger.info("DirectML not available, using CPU/CUDA")

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
    default_dtype = torch.float16
elif dml:
    device = "privateuseone"
    default_dtype = torch.float16
else:
    device = "cpu"
    default_dtype = torch.float16

logger.info(f"Using device: {device}")

MAX_SEED = np.iinfo(np.int32).max
RESOLUTION_PRESETS = {
    "Low VRAM (192x256)": (192, 256),
    "Mobile (256x256)": (256, 256),
    "Portrait (360x480)": (360, 480),
    "Standard (512x512)": (512, 512),
    "Widescreen (512x768)": (512, 768),
    "HD (640x360)": (640, 360)
}
SCHEDULERS = {
    "DPM++ Fast": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler
}
DEFAULT_STEPS = 15
MAX_BATCH_SIZE = 1

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

            scheduler_class = SCHEDULERS.get(scheduler_name, DPMSolverMultistepScheduler)
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

            # Move individual components to device
            logger.info(f"Moving pipeline components to {device}")
            self.pipe.text_encoder.to(device)
            self.pipe.unet.to(device)
            self.pipe.vae.to(device)

            self.pipe.set_progress_bar_config(disable=True)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt, width, height, steps, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            output_type="pil"
        ).images[0]

class ImageGenerator:
    def __init__(self):
        self.models = {}
        self.pipeline = None
        self.current_model = None
        self.low_vram = True

    def load_models(self, model_dir):
        model_dir = Path(model_dir)
        self.models = {
            model.stem: str(model)
            for model in model_dir.glob("*.safetensors")
            if model.is_file()
        }
        logger.info(f"Loaded {len(self.models)} models from {model_dir}")
        return list(self.models.keys())

    def create_pipeline(self, model_name, scheduler_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        logger.info(f"Initializing pipeline for {model_name}")
        if self.pipeline is not None:
            del self.pipeline
            torch.directml.synchronize()

        model_path = self.models[model_name]
        self.pipeline = UnifiedPipeline()
        self.pipeline.load_model(model_path, "sd", self.low_vram, use_float16=True, scheduler_name=scheduler_name)
        self.current_model = model_name
        logger.info(f"Pipeline ready for {model_name}")

    def generate_batch(self, prompt, resolution, steps, seed, num_images):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Please select a model and scheduler first.")
        
        width, height = RESOLUTION_PRESETS.get(resolution, (512, 512))
        seeds = [random.randint(0, MAX_SEED) if seed == -1 else seed + i for i in range(num_images)]

        logger.info(f"Generating {num_images} {width}x{height} images")
        start_time = time.time()
        images = []
        try:
            for s in seeds:
                image = self.pipeline.generate(prompt, width, height, steps, s)
                images.append(image)
                #torch.dml.synchronize()
        except RuntimeError as e:
            logger.error(f"Generation failed: {str(e)}")
            return [], f"Error: {str(e)}"

        gen_time = time.time() - start_time
        time_str = f"Generated {num_images} images in {gen_time:.2f}s ({gen_time/num_images:.2f}s/img)"
        return images, time_str

def create_interface():
    generator = ImageGenerator()

    with gr.Blocks(title="DirectML Image Generator", css=".gradio-container {max-width: 2000px}") as app:
        gr.Markdown("# 🚀 DirectML Image Generator (AMD/Intel GPUs, .safetensors support)")

        with gr.Row():
            with gr.Column(scale=5):
                model_dir = gr.Textbox(label="Model Directory", value="./models", placeholder="Path to .safetensors models")
                load_btn = gr.Button("Load Models", variant="secondary")

                model_selector = gr.Dropdown(label="Select Model", choices=[], interactive=True)
                scheduler = gr.Dropdown(label="Sampling Method", choices=list(SCHEDULERS.keys()), value="DPM++ Fast")
                low_vram = gr.Checkbox(label="Low VRAM Mode", value=True, interactive=True)

                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3)
                resolution = gr.Dropdown(label="Resolution Preset", choices=list(RESOLUTION_PRESETS.keys()), value="Standard (360x480)")
                steps = gr.Slider(minimum=8, maximum=50, value=DEFAULT_STEPS, step=1, label="Inference Steps")
                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                num_images = gr.Slider(minimum=1, maximum=MAX_BATCH_SIZE, value=2, step=1, label="Batch Size")
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=5):
                gallery = gr.Gallery(label="Generated Images", columns=8, height=800)
                status = gr.Textbox(label="Status", interactive=False)
                perf_info = gr.Textbox(label="DirectML Performance", interactive=False)

        load_btn.click(
            lambda model_dir: generator.load_models(model_dir),
            inputs=model_dir,
            outputs=model_selector
        ).then(
            lambda: gr.update(choices=list(generator.models.keys()), value=None, interactive=True),
            outputs=model_selector
        ).then(
            lambda: f"Loaded {len(generator.models)} DirectML-compatible models",
            outputs=status
        )

        model_selector.change(
            lambda model_name, scheduler_name: generator.create_pipeline(model_name, scheduler_name),
            inputs=[model_selector, scheduler],
            outputs=[]
        ).then(
            lambda: "Pipeline ready",
            outputs=status
        )

        low_vram.change(
            lambda x: setattr(generator, 'low_vram', x),
            inputs=low_vram,
            outputs=None
        ).then(
            generator.create_pipeline,
            inputs=[model_selector, scheduler],
            outputs=None
        ).then(
            lambda _: f"DirectML optimization: {'Low VRAM' if generator.low_vram else 'Normal'}",
            outputs=status
        )

        generate_btn.click(
            generator.generate_batch,
            inputs=[prompt, resolution, steps, seed, num_images],
            outputs=[gallery, perf_info]
        ).then(
            lambda _: "DirectML generation complete",
            outputs=status
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_port=7860, share=False, show_error=True)
