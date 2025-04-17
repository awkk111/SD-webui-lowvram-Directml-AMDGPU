import torch
import random
import time
import gc
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import gradio as gr
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

# Device setup
try:
    import torch_directml
    device = torch_directml.device()
    print("✅ Using DirectML device")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚠️ Using fallback: {device}")

SCHEDULERS = {
    "DPM++ Fast": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

DEFAULT_STEPS = 15
MAX_BATCH_SIZE = 1
MAX_SEED = 2**32 - 1
RESOLUTION_PRESETS = {
    "Standard (512x512)": (512, 512),
    "High (1024x1024)": (1024, 1024),
    "Low (256x256)": (256, 256),
    "192x256": (192, 256),
    "360x480": (360, 480)
}

class UnifiedPipeline:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_path, model_type, low_vram, use_float16, scheduler_name):
        model_path = Path(model_path)
        dtype = torch.float32 if "DirectML" in str(device) else (torch.float16 if use_float16 else torch.float32)
        pipe_params = {
            "pretrained_model_link_or_path": str(model_path.resolve()),
            "torch_dtype": dtype,
            "safety_checker": None,
            "use_safetensors": True,
        }

        if model_type == "sd":
            self.pipe = StableDiffusionPipeline.from_single_file(**pipe_params)
        elif model_type == "sdxl":
            self.pipe = StableDiffusionXLPipeline.from_single_file(**pipe_params)
        else:
            raise ValueError("Unsupported model type")

        scheduler_class = SCHEDULERS.get(scheduler_name, DPMSolverMultistepScheduler)
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

        for comp in ["text_encoder", "unet", "vae", "text_encoder_2"]:
            part = getattr(self.pipe, comp, None)
            if part and hasattr(part, "to"):
                part.to(device)

        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt, width, height, steps, seed, negative_prompt, cfg_scale):
        generator = torch.Generator(device=device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            negative_prompt=negative_prompt,
            guidance_scale=cfg_scale,
            output_type="pil"
        )
        image = result.images[0]
        del result
        gc.collect()
        return image

class ImageGenerator:
    def __init__(self):
        self.models = {}
        self.pipeline = None

    def load_models(self, model_dir):
        model_dir = Path(model_dir)
        self.models = {
            model.stem: str(model)
            for model in model_dir.glob("*.safetensors") if model.is_file()
        }
        return list(self.models.keys())

    def create_pipeline(self, model_name, scheduler_name, low_vram):
        model_path = self.models[model_name]
        self.pipeline = UnifiedPipeline()
        self.pipeline.load_model(model_path, "sd", low_vram, use_float16=True, scheduler_name=scheduler_name)

    def generate_batch(self, prompt, resolution, steps, seed, num_images, negative_prompt, cfg_scale, enhancement_type):
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        width, height = RESOLUTION_PRESETS[resolution]
        full_negative = (
            "(bad anatomy, blurry, extra limbs, disfigured, low quality), " + negative_prompt if negative_prompt else ""
        )
        prompt = prompt + ", indian housewife"

        seeds = [random.randint(0, MAX_SEED) if seed == -1 else seed + i for i in range(num_images)]
        images = []
        for s in tqdm(seeds, desc="Generating"):
            img = self.pipeline.generate(prompt, width, height, steps, s, full_negative, cfg_scale)
            img = self.enhance_image(img, enhancement_type)
            images.append(img)

            Path("outputs").mkdir(exist_ok=True)
            output_path = Path("outputs") / f"generated_image_{s}.png"
            img.save(output_path)
            print(f"Image saved at: {output_path}")
            
        return images, f"Generated {len(images)} images."

    def enhance_image(self, image, enhancement_type):
        if enhancement_type == "enhance_contrast":
            return ImageEnhance.Contrast(image).enhance(1.5)
        elif enhancement_type == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image

def create_interface():
    generator = ImageGenerator()

    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .dark .gradio-container {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.9);
    }
    .dark .card {
        background: rgba(0, 0, 0, 0.7);
    }
    .status {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status.success {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .status.error {
        background: #ffebee;
        color: #c62828;
    }
    .log-output {
        padding: 15px;
        background: rgba(0,0,0,0.05);
        border-radius: 8px;
        margin-top: 10px;
        max-height: 200px;
        overflow-y: auto;
    }
    .sys-info {
        margin-top: 20px;
        padding: 15px;
        background: rgba(0,0,0,0.05);
        border-radius: 8px;
    }
    """

    with gr.Blocks(title="AI Image Generator", theme="soft", css=custom_css) as app:
        gr.Markdown("""
        # 🎨 AI Image Generator
        *Powered by Stable Diffusion & DirectML*
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Model Setup Section
                with gr.Group():
                    gr.Markdown("### Model Setup")
                    with gr.Row():
                        model_dir = gr.Textbox(label="Model Folder", value="./models", scale=4)
                        load_btn = gr.Button("🔄 Load Models", scale=1)
                    model_selector = gr.Dropdown(label="Available Models", interactive=True, info="Select a model from your folder")
                    status_output = gr.Markdown("🟠 No model loaded", elem_classes="status")

                # Prompt Settings
                with gr.Accordion("📝 Prompt Settings", open=True):
                    prompt = gr.Textbox(label="Positive Prompt", lines=3, placeholder="Describe your image...")
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, 
                                                placeholder="What to exclude from the image...")

                # Generation Parameters
                with gr.Accordion("⚙️ Generation Parameters", open=True):
                    with gr.Row():
                        resolution = gr.Dropdown(label="Resolution", choices=list(RESOLUTION_PRESETS.keys()), 
                                                value="Standard (512x512)", scale=2)
                        steps = gr.Slider(8, 50, value=DEFAULT_STEPS, step=1, label="Steps", scale=1)
                    cfg_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.1, label="CFG Scale")
                    
                    with gr.Row():
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        num_images = gr.Slider(1, MAX_BATCH_SIZE, value=1, step=1, label="Batch Size")

                # Advanced Settings
                with gr.Accordion("🔧 Advanced Settings"):
                    with gr.Row():
                        scheduler = gr.Dropdown(choices=list(SCHEDULERS.keys()), value="DPM++ Fast", label="Scheduler")
                        enhancement_type = gr.Dropdown(["None", "enhance_contrast", "blur"], 
                                                     value="None", label="Post-Processing")
                    low_vram = gr.Checkbox(label="Low VRAM Mode", value=True, 
                                         info="Enable if experiencing memory issues")

                generate_btn = gr.Button("✨ Generate Images", variant="primary")

            # Output Column
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated Images", show_label=True, columns=2, 
                                   object_fit="scale-down", height=600)
                with gr.Group():
                    gr.Markdown("## Generation Info")
                    log_window = gr.HTML(elem_classes="log-output")
                    sys_info = gr.HTML()

        # Example Prompts
        with gr.Accordion("💡 Example Prompts", open=False):
            gr.Examples(
                examples=[
                    ["A majestic lion in the savannah at sunset, detailed fur, golden hour lighting", 
                     "Standard (512x512)", 25, 7.5],
                    ["Cyberpunk cityscape with neon lights and flying cars, rain reflections", 
                     "High (1024x1024)", 30, 9.0],
                    ["Watercolor painting of a medieval castle surrounded by cherry blossoms", 
                     "360x480", 20, 8.0]
                ],
                inputs=[prompt, resolution, steps, cfg_scale],
                label="Click to apply examples"
            )

        # System Information
        device_info = "DirectML" if "torch_directml" in globals() else ("CUDA" if torch.cuda.is_available() else "CPU")
        sys_info.value = f"""
        <div class="sys-info">
            <h3>System Information</h3>
            <p>Device: {device_info}<br>
            Torch Version: {torch.__version__}<br>
            Gradio Version: {gr.__version__}</p>
        </div>
        """

        # Callbacks
        def load_models_cb(model_dir):
            try:
                model_list = generator.load_models(model_dir)
                return gr.update(choices=model_list, value=model_list[0] if model_list else None), "🟢 Models loaded successfully"
            except Exception as e:
                return gr.update(choices=[], value=None), f"🔴 Error: {str(e)}"

        def set_model_cb(model_name, scheduler_name, low_vram_flag):
            try:
                generator.create_pipeline(model_name, scheduler_name, low_vram_flag)
                return f"🟢 Pipeline ready: {model_name}"
            except Exception as e:
                return f"🔴 Error loading model: {str(e)}"

        def generate_images_cb(prompt, resolution, steps, seed, num_images, negative_prompt, cfg_scale, enhancement_type):
            try:
                start_time = time.time()
                images, log = generator.generate_batch(prompt, resolution, steps, seed, num_images, negative_prompt, cfg_scale, enhancement_type)
                elapsed = time.time() - start_time
                return images, f"<pre>✅ Success! Generated {len(images)} images in {elapsed:.1f}s</pre>"
            except Exception as e:
                return [], f"<pre style='color: red'>❌ Error: {str(e)}</pre>"

        # Event handlers
        load_btn.click(
            load_models_cb,
            inputs=model_dir,
            outputs=[model_selector, status_output]
        )
        
        model_selector.change(
            set_model_cb,
            inputs=[model_selector, scheduler, low_vram],
            outputs=status_output
        )
        
        generate_btn.click(
            generate_images_cb,
            inputs=[prompt, resolution, steps, seed, num_images, negative_prompt, cfg_scale, enhancement_type],
            outputs=[gallery, log_window]
        )

    return app

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    app = create_interface()
    app.launch(server_name="0.0.0.0" if "torch_directml" in globals() else "localhost")
