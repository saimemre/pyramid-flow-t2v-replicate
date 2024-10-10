# predict.py

import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = snapshot_download("rain1011/pyramid-flow-sd3")
        self.model_dtype = 'bf16'
        self.torch_dtype = torch.bfloat16

        self.model = PyramidDiTForVideoGeneration(
            self.model_path,
            self.model_dtype,
            model_variant='diffusion_transformer_768p',
        )

        self.model.vae.to(self.device)
        self.model.dit.to(self.device)
        self.model.text_encoder.to(self.device)
        self.model.vae.enable_tiling()

    def predict(
        self,
        prompt: str = Input(description="Text prompt for video generation"),
        image: Path = Input(description="Input image for image-to-video generation", default=None),
        height: int = Input(description="Height of the video", default=768),
        width: int = Input(description="Width of the video", default=1280),
        num_frames: int = Input(description="Number of frames to generate", default=16),
        guidance_scale: float = Input(description="Guidance scale for generation", default=9.0),
        video_guidance_scale: float = Input(description="Video guidance scale", default=5.0),
    ) -> Path:
        """Run a single prediction on the model"""
        if image:
            # Image-to-video generation
            input_image = Image.open(image).convert("RGB").resize((width, height))
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
                frames = self.model.generate_i2v(
                    prompt=prompt,
                    input_image=input_image,
                    num_inference_steps=[10, 10, 10],
                    temp=num_frames,
                    video_guidance_scale=video_guidance_scale,
                    output_type="pil",
                    save_memory=True,
                )
        else:
            # Text-to-video generation
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
                frames = self.model.generate(
                    prompt=prompt,
                    num_inference_steps=[20, 20, 20],
                    video_num_inference_steps=[10, 10, 10],
                    height=height,
                    width=width,
                    temp=num_frames,
                    guidance_scale=guidance_scale,
                    video_guidance_scale=video_guidance_scale,
                    output_type="pil",
                    save_memory=True,
                )

        output_path = Path("output.mp4")
        export_to_video(frames, str(output_path), fps=24)
        return output_path