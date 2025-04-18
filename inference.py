import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid

from modeling.dmm_pipeline import StableDiffusionDMMPipeline


if __name__ == "__main__":
    pipe = StableDiffusionDMMPipeline.from_pretrained(
        "path/to/pipeline/checkpoint",
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
    pipe.to("cuda")

    seed = 12345678

    outputs = []
    for i in range(pipe.unet.get_num_models()):
        output = pipe(
            prompt="portrait photo of a girl, long golden hair, flowers, best quality",
            negative_prompt="worst quality,low quality,normal quality,lowres,watermark,nsfw",
            width=512,
            height=512,
            num_inference_steps=25,
            guidance_scale=7,
            model_id=i,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        outputs.append(output)

    image_grid: Image = make_image_grid(outputs, rows=1, cols=pipe.unet.get_num_models())
    image_grid.save("output.jpg")
