from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import load_image
import numpy as np
import inversion
from pathlib import Path
import sa_handler
import torch

num_inference_steps = 50

# some parameters you can adjust to control fidelity to reference
shared_score_shift = np.log(2)  # higher value induces higher fidelity, set 0 for no shift
shared_score_scale = 1  # higher value induces higher, set 1 for no rescale

# for very famouse images consider supressing attention to refference, here is a configuration example:
# shared_score_shift = np.log(1)
# shared_score_scale = 0.5

src_paths = [Path("../data/train/00/images/boat.png")]
src_images = [np.array(load_image(str(src_path)).resize((512, 512))) for src_path in src_paths]
prompts = [path.stem for path in src_paths] + ["lighthouse"]


# scheduler = DDIMScheduler(
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
# )
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", #scheduler=scheduler, torch_dtype=torch.float32,
)
pipeline = pipeline.to("cuda")

handler = sa_handler.Handler(pipeline)
sa_args = sa_handler.StyleAlignedArgs(
    share_group_norm=True,
    share_layer_norm=True,
    share_attention=True,
    adain_queries=True,
    adain_keys=True,
    adain_values=False,
    shared_score_shift=shared_score_shift,
    shared_score_scale=shared_score_scale,
    full_attention_share=False,  # To utilize all available examples
)
handler.register(sa_args)

zts = inversion.ddim_inversion(
    pipeline, src_images[0], prompts[0], num_inference_steps, 4.0
)
zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

latents = torch.randn(
    len(prompts), 4, 64, 64, device="cpu", dtype=torch.float32
).to("cuda")
latents[0] = zT

images_a = pipeline(
    prompts,
    latents=latents,
    callback_on_step_end=inversion_callback,
    num_inference_steps=num_inference_steps,
    guidance_scale=10.0,
).images

for i, image in enumerate(images_a):
    image.save(f"./output/{prompts[i]}.png")
