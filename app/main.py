import gradio as gr

# from mindone.diffusers import DiffusionPipeline, DDIMScheduler
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe.load_lora_weights("ByteDance/Hyper-SD", adapter_name="hyper", weight_name="Hyper-SDXL-2steps-lora.safetensors")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
import mindspore
mindspore.set_context(pynative_synchronize=True)
from mindone.diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", mindspore_dtype=mindspore.float16)


# def t2i(prompt):
#     return pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0)[0][0]
def t2i(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale):
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )[0][0]


if __name__ == "__main__":
    demo = gr.Interface(
        fn=t2i,
        inputs=[
            gr.Textbox(label="prompt", placeholder="The prompt to guide the image generation.", value="A photo of a cute cat."),
            gr.Textbox(label="negative_prompt", placeholder="[Optional] The prompt to guide what to not include in image generation."),
            gr.Number(label="height", value=1024),
            gr.Number(label="width", value=1024),
            gr.Slider(label="num_inference_steps", value=28, minimum=1, maximum=100, step=1),
            gr.Slider(label="guidance_scale", value=7.0, minimum=0.0, maximum=20.0, step=0.1),
        ],
        outputs=["image"],
        clear_btn=None,
        allow_flagging="never",
        title="Stable Diffusion 3",
        description="**Hello MindSpore** from **[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)**!",
        article="Check out [mindone/diffusers](https://github.com/mindspore-lab/mindone/blob/master/mindone/diffusers) that this demo is based off of.",
    )
    demo.launch(share=True, auth=("mindspore", "huawei"))
