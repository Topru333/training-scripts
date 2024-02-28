import gradio as gr
import os
import json
from lora_trainer_runner import LoraTrainerRunner

cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
training_runner = LoraTrainerRunner(last_setup_path=cache_dir, notification_fn=lambda notification: print(notification))


last_lora_training_setup = os.path.join(cache_dir, "last_training_lora.json")

gradio_values = {
    "command":"none"
}

def process(
    lora_module_choice_dropdown,
    project_name_textbox, 
    model_path_textbox, 
    dataset_path_choice_dropdown,
    dataset_path_textbox,
    chackbox_settings_group,
    output_dir_path_textbox,
    save_every_n_numberbox,
    resume_traning_checkbox,
    resume_path_textbox,
    logs_checkbox,
    logs_path_textbox,
    noise_offset_slider,
    batch_size_slider,
    gradient_accumulation_steps_slider,
    lr_scheduler_dropdown, 
    cosine_restarts_slider,
    mixed_precision_dropdown,
    save_precision_dropdown,
    warmup_steps_numberbox, 
    unet_lr_numberbox,
    text_encoder_numberbox,
    network_dim_slider,
    network_alpha_slider,
    conv_dim_slider,
    conv_alpha_slider,
    count_type_dropdown,
    count_value_numberbox,
    clip_skip_slider,
    max_token_length_slider,
    stop_text_encoder_training_numberbox,
    generate_samples_checkbox,
    sample_count_type_dropdown,
    sample_count_numberbox,
    path_to_prompts_textbox,
    samplers_dropdown,
    max_grad_norm
):
    setup_json = {
        'lora_module': lora_module_choice_dropdown,
        'project_name': project_name_textbox,
        'model_path': model_path_textbox,
        'dataset_choice': dataset_path_choice_dropdown,
        'dataset_path': dataset_path_textbox,
        'use_flip_aug': 'use_flip_aug' in chackbox_settings_group,
        "use_random_crop": 'use_random_crop' in chackbox_settings_group,
        'shuffle_caption': 'shuffle_caption' in chackbox_settings_group,
        "use_cache_latents": 'use_cache_latents' in chackbox_settings_group,
        'enable_bucket': 'enable_bucket' in chackbox_settings_group,
        'bucket_no_upscale': 'bucket_no_upscale' in chackbox_settings_group,
        'use_xformers': 'use_xformers' in chackbox_settings_group,
        'use_mem_eff_attn': 'use_mem_eff_attn' in chackbox_settings_group,
        'send_notification': 'send_notification' in chackbox_settings_group,
        'save_states': 'save_states' in chackbox_settings_group,
        'output_dir': output_dir_path_textbox,
        'save_every_n_epochs': save_every_n_numberbox,
        'resume_training': resume_traning_checkbox,
        'resume_path': resume_path_textbox,
        'do_logs': logs_checkbox,
        'logs_dir': logs_path_textbox,
        'noise_offset': noise_offset_slider,
        'train_batch_size': batch_size_slider,
        'gradient_accumulation_steps': gradient_accumulation_steps_slider,
        'lr_scheduler': lr_scheduler_dropdown, 
        'cosine_restarts': cosine_restarts_slider, 
        'mixed_precision': mixed_precision_dropdown,
        'save_precision': save_precision_dropdown,
        'lr_warmup_steps': warmup_steps_numberbox, 
        'count_type': count_type_dropdown,
        'max_count': int(count_value_numberbox),
        'unet_lr': unet_lr_numberbox,
        'text_encoder_lr': text_encoder_numberbox,
        'network_dim': network_dim_slider,
        'network_alpha': network_alpha_slider,
        'conv_dim': conv_dim_slider,
        'conv_alpha': conv_alpha_slider,
        'clip_skip': clip_skip_slider,
        'max_token_length': max_token_length_slider,
        'stop_text_encoder_training': stop_text_encoder_training_numberbox,
        'generate_samples': generate_samples_checkbox,
        'sample_count_type': sample_count_type_dropdown,
        'sample_per_count': sample_count_numberbox,
        'sample_prompts_path': path_to_prompts_textbox,
        'sample_sampler': samplers_dropdown,
        'max_grad_norm': max_grad_norm
    }

    training_runner.submit(setup_json)

if os.path.exists(last_lora_training_setup):
    with open(last_lora_training_setup, "r") as file:
        gradio_values = json.load(file)
else:
    print("Last lora setup does not exist.")

def config_choice_change(choice):
    if choice == "Config path":
        return gr.Textbox.update(label="Config path:", visible=True)
    elif choice == "Directory path":
        return gr.Textbox.update(label="Directory path:", visible=True)
    else:
        return gr.Textbox.update(visible=False)
    
def lr_scheduler_dropdown_change(choice):
    if choice == "cosine_with_restarts":
        return gr.Textbox.update(visible=True)
    else:
        return gr.Textbox.update(visible=False)

def count_type_change(choice):
    if choice == "step":
        return gr.Textbox.update(label="Steps count:", visible=True, interactive=True)
    elif choice == "epoch":
        return gr.Textbox.update(label="Epoches count:", visible=True, interactive=True)
    else:
        return gr.Textbox.update(visible=False)

        

lr_schedulers = ['constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'linear', 'polynomial']
save_precisions = ['fp16', 'bf16', 'float', None]
mixed_precisions = ['no', 'fp16', 'bf16']
count_types = ['step', 'epoch']
samplers = ['ddim', 'k_euler']
lora_modules = ['Lora', 'Lora-SDXL','LoRA-C3Lier','Locon', 'Loha', 'Lokr', 'Ia3']
optimizer_type = []

def training_gui():
    with gr.Blocks() as app:
        lora_module_choice_dropdown = gr.components.Dropdown(label="Lora module:", choices=lora_modules, value=gradio_values.get("lora_module", lora_modules[0]), interactive=True)
        project_name_textbox = gr.Text(label="Project name:", value=gradio_values.get("project_name", "default_name"), interactive=True)
        model_path_textbox = gr.Text(label="Model path:", value=gradio_values.get("model_path", "/media/ssd/AI_MODELS/StableDiffusion/anything/anything-v4.5.ckpt"), interactive=True)
        output_dir_path_textbox = gr.Text(label="Output directory path:", value=gradio_values.get("output_dir", ""), interactive=True)

        with gr.Group():
            resume_traning_checkbox = gr.Checkbox(value=False, label="Resume training?", interactive=True)
            resume_path_textbox = gr.Textbox(label="Resume path:", lines=1, interactive=True, visible=False, value=gradio_values.get("resume_path", ""))
            resume_traning_checkbox.change(fn=lambda is_enable : gr.Textbox.update(visible=is_enable), inputs=resume_traning_checkbox, outputs=resume_path_textbox)

        with gr.Group():
            logs_checkbox = gr.Checkbox(value=False, label="Turn on logs?", interactive=True)
            logs_path_textbox = gr.Textbox(label="Logs path:", lines=1, interactive=True, visible=False, value=gradio_values.get("logs_dir", ""))
            logs_checkbox.change(fn=lambda is_enable : gr.Textbox.update(visible=is_enable), inputs=logs_checkbox, outputs=logs_path_textbox)

        with gr.Row():
            dataset_path_choice_dropdown = gr.components.Dropdown(label="Dataset command:", choices=['Config path', 'Directory path'], value=gradio_values.get("dataset_path_type", "Config path"), interactive=True)
            dataset_path_textbox = gr.Textbox(label="Path:", lines=1, interactive=True, value=gradio_values.get("dataset_path", ""), visible=True)
            dataset_path_choice_dropdown.change(fn=config_choice_change, inputs=dataset_path_choice_dropdown, outputs=dataset_path_textbox)

        with gr.Row():
            unet_lr_numberbox = gr.Number(value=gradio_values.get("unet_lr", 5e-5), label="Unet learning rate:", precision=None, interactive=True)
            text_encoder_numberbox = gr.Number(value=gradio_values.get("text_encoder_lr", 2.5e-5), label="Text encoder learning rate:", precision=None, interactive=True)

        with gr.Group():
            lr_scheduler_dropdown = gr.components.Dropdown(lr_schedulers, label="Scheduler:", value=gradio_values.get("lr_scheduler", "cosine"), interactive=True)
            cosine_restarts_slider = gr.Slider(0, 10, label="Restarts count:", value=gradio_values.get("cosine_restarts", 1), step=1, visible=lr_scheduler_dropdown.value == "cosine_restarts", interactive=True)
            lr_scheduler_dropdown.change(fn=lr_scheduler_dropdown_change, inputs=lr_scheduler_dropdown, outputs=cosine_restarts_slider)

        batch_size_slider = gr.Slider(1, 12, value=gradio_values.get("train_batch_size", 2), step=1, label="Batch size:", interactive=True)
        gradient_accumulation_steps_slider = gr.Slider(0, 20, value=gradio_values.get("gradient_accumulation_steps", 0), step=1, label="Gradient accumulation steps:", interactive=True)
        max_grad_norm_slider = gr.Slider(1, 512, value=gradio_values.get("max_grad_norm", 0), label="Max token length:", step=1, interactive=True)

        with gr.Row():
            count_type_dropdown = gr.components.Dropdown(count_types, label="Count type:", value=gradio_values.get("count_type", "step"), interactive=True)
            count_value_numberbox = gr.Number(value=gradio_values.get("max_count", 0), label="Count:", visible=True, interactive=True, precision=0)
            count_type_dropdown.change(fn=count_type_change, inputs=count_type_dropdown, outputs=count_value_numberbox)

        warmup_steps_numberbox = gr.Number(value=gradio_values.get("lr_warmup_steps", 60), label="Warmup steps:", interactive=True)

        with gr.Row():
            network_dim_slider = gr.Slider(0, 320, value=gradio_values.get("network_dim", 128), step=32, label="Network dim:", interactive=True)
            network_alpha_slider = gr.Slider(0, 320, value=gradio_values.get("network_alpha", 64), step=16, label="Network alpha:", interactive=True)
        with gr.Row(visible=gradio_values.get("lora_module", lora_modules[0])!="Lora") as conv_row:
            conv_dim_slider = gr.Slider(0, 160, value=gradio_values.get("conv_dim", 0), step=1, label="Conv dim:", interactive=True)
            conv_alpha_slider = gr.Slider(0, 160, value=gradio_values.get("conv_alpha", 0), step=1, label="Conv alpha:", interactive=True)
        lora_module_choice_dropdown.change(fn=lambda choice : gr.Row.update(visible=choice!="Lora"), inputs=lora_module_choice_dropdown, outputs=conv_row)
        noise_offset_slider = gr.Slider(0, 0.5, value=gradio_values.get("noise_offset", 0), label="Noise offset:", step=0.02,interactive=True)

        with gr.Row():
            mixed_precision_dropdown = gr.components.Dropdown(mixed_precisions, value=gradio_values.get("precision", mixed_precisions[0]), label="Mixed precision:", interactive=True)
            save_precision_dropdown = gr.components.Dropdown(save_precisions, value=gradio_values.get("precision", save_precisions[0]), label="Save precision:", interactive=True)

        clip_skip_slider = gr.Slider(1, 7, value=gradio_values.get("clip_skip", 2), label="Clip skip:", step=1, interactive=True)
        max_token_length_slider = gr.Slider(1, 512, value=gradio_values.get("max_token_length", 150), label="Max token length:", step=1, interactive=True)

        stop_text_encoder_training_numberbox = gr.Number(value=gradio_values.get("stop_text_encoder_training", 0), label="Stop textencoder after (steps):", interactive=True)

        checkbox_settings_group = gr.CheckboxGroup(
            ["save_states", "use_flip_aug", "use_random_crop", "shuffle_caption", "use_cache_latents", "enable_bucket", "bucket_no_upscale", "use_xformers", "use_mem_eff_attn", "send_notification"], 
            value=[
                "use_flip_aug" if gradio_values.get("use_flip_aug", False) else "",
                "use_random_crop" if gradio_values.get("use_random_crop", False) else "",
                "use_cache_latents" if gradio_values.get("use_cache_latents", False) else "",
                "enable_bucket" if gradio_values.get("enable_bucket", False) else "",
                "bucket_no_upscale" if gradio_values.get("bucket_no_upscale", False) else "",
                "use_xformers" if gradio_values.get("use_xformers", False) else "",
                "use_mem_eff_attn" if gradio_values.get("use_mem_eff_attn", False) else "",
                "send_notification" if gradio_values.get("send_notification", False) else "",
                "shuffle_caption" if gradio_values.get("shuffle_caption", False) else "",
                "save_states" if gradio_values.get("save_states", False) else "",
            ],
            label="Options: ",
            interactive=True
        )

        save_every_n_numberbox = gr.Number(value=gradio_values.get("save_every_n_epochs", 100), label="Save every n epoches:", interactive=True)

        generate_samples_checkbox = gr.Checkbox(value=False, label="Generate samples?", interactive=True)

        with gr.Group(visible=False) as samples_app:
            with gr.Row():
                sample_count_type_dropdown = gr.components.Dropdown(count_types, value=gradio_values.get("sample_count_type", "epoch"), label="Sample count type:", interactive=True)
                sample_count_numberbox = gr.Number(value=gradio_values.get("sample_per_count", 1), precision=0, label="One sample in epoch/step count:", interactive=True)
            path_to_prompts_textbox = gr.Text(value=gradio_values.get("sample_prompts_path", ""), label="Path to prompts for samples:", interactive=True)
            samplers_dropdown = gr.components.Dropdown(samplers, value=gradio_values.get("sample_sampler", "k_euler"), label="Sampler:", interactive=True)
        generate_samples_checkbox.change(fn=lambda is_enable : gr.Group.update(visible=is_enable), inputs=generate_samples_checkbox, outputs=samples_app)
        add_training_to_queue_btn = gr.Button(value="Add training to queue", variant='primary')
        add_training_to_queue_btn.click(process, inputs=[
            lora_module_choice_dropdown,
            project_name_textbox, 
            model_path_textbox, 
            dataset_path_choice_dropdown,
            dataset_path_textbox,
            checkbox_settings_group,
            output_dir_path_textbox,
            save_every_n_numberbox,
            resume_traning_checkbox,
            resume_path_textbox,
            logs_checkbox,
            logs_path_textbox,
            noise_offset_slider,
            batch_size_slider,
            gradient_accumulation_steps_slider,
            lr_scheduler_dropdown, 
            cosine_restarts_slider,
            mixed_precision_dropdown,
            save_precision_dropdown,
            warmup_steps_numberbox, 
            unet_lr_numberbox,
            text_encoder_numberbox,
            network_dim_slider,
            network_alpha_slider,
            conv_dim_slider,
            conv_alpha_slider,
            count_type_dropdown,
            count_value_numberbox,
            clip_skip_slider,
            max_token_length_slider,
            stop_text_encoder_training_numberbox,
            generate_samples_checkbox,
            sample_count_type_dropdown,
            sample_count_numberbox,
            path_to_prompts_textbox,
            samplers_dropdown,
            max_grad_norm_slider
        ])
        
    return app
    
if __name__ == '__main__':
    training_gui().launch()