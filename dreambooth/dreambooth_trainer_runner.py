import concurrent.futures
import os
import json
from datetime import datetime
import re

from tensorboard import program

class DreamboothTrainerRunner():
    def __init__(
            self, 
            last_setup_path: str, 
            notification_fn
        ):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_lora_training_setup = os.path.join(last_setup_path, "last_training_lora.json")
        self.notification_fn = notification_fn

    def submit(self, setup_json):

        project_name = setup_json["project_name"].replace(" ", "_")
        
        resume_path_command=""
        if bool(setup_json["resume_training"]):
            resume_path_command = f"--resume=\"{setup_json['resume_path']}\""

        logging_dir_command=""
        if bool(setup_json["do_logs"]):
            logging_dir_command = f"--logging_dir=\"{setup_json['logs_dir']}\""

        stop_text_encoder_training_command=""
        if setup_json["stop_text_encoder_training"] != 0:
            stop_text_encoder_training_command = f"--stop_text_encoder_training=\"{setup_json['stop_text_encoder_training']}\""

        xformers_command=""
        if setup_json["use_xformers"]:
            xformers_command = "--xformers"

        use_mem_eff_attn_command=""
        if setup_json["use_mem_eff_attn"]:
            use_mem_eff_attn_command = "--mem_eff_attn"
        
        noise_offset_command=""
        if setup_json["noise_offset"] != 0:
            noise_offset = setup_json["noise_offset"]
            noise_offset_command=f"--noise_offset={noise_offset}"
                
        mixed_precision = setup_json["mixed_precision"]
        save_precision = setup_json["save_precision"]
        # precision_part=f"'{precision}'"

        count_command = "--" + ("max_train_epochs" if setup_json["count_type"] == "epoch" else "max_train_steps") + "=" + str(int(setup_json["max_count"]))
        cosine_restarts_command = "" if setup_json["lr_scheduler"] != "cosine_with_restarts" else "--lr_scheduler_num_cycles=" + str(int(setup_json["cosine_restarts"]))

        gradient_steps_command = f'--gradient_accumulation_steps={setup_json["gradient_accumulation_steps"]}'
        generate_samples_command=""
        if bool(setup_json["generate_samples"]):
            sample_count_type = setup_json["sample_count_type"]
            sample_per_count = setup_json["sample_per_count"]
            if sample_count_type == "epoch":
                generate_samples_command += f'  --sample_every_n_epochs={setup_json["sample_per_count"]}'
            else:
                generate_samples_command += f'  --sample_every_n_steps={sample_per_count}'
            
            generate_samples_command += f'  --sample_prompts="{setup_json["sample_prompts_path"]}"'
            generate_samples_command += f'  --sample_sampler="{setup_json["sample_sampler"]}"'
            
   

        project_name = self.transformName(project_name, setup_json)
        setup_json['project_name'] = project_name

        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d_%H-%M-%S')
        new_output_path = os.path.join(setup_json['output_dir'], f"{project_name}_{date_string}")
        if not os.path.exists(new_output_path):
            os.makedirs(new_output_path)
        setup_json['final_output_path'] = new_output_path

        if setup_json['dataset_choice']=="Config path":
            dataset_config = setup_json['dataset_path']
            dataset_command = f'--dataset_config="{dataset_config}"'
        else:
            train_data_dir = setup_json['dataset_path']
            dataset_command = f'--train_data_dir="{train_data_dir}"'

        train_batch_size = int(setup_json['train_batch_size'])
        command_launch = f"""accelerate launch \
        --num_cpu_threads_per_process 1 \
        train_db.py \
        --pretrained_model_name_or_path="{setup_json['model_path']}" \
        {dataset_command} \
        --output_dir="{new_output_path}" \
        --reg_data_dir="{setup_json['reg_dataset_path']}" \
        --train_batch_size="{train_batch_size}" \
        --lr_scheduler={setup_json['lr_scheduler']} \
        {cosine_restarts_command} \
        --lr_warmup_steps={int(setup_json['lr_warmup_steps'])} \
        {count_command} \
        --optimizer_type="AdamW" \
        {gradient_steps_command} \
        {noise_offset_command} \
        --mixed_precision="{mixed_precision}" \
        --save_every_n_epochs={int(setup_json['save_every_n_epochs'])} \
        --seed=4242 \
        --save_precision={save_precision} \
        {logging_dir_command} \
        --save_model_as=safetensors \
        --text_encoder_lr={setup_json['text_encoder_lr']*train_batch_size} \
        --unet_lr={setup_json['unet_lr']*train_batch_size} \
        --max_token_length={setup_json['max_token_length']} \
        --network_dim={setup_json['network_dim']} \
        --network_alpha={setup_json['network_alpha']} \
        --clip_skip={setup_json['clip_skip']} \
        {"--shuffle_caption" if bool(setup_json['shuffle_caption']) else ""} \
        {stop_text_encoder_training_command} \
        --output_name={project_name} \
        {"--save_state" if bool(setup_json['save_states']) else ""} \
        {"--flip_aug" if bool(setup_json['use_flip_aug']) else ""} \
        {"--random_crop" if bool(setup_json['use_random_crop']) else ""} \
        {resume_path_command} \
        {"--cache_latents" if bool(setup_json['use_cache_latents']) else ""} \
        {"--enable_bucket" if bool(setup_json['enable_bucket']) else ""} \
        {"--bucket_no_upscale" if bool(setup_json['bucket_no_upscale']) else ""} \
        {xformers_command} \
        {use_mem_eff_attn_command} \
        {generate_samples_command}"""

        print(command_launch)
        print(f"Submiting task for training. Size of queue: {self.executor._work_queue.qsize() + 1}")
        self.executor.submit(self.process, command_launch, setup_json)
        
        return f"Training in queue! Setup: \n {command_launch}"

    def transformName(self, project_name, setup_json):
        placeholders = re.findall(r'\[(.*?)\]', project_name)
        print(placeholders)
        for placeholder in placeholders:
            value = re.sub(r"\W", "_", str(setup_json.get(placeholder, placeholder)))
            project_name = project_name.replace(f"[{placeholder}]", value)
        return project_name

    def process(self, command_launch, setup_json):
        print(f"Task for training in process. Current queue size: {self.executor._work_queue.qsize()}")
        setup_json_string = json.dumps(setup_json, indent=4)
        
        with open(self.last_lora_training_setup, "w") as file:
            file.write(setup_json_string)
        
        info_file_path = os.path.join(setup_json['final_output_path'], "info.json")
        
        with open(info_file_path, 'w') as file:
            file.write(setup_json_string)

        command_file_path = os.path.join(setup_json['final_output_path'], "command.txt")

        with open(command_file_path, 'w') as file:
            file.write(command_launch)
            
        if setup_json['send_notification']:
            self.notification_fn(setup_json_string+"\nTraining is started!")
        
        tb = program.TensorBoard()
        if (setup_json['logs_dir'] != ""):
            tb.configure(argv=[None, '--logdir', setup_json['logs_dir']])
            # Create a TensorBoard program instance and configure it to serve the log directory

            # Launch TensorBoard
            tb_url = tb.launch()
            if setup_json['send_notification']:
                self.notification_fn(f"TensorBoard URL: {tb_url}")
            print(tb_url)
        
        os.system(f"cd /media/ssd/datasets/apps/training/sd-scripts && {command_launch}")
        
        if setup_json['send_notification']:
            self.notification_fn(f"Training for {setup_json['project_name']} is done! Current queue size: {self.executor._work_queue.qsize()}")
        
            
        if setup_json['logs_dir']!="":
            tb.shutdown()
                
        print(setup_json_string + "\n\n" + "DONE!")