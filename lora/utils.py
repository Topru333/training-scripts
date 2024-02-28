def processIntoConfig(setup_json):
    tomlMap = {}
    if setup_json["lora_modules"] == "Lora-SDXL":
        tomlMap["model_train_type"] = "sdxl-lora"
    else:
        tomlMap["model_train_type"] = "sd-lora"
    tomlMap["output_name"] = setup_json["project_name"]
    tomlMap["output_dir"] = setup_json["output_dir"]
    tomlMap["cache_latents"] = setup_json["save_states"]
    if setup_json["save_states"]:
        tomlMap["cache_latents_to_disk"] = setup_json["save_states"]
    tomlMap["enable_bucket"] = setup_json["enable_bucket"]
    if setup_json["enable_bucket"]:
        tomlMap["bucket_reso_steps"] = 64
    tomlMap["caption_extension"] = ".txt"
    if setup_json["gradient_accumulation_steps"] == 0:
        tomlMap["gradient_checkpointing"] = False
    