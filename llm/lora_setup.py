from peft import LoraConfig, get_peft_model

def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model
