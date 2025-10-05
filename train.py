import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

# ===================== 配置 =====================
class Config:
    model_name_or_path = "llava-hf/llava-1.5-7b-hf"
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    output_dir = "./llava-nutrition-lora"
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_train_epochs = 5
    logging_steps = 10
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    metric_for_best_model = "rmse"
    fp16 = True
    report_to = "none"
    image_token_index = 32000
    max_length = 1024

    # 归一化缩放因子（根据任务可调整）
    scale_factors = np.array([1000.0, 100.0, 100.0, 200.0])  
    # 分别对应 Calories, Protein, Fat, Carbs

# ===================== 数据集 =====================
class NutritionDataset(Dataset):
    def __init__(self, json_file_path, processor, tokenizer):
        self.data = self.load_json_data(json_file_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = Config.max_length
        self._validate_data_format()
        self._validate_image_files()
        
    def load_json_data(self, json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)
    
    def _validate_data_format(self):
        for i, item in enumerate(self.data):
            if "labels" not in item or len(item["labels"]) != 4:
                raise ValueError(f"数据项 {i} 的 labels 格式不正确")

    def _validate_image_files(self):
        for i, item in enumerate(self.data):
            if not os.path.exists(item['image']):
                raise FileNotFoundError(f"图像文件不存在: {item['image']}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')

        prompt = "<image> Please estimate the nutritional values (calories, protein, fat, carbs) for this food image."

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=Config.max_length
        )

        # === 归一化标签 ===
        labels = np.array(item["labels"], dtype=np.float32) / Config.scale_factors
        labels = torch.tensor(labels, dtype=torch.float32)

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["labels"] = labels
        return inputs

# ===================== 自定义 Trainer =====================
class NutritionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression_head = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        true_batch_size = labels.shape[0]
    
        outputs = model(**inputs, output_hidden_states=True)
    
        last_hidden_state = outputs.hidden_states[-1]
        pooled_features = torch.mean(last_hidden_state, dim=1)
    
        if self.regression_head is None:
            self.regression_head = torch.nn.Linear(pooled_features.shape[-1], 4).to(pooled_features.device)
    
        predictions = self.regression_head(pooled_features)
    
        loss = torch.nn.functional.mse_loss(predictions, labels)
    
        if return_outputs:
            return loss, predictions
        return loss


    

    # 👇 保证 eval 时能返回预测结果
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            pooled_features = torch.mean(last_hidden_state, dim=1)
            predictions = self.regression_head(pooled_features)
            loss = torch.nn.functional.mse_loss(predictions, labels)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, predictions.detach().cpu(), labels.detach().cpu())

# ===================== 指标计算 =====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 反归一化
    predictions = predictions * Config.scale_factors
    labels = labels * Config.scale_factors

    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
    for i, name in enumerate(["calories", "protein", "fat", "carbs"]):
        metrics[f"{name}_mse"] = mean_squared_error(labels[:, i], predictions[:, i])
        metrics[f"{name}_rmse"] = np.sqrt(metrics[f"{name}_mse"])
        metrics[f"{name}_mae"] = mean_absolute_error(labels[:, i], predictions[:, i])
    return metrics

# ===================== 主训练函数 =====================
def main(args):
    config = Config()
    processor = AutoProcessor.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_length

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = NutritionDataset(args.train_json, processor, tokenizer)
    eval_dataset = NutritionDataset(args.test_json, processor, tokenizer)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16,
        report_to=config.report_to,
    )

    trainer = NutritionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained(f"{config.output_dir}/final_model")
    processor.save_pretrained(f"{config.output_dir}/processor")
    tokenizer.save_pretrained(f"{config.output_dir}/tokenizer")
    if trainer.regression_head is not None:
        torch.save(trainer.regression_head.state_dict(), f"{config.output_dir}/regression_head.pth")

# ===================== 入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--test_json', type=str, required=True)
    args = parser.parse_args()
    main(args)
