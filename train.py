import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
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

# 配置参数
class Config:
    # 模型参数
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
    # 训练参数
    output_dir = "./llava-nutrition-lora"
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    num_train_epochs = 3
    logging_steps = 10
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    metric_for_best_model = "rmse"
    fp16 = True
    report_to = "none"
    # LLaVA特定参数
    image_token_index = 32000
    max_length = 1024

# 数据集类
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
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_file_path}")
            
        with open(json_file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON文件解析错误: {e}")
                
        return data
    
    def _validate_data_format(self):
        required_keys = ["image", "text_input", "labels"]
        for i, item in enumerate(self.data):
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"数据项 {i} 缺少必要的键: {key}")
            
            if not isinstance(item["labels"], list) or len(item["labels"]) != 4:
                raise ValueError(f"数据项 {i} 的labels格式不正确，需要包含4个数值")
    
    def _validate_image_files(self):
        """验证所有图像文件是否存在"""
        missing_images = []
        for i, item in enumerate(self.data):
            if not os.path.exists(item['image']):
                missing_images.append(f"数据项 {i}: {item['image']}")
        
        if missing_images:
            raise FileNotFoundError(f"以下图像文件不存在:\n" + "\n".join(missing_images))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # 加载图像
            image = Image.open(item['image']).convert('RGB')
        except Exception as e:
            raise Exception(f"加载图像 {item['image']} 失败: {e}")
        
        # 准备输入文本
        base_prompt = "Please estimate the nutritional values (calories, protein, fat, carbs) for this food image."
        prompt = f"<image> {base_prompt}"
        
        # 处理输入
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=Config.max_length
        )
        
        # 将标签转换为回归目标
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        
        # 移除batch维度
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        inputs['labels'] = labels
        return inputs

# 自定义训练器 - 正确处理隐藏状态批次维度
class NutritionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取标签和真实批次大小
        labels = inputs.pop("labels")  # 形状应为 [batch_size, 4]
        true_batch_size = labels.shape[0]
        
        # 显式设置返回隐藏状态
        outputs = model(** inputs, output_hidden_states=True)
        
        # 调试：打印输出结构信息
        # print(f"模型输出类型: {type(outputs)}")
        # if hasattr(outputs, 'hidden_states'):
        #     print(f"隐藏状态类型: {type(outputs.hidden_states)}")
        #     if isinstance(outputs.hidden_states, tuple):
        #         print(f"隐藏状态元组长度: {len(outputs.hidden_states)}")
        #         for i, hs in enumerate(outputs.hidden_states[:2]):  # 只打印前两个元素的形状
        #             print(f"隐藏状态[{i}]形状: {hs.shape}")
        
        # 正确提取隐藏状态 (重新设计的核心部分)
        last_hidden_state = None
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 对于LLaVA，隐藏状态结构可能是 (语言模型隐藏状态, 视觉-语言融合隐藏状态)
            if isinstance(outputs.hidden_states, tuple):
                # 检查哪个元素的第一个维度匹配批次大小
                for hs in outputs.hidden_states:
                    if hs.shape[0] == true_batch_size:
                        last_hidden_state = hs
                        break
                # 如果没找到，尝试第二个维度（有些模型可能将批次放在第二维）
                if last_hidden_state is None:
                    for hs in outputs.hidden_states:
                        if hs.ndim >= 2 and hs.shape[1] == true_batch_size:
                            last_hidden_state = hs.transpose(0, 1)  # 转置批次维度到第一维
                            break
            
            # 如果还是没找到，使用最后一层隐藏状态
            if last_hidden_state is None:
                if isinstance(outputs.hidden_states, tuple):
                    last_hidden_state = outputs.hidden_states[-1]
                else:
                    last_hidden_state = outputs.hidden_states
        
        # 如果无法从hidden_states获取，尝试其他方式
        if last_hidden_state is None:
            print("使用logits作为特征来源")
            logits = outputs.logits
            # 检查logits的批次维度
            if logits.shape[0] == true_batch_size:
                last_hidden_state = logits
            elif logits.ndim >= 2 and logits.shape[1] == true_batch_size:
                last_hidden_state = logits.transpose(0, 1)
            else:
                # 最后的备选方案：取平均并调整形状
                last_hidden_state = torch.mean(logits, dim=1, keepdim=True)
                if last_hidden_state.shape[0] != true_batch_size:
                    last_hidden_state = last_hidden_state[:true_batch_size]
        
        # 确保批次维度正确
        if last_hidden_state.shape[0] != true_batch_size:
            # 打印调试信息
            print(f"紧急修复：隐藏状态形状 {last_hidden_state.shape}，批次大小 {true_batch_size}")
            # 尝试切片以匹配批次大小（紧急修复）
            if last_hidden_state.shape[0] > true_batch_size:
                last_hidden_state = last_hidden_state[:true_batch_size]
            else:
                # 如果小于批次大小，重复填充（不推荐，但为了继续训练）
                repeat_factor = (true_batch_size // last_hidden_state.shape[0]) + 1
                last_hidden_state = last_hidden_state.repeat(repeat_factor, *([1]*(last_hidden_state.ndim-1)))[:true_batch_size]
        
        # 提取特征（使用均值池化确保批次维度正确）
        # 不管序列长度，直接对序列维度求平均
        if last_hidden_state.ndim >= 3:  # [batch_size, seq_len, hidden_size]
            pooled_features = torch.mean(last_hidden_state, dim=1)  # [batch_size, hidden_size]
        else:  # [batch_size, hidden_size]
            pooled_features = last_hidden_state
        
        # 确保池化后的批次维度正确
        if pooled_features.shape[0] != true_batch_size:
            raise ValueError(f"池化后批次大小 {pooled_features.shape[0]} 与标签批次大小 {true_batch_size} 不匹配")
        
        # 初始化回归头
        if not hasattr(self, 'regression_head'):
            self.regression_head = torch.nn.Linear(pooled_features.shape[-1], 4).to(pooled_features.device)
        
        # 获取预测结果
        predictions = self.regression_head(pooled_features)  # 应该是 [batch_size, 4]
        
        # 检查最终形状
        if predictions.shape != labels.shape:
            print(f"预测形状: {predictions.shape}, 标签形状: {labels.shape}")
            print(f"池化特征形状: {pooled_features.shape}")
            print(f"隐藏状态形状: {last_hidden_state.shape}")
            raise ValueError(f"预测形状 {predictions.shape} 与标签形状 {labels.shape} 不匹配")
        
        # 计算MSE损失
        loss = torch.nn.functional.mse_loss(predictions, labels)
        
        return (loss, {"predictions": predictions, "labels": labels}) if return_outputs else loss

# 评估指标计算
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions["predictions"].squeeze()
    labels = eval_pred.predictions["labels"].squeeze()
    
    # 确保评估时形状匹配
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    
    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
    for i, name in enumerate(["calories", "protein", "fat", "carbs"]):
        metrics[f"{name}_mse"] = mean_squared_error(labels[:, i], predictions[:, i])
        metrics[f"{name}_rmse"] = np.sqrt(metrics[f"{name}_mse"])
        metrics[f"{name}_mae"] = mean_absolute_error(labels[:, i], predictions[:, i])
    
    return metrics

# 加载量化配置
def get_quantization_config(config):
    return BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

# 配置LoRA
def get_lora_config(config):
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

# 主函数
def main(args):
    config = Config()
    
    print(f"Loading model: {config.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_length
    
    # 量化配置
    quantization_config = get_quantization_config(config)
    
    # 加载LLaVA模型
    model = LlavaForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    # 配置LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("Loading datasets...")
    try:
        train_dataset = NutritionDataset(args.train_json, processor, tokenizer)
        eval_dataset = NutritionDataset(args.test_json, processor, tokenizer)
        print(f"成功加载训练集: {len(train_dataset)} 样本")
        print(f"成功加载测试集: {len(eval_dataset)} 样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 定义训练参数
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
        gradient_checkpointing=False
    )
    
    # 初始化训练器
    trainer = NutritionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    model.save_pretrained(f"{config.output_dir}/final_model")
    processor.save_pretrained(f"{config.output_dir}/processor")
    tokenizer.save_pretrained(f"{config.output_dir}/tokenizer")
    
    # 最终评估
    print("Final evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    
    with open(f"{config.output_dir}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

# 推理函数
def predict_nutrition(model, processor, tokenizer, image_path, text_input=None):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise Exception(f"加载图像 {image_path} 失败: {e}")
    
    if text_input is None:
        text_input = "Please estimate the nutritional values (calories, protein, fat, carbs) for this food image."
    
    prompt = f"<image> {text_input}"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        max_length=Config.max_length
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model(** inputs, output_hidden_states=True)
    
    # 处理隐藏状态
    last_hidden_state = None
    true_batch_size = inputs['input_ids'].shape[0]
    
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        if isinstance(outputs.hidden_states, tuple):
            for hs in outputs.hidden_states:
                if hs.shape[0] == true_batch_size:
                    last_hidden_state = hs
                    break
            if last_hidden_state is None:
                for hs in outputs.hidden_states:
                    if hs.ndim >= 2 and hs.shape[1] == true_batch_size:
                        last_hidden_state = hs.transpose(0, 1)
                        break
        if last_hidden_state is None:
            last_hidden_state = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
    
    if last_hidden_state is None:
        logits = outputs.logits
        last_hidden_state = torch.mean(logits, dim=1, keepdim=True)
    
    # 池化操作
    if last_hidden_state.ndim >= 3:
        pooled_features = torch.mean(last_hidden_state, dim=1)
    else:
        pooled_features = last_hidden_state
    
    # 预测
    regression_head = torch.nn.Linear(pooled_features.shape[-1], 4).to(pooled_features.device)
    predictions = regression_head(pooled_features).squeeze().cpu().numpy()
    
    nutrients = ["Calories", "Protein", "Fat", "Carbs"]
    units = ["kcal", "g", "g", "g"]
    result = ", ".join([f"{n}: {p:.1f} {u}" for n, p, u in zip(nutrients, predictions, units)])
    
    return result, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练食物营养成分回归模型')
    parser.add_argument('--train_json', type=str, required=True, help='训练集JSON文件路径')
    parser.add_argument('--test_json', type=str, required=True, help='测试集JSON文件路径')
    args = parser.parse_args()
    
    main(args)
