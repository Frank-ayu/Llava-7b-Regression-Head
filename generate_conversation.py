import os
import base64
import json
import requests

class FoodImageProcessor:
    def __init__(self, api_key, images_dir, output_file="food_data_en.json"):
        """
        初始化处理器
        :param api_key: 豆包API密钥
        :param images_dir: 菜品图片文件夹路径
        :param output_file: 输出JSON文件路径
        """
        self.api_key = api_key
        self.images_dir = images_dir
        self.output_file = output_file
        self.api_url = "https://api.doubao.com/chat/completions"  # 豆包API地址(请确认最新地址)
        # 英文问题列表，添加了限制只回答问题的提示
        self.questions = [
            "Please tell me the preparation method of this dish. Answer only the question without extra explanation.",
            "Please develop a suitable diet plan for this dish. Answer only the question without extra explanation.",
            "What common allergens might this dish contain? Answer only the question without extra explanation.",
            "What foods is this dish suitable to be paired with? Answer only the question without extra explanation."
        ]

    def encode_image(self, image_path):
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def call_doubao_api(self, image_path, question):
        """调用豆包视觉文字理解API获取回答"""
        try:
            # 编码图片
            base64_image = self.encode_image(image_path)
            
            # 构建请求数据
            payload = {
                "model": "ernie-bot-vilg",  # 请使用正确的多模态模型名称
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image", "image": base64_image}
                        ]
                    }
                ],
                "temperature": 0.7
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功
            
            # 解析响应
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return f"Failed to get answer: {str(e)}"

    def process_images(self):
        """处理文件夹中所有图片并生成训练数据"""
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_files = [
            f for f in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, f)) and
            os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            print("No image files found")
            return
        
        # 处理每个图片
        training_data = []
        total = len(image_files)
        
        for i, filename in enumerate(image_files, 1):
            print(f"Processing image {i}/{total}: {filename}")
            image_path = os.path.join(self.images_dir, filename)
            
            # 为每个问题生成回答
            for question in self.questions:
                print(f"  Generating answer for: {question[:50]}...")
                answer = self.call_doubao_api(image_path, question)
                
                # 添加到训练数据
                training_data.append({
                    "instruction": question.split(". ")[0],  # 移除回答格式限制提示
                    "input": "",
                    "output": answer,
                    "images": [image_path]
                })
        
        # 保存结果
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processing completed. Generated {len(training_data)} entries. Saved to {self.output_file}")

if __name__ == "__main__":
    # 配置参数
    API_KEY = "你的豆包API密钥"  # 替换为你的实际API密钥
    IMAGES_DIRECTORY = "./food_images"  # 替换为你的图片文件夹路径
    OUTPUT_FILE = "food_lora_data_en.json"
    
    # 创建处理器并开始处理
    processor = FoodImageProcessor(API_KEY, IMAGES_DIRECTORY, OUTPUT_FILE)
    processor.process_images()
    