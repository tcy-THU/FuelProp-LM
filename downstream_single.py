import os
import sys
import time
import fire
import torch
from datasets import load_dataset
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

from prompter import Prompter
import numpy as np
import re as regex
from tqdm import tqdm
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

from sklearn.metrics import (r2_score, roc_auc_score)
import pandas as pd

def main(
    CLI: bool = False,
    protein: bool = False,
    load_8bit: bool = True,
    base_model: str = "./models/Qwen2.5-7B-Instruct",
    lora_weights: str = "./models/Qwen_100000",
    # lora_weights: str = "",
    prompt_template: str = "",  
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
    path: str = "./data",
    shots: list = None,  # 改为接受列表，默认为None
    batch_size: int =64,
    model_name = "qwen_100000"
):
    # 如果没有指定shots，使用默认值
    if shots is None:
        shots = [0, 1, 4, 8]
    
    print(f"将对以下shot值进行测试: {shots}")
    print("="*50)
    
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model"

    prompter = Prompter(prompt_template)

    # ========== 只加载一次模型和tokenizer ==========
    print("开始加载模型和tokenizer...")
    start_time = time.time()
    
    # 对于Gemma模型，可能需要特殊处理
    if "gemma" in base_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "left"

    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    if tokenizer.pad_token is None or tokenizer.pad_token == "!":
        print('set pad token')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device == "cuda":
        print(f'从{base_model}路径加载基础模型')
        torch.cuda.empty_cache()
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_8bit,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa"
        )

    if lora_weights and lora_weights.strip():
        print(f'从{lora_weights}路径加载微调后lora权重')    
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
    else:
        print("未提供LoRA权重，使用基础模型进行推理")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    print("="*50)

    # # 在模型加载后立即添加
    # print("\n=== 词汇表大小不匹配处理 ===")
    # print(f"Tokenizer词汇表大小: {len(tokenizer)}")
    # print(f"模型embedding大小: {model.get_input_embeddings().num_embeddings}")
    # # 由于tokenizer的词汇表小于模型，我们需要确保不使用超出tokenizer范围的token
    # # 但是image_token_index是262144，刚好在tokenizer范围内，所以应该是安全的
    # # 检查特殊token是否在有效范围内
    # special_tokens = {
    #     'boi_token': model.config.boi_token_index,  # 255999
    #     'eoi_token': model.config.eoi_token_index,  # 256000
    #     'image_token': model.config.image_token_index  # 262144
    # }
    # for name, idx in special_tokens.items():
    #     if idx >= len(tokenizer):
    #         print(f"警告: {name} (ID: {idx}) 超出了tokenizer词汇表范围!")
    
    # ========== 定义所有辅助函数 ==========
    label_ignore = [-100]
    raw_label = {1: "Yes", 0: "No", 'invalid': label_ignore}
    label_y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[1]))
    label_n = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[0]))
    label_dict = {1: label_y, 0: label_n, 'invalid': label_ignore}        
    
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,truncation=True,max_length=4096,padding=False,return_tensors=None)

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 4096
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )

        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
            
        tokenized_user_prompt["labels"] = tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
 
        return tokenized_user_prompt 

    def extract_number_from_text(text):
        boxed_match = regex.search(r'\$\\boxed\{([-+]?\d+(?:\.\d+)?)\}\$', text)
        if boxed_match:
            return boxed_match.group(1)
        
        boxed_match2 = regex.search(r'\\boxed\{([-+]?\d+(?:\.\d+)?)\}', text)
        if boxed_match2:
            return boxed_match2.group(1)
        
        answer_match = regex.search(r'answer\s+is[:\s]+([+-]?\d+(?:\.\d+)?)', text, regex.IGNORECASE)
        if answer_match:
            return answer_match.group(1)
        
        float_match = regex.search(r'[-+]?\d+\.\d+', text)
        if float_match:
            return float_match.group(0)
        
        int_match = regex.search(r'[-+]?\d+', text)
        if int_match:
            return int_match.group(0)
        
        return text

    def evaluate(
        instruction,
        input=None,
        output=None,
        temperature=0.1,
        repetition_penalty=1,
        max_new_tokens=128,
        **kwargs,
    ):
        modified_instruction = instruction + " Just give me a single numerical value without any explanation or additional text."
        prompt = prompter.generate_prompt(modified_instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        do_sample=False
        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            
            text = tokenizer.decode(generation_output.sequences[0])
            re=prompter.get_response(text)
            re_ori = re

            if not lora_weights or not lora_weights.strip():
                print('try to convert')
                re = extract_number_from_text(re)
            else:
                re=tokenizer(re)         
                re=tokenizer.decode(re['input_ids'][1:-1])
            
        return float(output), float(re)

    def evaluate_batch(
        instructions,
        inputs,
        outputs,
        temperature=0.1,
        repetition_penalty=1,
        max_new_tokens=128,
        **kwargs,
    ):
        prompts = []
        for instruction, input_text in zip(instructions, inputs):
            modified_instruction = instruction + " Just give me a single numerical value without any explanation or additional text."
            prompt = prompter.generate_prompt(modified_instruction, input_text)
            prompts.append(prompt)
        
        inputs_batch = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096
        ).to(device)
        
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_outputs = model.generate(
                input_ids=inputs_batch["input_ids"],
                attention_mask=inputs_batch["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        
        results = []
        for i in range(len(instructions)):
            text = tokenizer.decode(generation_outputs.sequences[i],skip_special_tokens=True)
            re=prompter.get_response(text)
            

            # # 添加调试信息
            # print(f"\n样本 {i}:")
            # print(f"原始生成文本: {text[:200]}...")  # 只打印前200个字符
            # print(f"提取的响应: '{re}'")
            
            # 无论是否使用lora权重，都尝试提取数字
            extracted_number = extract_number_from_text(re)
            # print(f"提取的数字: '{extracted_number}'")

            # if not lora_weights or not lora_weights.strip():
            #     print('原始模型，需要提取')
            #     re = extract_number_from_text(re)
            # else:
            #     re=tokenizer(re)
            #     print(f'token {re}')
            #     re=tokenizer.decode(re['input_ids'][1:-1])
            #     print(f'decode {re}')
                
            try:
                pred_value = float(extracted_number)
                # print(f"成功转换为数字: {pred_value}")
            except:
                print(f"批次中第{i}个样本无法转换为数字: '{extracted_number}'")
                pred_value = 0.0
                
            results.append((float(outputs[i]), pred_value))
        
        return results

    # ========== 主循环：遍历所有shot值 ==========
    all_results = {}  # 存储所有结果
    for shot in shots:
        print(f"\n{'='*60}")
        print(f"开始运行 {shot}-shot 测试")
        print(f"{'='*60}")
        
        path_0 = os.path.join(path, str(shot)+ "-shot")
        data = []
        data_score = []
        
        # 检查路径是否存在
        if not os.path.exists(path_0):
            print(f"警告: 路径 {path_0} 不存在，跳过 {shot}-shot")
            continue
        
        file_list = os.listdir(path_0)
        print(f"开始测试，共有{len(file_list)}个数据集")
        
        for file_idx, f in enumerate(file_list):
            print(f"\n当前处理第 [{file_idx+1}/{len(file_list)}] 个数据集: {f}")
            
            path1 = os.path.join(path_0, f)
            data.append(f.split(".")[0])
            raw_datasets_val = load_dataset("json", data_files=path1)
            val_data = raw_datasets_val["train"]
            
            print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print(f"数据集 {f} 包含 {len(val_data)} 个样本")
            print(f"使用批处理大小: {batch_size}")
            
            labels_list = []
            predict_list = []
            detailed_results = []  # 🔴 修复1：初始化detailed_results列表
            
            # 批处理
            for i in tqdm(range(0, len(val_data), batch_size), desc=f"处理 {f} (批次)"):
                batch_end = min(i + batch_size, len(val_data))
                
                instructions = []
                inputs_text = []
                outputs = []
                
                for idx in range(i, batch_end):
                    instructions.append(val_data[idx]['instruction'])
                    inputs_text.append(val_data[idx]['input'])
                    outputs.append(val_data[idx]['output'])
                
                try:
                    batch_results = evaluate_batch(
                        instructions, inputs_text, outputs,
                        temperature=0.1, repetition_penalty=1, 
                        max_new_tokens=128
                    )
                    
                    for j, (true_val, pred_val) in enumerate(batch_results):  # 🔴 修复2：添加enumerate来定义j
                        labels_list.append(true_val)
                        predict_list.append(pred_val)
                        # ========== 新增：保存详细信息 ==========
                        current_idx = i + j
                        detailed_results.append({
                            'index': current_idx,
                            'instruction': val_data[current_idx]['instruction'],
                            'input': val_data[current_idx]['input'],
                            'true_value': true_val,
                            'pred_value': pred_val,
                            'absolute_error': abs(pred_val - true_val),
                            'relative_error': abs(pred_val - true_val) / abs(true_val) * 100 if true_val != 0 else (100 if pred_val != 0 else 0)
                        })
                        
                except Exception as e:
                    print(f"批处理失败，回退到单个处理。错误: {e}")
                    for idx in range(i, batch_end):
                        try:
                            output, re = evaluate(
                                val_data[idx]['instruction'], 
                                val_data[idx]['input'], 
                                val_data[idx]['output'],
                                temperature=0.1, repetition_penalty=1, max_new_tokens=128
                            )
                            labels_list.append(output)
                            predict_list.append(re)
                            # ========== 新增：保存详细信息（单个处理） ==========
                            detailed_results.append({
                                'index': idx,
                                'instruction': val_data[idx]['instruction'],
                                'input': val_data[idx]['input'],
                                'true_value': output,
                                'pred_value': re,
                                'absolute_error': abs(re - output),
                                'relative_error': abs(re - output) / abs(output) * 100 if output != 0 else (100 if re != 0 else 0)
                            })
                        except Exception as e2:
                            print(f"处理样本 {idx} 时发生错误: {e2}")
                            continue
                            
                if (i + batch_size) % (batch_size * 5) == 0:
                    print(f"已处理 {min(i + batch_size, len(val_data))}/{len(val_data)} 个样本")
            
            model_type = "base" if not lora_weights or not lora_weights.strip() else "finetuned"
            
            # 创建保存目录
            detail_dir = f'./downstream_test0520/{model_type}/{model_name}/detailed_results/{shot}-shot'
            os.makedirs(detail_dir, exist_ok=True)
            
            # 保存详细结果
            if detailed_results:
                detailed_df = pd.DataFrame(detailed_results)
                dataset_name = f.split(".")[0]
                detail_path = os.path.join(detail_dir, f'{dataset_name}_detailed.csv')
                detailed_df.to_csv(detail_path, index=False)
                print(f"详细预测结果已保存到: {detail_path}")
            # 计算相对误差
            relative_errors = [abs(pred - true) / abs(true) * 100 if true != 0 else (100 if pred != 0 else 0) 
                            for pred, true in zip(predict_list, labels_list)]
            score = sum(relative_errors) / len(relative_errors) if relative_errors else 0
            
            print(f"数据集 {f} 的平均相对误差: {score:.2f}%")
            data_score.append(float(score))
            
            # 保存当前结果
            df = pd.DataFrame({'dataset': data, 'score': data_score})
            model_type = "base" if not lora_weights or not lora_weights.strip() else "finetuned"
            
            os.makedirs(f'./downstream_test0520/{model_type}/{model_name}', exist_ok=True)
            save_path = f'./downstream_test0520/{model_type}/{model_name}/{f.split(".")[0]}_{shot}_{model_type}_RE.csv'
            df.to_csv(save_path)
            print(f"已将结果保存到 {save_path}")
            
            print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print(f"总进度: {file_idx+1}/{len(file_list)} 数据集已完成")
            print("-" * 50)
        
        # 保存该shot的汇总结果
        all_results[shot] = {'data': data, 'scores': data_score}
        print(f"\n{shot}-shot 测试完成！")
    
    # ========== 生成汇总报告 ==========
    print(f"\n{'='*60}")
    print("所有测试完成！汇总结果：")
    print(f"{'='*60}")
    
    for shot, result in all_results.items():
        if result['scores']:
            avg_score = sum(result['scores']) / len(result['scores'])
            print(f"{shot}-shot 平均相对误差: {avg_score:.2f}%")
    
    # 可选：保存汇总结果
    summary_data = []
    for shot in all_results:
        for dataset, score in zip(all_results[shot]['data'], all_results[shot]['scores']):
            summary_data.append({
                'shot': shot,
                'dataset': dataset,
                'score': score
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = f'./downstream_test0520/{model_type}/{model_name}/summary_{model_type}_RE.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n汇总结果已保存到: {summary_path}")

if __name__ == "__main__":
    fire.Fire(main)
