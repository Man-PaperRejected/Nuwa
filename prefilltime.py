import torch
import time
import json
import numpy as np
from pathlib import Path
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from PIL import Image
import pdb
import types


from lmms_eval.models.llava import Llava

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



class Predictor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        
        
        pretrained = "/data/model/llava-v1.5-13b"
        llava = Llava(pretrained)
        self.model = llava.model
        self.tokenizer = llava.tokenizer
        self.image_processor = llava._image_processor



    def measure_prefill_time(self, image_path: Path, prompt: str) -> float:
        """
        Runs a single prefill pass on the model and returns the execution time in milliseconds.
        Does not generate any text.
        """
        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        

        image_data = load_image(str(image_path))
        
        
        
        image_tensor = process_images([image_data], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None) 
        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
            for _ in range(3):
                # device warnup
                # pdb.set_trace()
                _ = self.model(
                    input_ids=input_ids,
                    images=image_tensor,
                    use_cache=True 
                )
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = self.model(input_ids=input_ids,images=image_tensor,use_cache=True)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()

        prefill_duration_ms = (end_time - start_time) * 1000
        return prefill_duration_ms, input_ids.shape[1]


def evaluate_batch_prefill_time(data_root, json_file, task="grounding", num_samples=None):

    predictor = Predictor()
    
    with open(json_file, 'r') as f:
        samples = json.load(f)
    
    if num_samples is not None:
        samples = samples[:num_samples]
    
    print(f"Processing {len(samples)} samples from {json_file}")
    
    prefill_times = []
    token_lengths = []
    
    for i, sample in enumerate(samples):
        image_path = data_root / sample['image']
        description = sample['normal_caption']
        
        prompt = f"Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y). All values are floating point numbers bounded between 0 and 1. Please provide the bounding box coordinate of the region this sentence describes: {description}." \
                if task=="grounding" else f"Answer Yes or No. Is there an object in this sentence describes: {description}."
        
        print(f"[{i+1}/{len(samples)}] Processing: {image_path}")
        prefill_time, token_length = predictor.measure_prefill_time(image_path, prompt)
        
        if prefill_time is not None:
            prefill_times.append(prefill_time)
            token_lengths.append(token_length)
            print(f"  - Prefill time: {prefill_time:.2f} ms, Tokens: {token_length}")
    
    if prefill_times:
        avg_prefill_time = np.mean(prefill_times)
        std_prefill_time = np.std(prefill_times)
        min_prefill_time = np.min(prefill_times)
        max_prefill_time = np.max(prefill_times)
        avg_token_length = np.mean(token_lengths)
        
        print("\n===== Prefill Time Statistics =====")
        print(f"Average prefill time: {avg_prefill_time:.2f} ms")
        print(f"Standard deviation: {std_prefill_time:.2f} ms")
        print(f"Min prefill time: {min_prefill_time:.2f} ms")
        print(f"Max prefill time: {max_prefill_time:.2f} ms")
        print(f"Average token length: {avg_token_length:.1f} tokens")
        print(f"Throughput: {avg_token_length / (avg_prefill_time / 1000):.2f} tokens/second")
        
        return {
            "avg_prefill_time_ms": avg_prefill_time,
            "std_prefill_time_ms": std_prefill_time,
            "min_prefill_time_ms": min_prefill_time,
            "max_prefill_time_ms": max_prefill_time,
            "avg_token_length": avg_token_length,
            "throughput_tokens_per_second": avg_token_length / (avg_prefill_time / 1000),
            "num_samples": len(prefill_times)
        }
    else:
        print("No valid samples processed.")
        return None

if __name__ == "__main__":
    data_root = Path("coco_data")
    json_file = data_root / "refcoco.json"
    

    num_samples = None
    
    task = "grounding"  
    
    results = evaluate_batch_prefill_time(data_root, json_file, task, num_samples)
    
    if results:
        with open("prefill_time_results.json", "w") as f:
            json.dump(results, f, indent=4)