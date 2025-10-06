#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoModel, AutoTokenizer
from verl.workers.rollout.vllm_rollout import vLLMRollout
# Replace with your desired model name (from https://huggingface.co/models)
model_name = "emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold"

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# from /u/zluo8/sim2real/ToolRL/verl/trainer/main_ppo.py line 116
config = {
    'enforce_eager': True,
    'free_cache_engine': True,
    'prompt_length': 2048,
    'response_length': 1024,
    'dtype': 'bfloat16',
    'gpu_memory_utilization': 0.6,
    'load_format': 'dummy_dtensor',
}
rollout = vLLMRollout(actor_module=model, config=config)


# In[2]:


# load dataset

import pyarrow.parquet as pq

# Load the Parquet file
table = pq.read_table("../dataset/rlla_4k/test.parquet")

# Convert to a record batch (like a table of rows)
records = table.to_pylist()  # This is a list of dicts, one per row


# In[ ]:


print("Model and tokenizer loaded successfully.")


for idx, record in enumerate(records[:3]):  # Display the first 3 records
    print(f'----- Record {idx} -----')
    prompt = record['prompt']
    print("Prompt:\n", prompt)
    # This is from /u/zluo8/sim2real/ToolRL/verl/utils/dataset/rl_dataset.py line 131
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    output = rollout.generate_sequences([prompt])
    print("Output:\n", output[0])


# In[ ]:




