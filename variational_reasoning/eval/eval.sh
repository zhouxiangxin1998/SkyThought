#!/bin/bash



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task olympiadbench_math_en \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 2 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task math500 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 2 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task mmlu_pro \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 1 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task aime24 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 32 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task aime25 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 32 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task amc23 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 32 \
    --overwrite \
    --result-dir eval_results

sleep 10


## users shoud apply for access of Idavidrein/gpqa on huggingface
CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task gpqa_diamond \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 8 \
    --overwrite \
    --result-dir eval_results

sleep 10



CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task livecodebench_easy \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 8 \
    --overwrite \
    --result-dir eval_results

sleep 10


CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task livecodebench_medium \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 8 \
    --overwrite \
    --result-dir eval_results

sleep 10


CUDA_VISIBLE_DEVICES=0,1,2,3 \
skythought evaluate \
    --model "zhouxiangxin/Variational-Reasoning-4B-Acc" \
    --task livecodebench_hard \
    --backend vllm \
    --backend-args tensor_parallel_size=4,dtype=float32 \
    --sampling-params "temperature=0.7,top_p=1.0,max_tokens=38912,stop=['<|end_of_solution|>']" \
    --n 8 \
    --overwrite \
    --result-dir eval_results

sleep 10


