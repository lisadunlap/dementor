"""
Push all 76 OpenAssistant LoRA adapters (40 SFT + 36 DPO) to HuggingFace.

Usage:
  python scripts/tools/push_oasst_76adapters_to_hf.py --hf-token hf_...
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

HF_ORG = "dementor-research"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REGISTRY = Path("data/tinker_adapters.json")

ADAPTERS: List[Dict[str, str]] = [
    {
        "tinker_path": "tinker://7b29d209-ac68-53e7-b22b-1ae4b72a323b:train:0/sampler_weights/oasst_gpt-oss-20b_as_llama-3.1-8b_sft_seed42_20260605224622",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-sft-seed42",
    },
    {
        "tinker_path": "tinker://67d09a6d-309c-562a-b9a6-a8109c10000f:train:0/sampler_weights/oasst_gpt-oss-20b_as_llama-3.1-8b_sft_seed43_20260605224523",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-sft-seed43",
    },
    {
        "tinker_path": "tinker://9b5a774d-f7c3-5aef-a5b7-2009fb9fb2bd:train:0/sampler_weights/oasst_gpt-oss-20b_as_llama-3.1-8b_sft_seed44_20260605224519",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-sft-seed44",
    },
    {
        "tinker_path": "tinker://edb233d5-65c1-57e4-89ad-001f533cc469:train:0/sampler_weights/oasst_gpt-oss-20b_as_nemotron-nano_sft_seed42_20260605224533",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-sft-seed42",
    },
    {
        "tinker_path": "tinker://c2fea077-6b2a-504c-b8cd-85f9cd7fc35c:train:0/sampler_weights/oasst_gpt-oss-20b_as_nemotron-nano_sft_seed43_20260605224623",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-sft-seed43",
    },
    {
        "tinker_path": "tinker://59f97409-b395-5dbc-a0af-f13f963e75fc:train:0/sampler_weights/oasst_gpt-oss-20b_as_nemotron-nano_sft_seed44_20260605224626",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-sft-seed44",
    },
    {
        "tinker_path": "tinker://c995ac59-2355-5a75-96c0-c9b948aa8d01:train:0/sampler_weights/oasst_gpt-oss-20b_as_qwen3.6-27b_sft_seed42_20260605224523",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-sft-seed42",
    },
    {
        "tinker_path": "tinker://dbf5adb7-d7cb-55ca-b03d-a4a6b04cd9e2:train:0/sampler_weights/oasst_gpt-oss-20b_as_qwen3.6-27b_sft_seed43_20260605224622",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-sft-seed43",
    },
    {
        "tinker_path": "tinker://1473b144-4f8c-5307-be10-d77497715029:train:0/sampler_weights/oasst_gpt-oss-20b_as_qwen3.6-27b_sft_seed44_20260605224614",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-sft-seed44",
    },
    {
        "tinker_path": "tinker://2c61a1d5-e305-5d0b-87fc-74a5ec502441:train:0/sampler_weights/oasst_gpt-oss-20b_self_sft_seed42_20260605224530",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-gpt-oss-20b-sft-seed42",
    },
    {
        "tinker_path": "tinker://b129eac6-8a10-5a16-95f3-53b19bb7e72c:train:0/sampler_weights/oasst_llama-3.1-8b_as_gpt-oss-20b_sft_seed42_20260605224457",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-sft-seed42",
    },
    {
        "tinker_path": "tinker://d26339d5-ce61-52ce-8ace-2b3b2e6e8f4e:train:0/sampler_weights/oasst_llama-3.1-8b_as_gpt-oss-20b_sft_seed43_20260605224457",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-sft-seed43",
    },
    {
        "tinker_path": "tinker://fad692d3-2fc1-5a2a-b286-a6e164743917:train:0/sampler_weights/oasst_llama-3.1-8b_as_gpt-oss-20b_sft_seed44_20260605224502",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-sft-seed44",
    },
    {
        "tinker_path": "tinker://f29c82df-6227-59cc-a635-4ad8d2eb407c:train:0/sampler_weights/oasst_llama-3.1-8b_as_nemotron-nano_sft_seed42_20260605224455",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-sft-seed42",
    },
    {
        "tinker_path": "tinker://0edb0537-ea41-5cc8-9ef5-7617a69334fa:train:0/sampler_weights/oasst_llama-3.1-8b_as_nemotron-nano_sft_seed43_20260605224458",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-sft-seed43",
    },
    {
        "tinker_path": "tinker://55112ca7-b247-5770-9335-f35df0d5ccb6:train:0/sampler_weights/oasst_llama-3.1-8b_as_nemotron-nano_sft_seed44_20260605224459",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-sft-seed44",
    },
    {
        "tinker_path": "tinker://dcb1eb33-43da-5296-b00b-071f2cce68e3:train:0/sampler_weights/oasst_llama-3.1-8b_as_qwen3.6-27b_sft_seed42_20260605224533",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-sft-seed42",
    },
    {
        "tinker_path": "tinker://15064109-1679-54d5-b458-be226b0e9baa:train:0/sampler_weights/oasst_llama-3.1-8b_as_qwen3.6-27b_sft_seed43_20260605224458",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-sft-seed43",
    },
    {
        "tinker_path": "tinker://194cebb7-4af0-50b4-b526-bd8b93154acc:train:0/sampler_weights/oasst_llama-3.1-8b_as_qwen3.6-27b_sft_seed44_20260605224511",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-sft-seed44",
    },
    {
        "tinker_path": "tinker://75aa5598-6b80-5914-a463-d2d31708cf93:train:0/sampler_weights/oasst_llama-3.1-8b_self_sft_seed42_20260605224457",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-llama-3.1-8b-sft-seed42",
    },
    {
        "tinker_path": "tinker://374c1a92-97c8-56e5-9765-4d7c86a29b40:train:0/sampler_weights/oasst_nemotron-nano_as_gpt-oss-20b_sft_seed42_20260605224821",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-sft-seed42",
    },
    {
        "tinker_path": "tinker://8141bf31-c5f5-5c97-8bd5-b086e16f2f0d:train:0/sampler_weights/oasst_nemotron-nano_as_gpt-oss-20b_sft_seed43_20260605224830",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-sft-seed43",
    },
    {
        "tinker_path": "tinker://cf58abf7-8311-547a-a507-4bb14b1383db:train:0/sampler_weights/oasst_nemotron-nano_as_gpt-oss-20b_sft_seed44_20260605225618",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-sft-seed44",
    },
    {
        "tinker_path": "tinker://de2312de-f44a-5941-8e3f-85441d58d734:train:0/sampler_weights/oasst_nemotron-nano_as_llama-3.1-8b_sft_seed42_20260605224827",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-sft-seed42",
    },
    {
        "tinker_path": "tinker://c7b67d1d-2fdd-5992-b601-b64cc34c2955:train:0/sampler_weights/oasst_nemotron-nano_as_llama-3.1-8b_sft_seed43_20260605224827",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-sft-seed43",
    },
    {
        "tinker_path": "tinker://c65245e1-5809-51aa-aafe-51f011ab7bea:train:0/sampler_weights/oasst_nemotron-nano_as_llama-3.1-8b_sft_seed44_20260605224821",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-sft-seed44",
    },
    {
        "tinker_path": "tinker://1153c432-f24a-58ed-a8e6-48849f632fab:train:0/sampler_weights/oasst_nemotron-nano_as_qwen3.6-27b_sft_seed42_20260605224827",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-sft-seed42",
    },
    {
        "tinker_path": "tinker://94ab4ad0-4e02-5d5a-8d78-4e4d18b28a57:train:0/sampler_weights/oasst_nemotron-nano_as_qwen3.6-27b_sft_seed43_20260605224822",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-sft-seed43",
    },
    {
        "tinker_path": "tinker://951942d7-746a-51f0-949e-8d33cb0ed11c:train:0/sampler_weights/oasst_nemotron-nano_as_qwen3.6-27b_sft_seed44_20260605224827",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-sft-seed44",
    },
    {
        "tinker_path": "tinker://28d0ca01-e98e-5efe-b9c4-61a6c46e09f5:train:0/sampler_weights/oasst_nemotron-nano_self_sft_seed42_20260605224821",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-nemotron-nano-sft-seed42",
    },
    {
        "tinker_path": "tinker://7cad2973-d153-512d-9161-988da87bca7e:train:0/sampler_weights/oasst_qwen3.6-27b_as_gpt-oss-20b_sft_seed42_20260605225002",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-sft-seed42",
    },
    {
        "tinker_path": "tinker://11191603-f14b-5fbb-a8df-8a39711260e0:train:0/sampler_weights/oasst_qwen3.6-27b_as_gpt-oss-20b_sft_seed43_20260605224956",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-sft-seed43",
    },
    {
        "tinker_path": "tinker://e02d751f-0f4c-528b-8665-989c4601abf8:train:0/sampler_weights/oasst_qwen3.6-27b_as_gpt-oss-20b_sft_seed44_20260605225003",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-sft-seed44",
    },
    {
        "tinker_path": "tinker://cae3e61e-2164-560e-a289-17c6b355f5f4:train:0/sampler_weights/oasst_qwen3.6-27b_as_llama-3.1-8b_sft_seed42_20260605225002",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-sft-seed42",
    },
    {
        "tinker_path": "tinker://5143a776-885c-57f9-bd42-8d526cef65c7:train:0/sampler_weights/oasst_qwen3.6-27b_as_llama-3.1-8b_sft_seed43_20260605225002",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-sft-seed43",
    },
    {
        "tinker_path": "tinker://56cd6a52-21c7-530b-801e-78ffbf8694dd:train:0/sampler_weights/oasst_qwen3.6-27b_as_llama-3.1-8b_sft_seed44_20260605225003",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-sft-seed44",
    },
    {
        "tinker_path": "tinker://17ab7df6-ce8c-50f5-9299-629a3c6388cc:train:0/sampler_weights/oasst_qwen3.6-27b_as_nemotron-nano_sft_seed42_20260605225003",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-sft-seed42",
    },
    {
        "tinker_path": "tinker://ca86201e-4d4d-56f7-a521-50c17fce871a:train:0/sampler_weights/oasst_qwen3.6-27b_as_nemotron-nano_sft_seed43_20260605225002",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-sft-seed43",
    },
    {
        "tinker_path": "tinker://b09ce90c-797c-5073-8220-4498934ccacf:train:0/sampler_weights/oasst_qwen3.6-27b_as_nemotron-nano_sft_seed44_20260605225003",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-sft-seed44",
    },
    {
        "tinker_path": "tinker://a2aed764-5d98-5244-a2fd-235a881511a6:train:0/sampler_weights/oasst_qwen3.6-27b_self_sft_seed42_20260605225002",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-qwen3.6-27b-sft-seed42",
    },
    {
        "tinker_path": "tinker://09607f10-3841-5f6d-95f5-378d351bf468:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://8bb917ed-12b1-5f67-87c5-23b67e4cfc3c:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://bc8b4dc0-050e-5b38-bda8-8324dfa34f55:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-llama-3.1-8b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://f78c5144-f8a2-5b47-bb74-43a43e8883b1:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-dpo-seed42",
    },
    {
        "tinker_path": "tinker://d3e6f95c-3819-5621-afdf-400bfd615e78:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-dpo-seed43",
    },
    {
        "tinker_path": "tinker://ba6f2d5e-af49-5549-8701-55f2bf38923e:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-nemotron-nano-dpo-seed44",
    },
    {
        "tinker_path": "tinker://fbc24775-ab2f-512a-88c7-723ffaa32375:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://ac63d3cf-9fb5-5a3c-8d14-0e969ecee430:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://dfa3fd26-36a4-5db9-974d-b341935016a2:train:0/weights/final",
        "repo_id": "dementor-research/oasst-gpt-oss-20b-as-qwen3.6-27b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://3bce597d-f243-589f-8cc1-bd5528200bdd:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://8e9f07fb-a482-5e68-990f-9973ebda493f:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://eec7ba32-68da-51da-aa0f-ba0d346a9622:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-gpt-oss-20b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://84f7e912-7b21-57b2-bba8-342a712231a0:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-dpo-seed42",
    },
    {
        "tinker_path": "tinker://c7d70093-9315-5e0f-96f0-691f23dacb26:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-dpo-seed43",
    },
    {
        "tinker_path": "tinker://1e0dc3c0-afcb-5940-bebd-b6be16789e0b:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-nemotron-nano-dpo-seed44",
    },
    {
        "tinker_path": "tinker://9d7273c8-06f7-568a-8448-b3c284e7d09b:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://32d979e1-db5c-5e12-bed5-c96fb058d22a:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://8b9c961f-9f89-5c92-bc3f-a12d7451608a:train:0/weights/final",
        "repo_id": "dementor-research/oasst-llama-3.1-8b-as-qwen3.6-27b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://5e93ae88-346e-51ff-a84c-15bc4ce141cf:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://59ba9f42-c9cb-5388-8cac-305abfd2829f:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://3c849b32-4f15-5e3b-82b0-779c02905b14:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-gpt-oss-20b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://5023c215-5a66-5d9b-82c1-c7488b5e363a:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://e43de4e2-e53b-5c95-b285-9670c723db6f:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://c6088191-e228-55e7-a62e-1b22bac21e58:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-llama-3.1-8b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://83d8eb51-13c6-5101-83ac-7ead072f2885:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://d5486bf4-011d-5a03-b225-2416379e3304:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://aa3b33fc-0010-5104-8ec6-bd757a02f842:train:0/weights/final",
        "repo_id": "dementor-research/oasst-nemotron-nano-as-qwen3.6-27b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://2914267e-32d5-5a09-ae3f-ba0843a4b83b:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://8caad199-2aa8-567e-81e6-9ef58893f682:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://7865d1eb-17d1-5692-8dd3-f1486fc50410:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-gpt-oss-20b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://2d2e3255-cfda-56fa-a973-4f7b44f6eb50:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-dpo-seed42",
    },
    {
        "tinker_path": "tinker://87d50c2a-604d-5ec5-9b77-3822ca44f907:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-dpo-seed43",
    },
    {
        "tinker_path": "tinker://7a87e9b7-7051-5aaa-aa1f-2248c18e7454:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-llama-3.1-8b-dpo-seed44",
    },
    {
        "tinker_path": "tinker://79548e53-e521-5e40-be3f-8417cde1fb80:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-dpo-seed42",
    },
    {
        "tinker_path": "tinker://2e0052cd-b550-5b76-8beb-bf7098d5f3de:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-dpo-seed43",
    },
    {
        "tinker_path": "tinker://7068a78c-2a05-53b9-b7c5-0f779770b607:train:0/weights/final",
        "repo_id": "dementor-research/oasst-qwen3.6-27b-as-nemotron-nano-dpo-seed44",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true", default=False,
                   help="Make repos private (default: public).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip repos that already exist on HuggingFace (default: True).")
    return p.parse_args()


def push_one(adapter: dict, *, token: str, private: bool) -> str:
    from tinker_cookbook import weights
    from tinker_cookbook.weights import ModelCardConfig
    import tempfile

    tinker_path = adapter["tinker_path"]
    repo_id = adapter["repo_id"]

    model_card = ModelCardConfig(
        base_model=BASE_MODEL,
        datasets=["OpenAssistant/oasst1"],
        tags=["lora", "peft", "dementor", "behavioral-inertia", "openassistant", "oasst1"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        print(f"  [download] {repo_id} from Tinker …")
        local_path = weights.download(tinker_path=tinker_path, output_dir=tmp)
        print(f"  [push]     {repo_id}")
        url = weights.publish_to_hf_hub(
            model_path=local_path,
            repo_id=repo_id,
            private=private,
            token=token,
            model_card=model_card,
        )

    print(f"  [done]     {repo_id}  →  {url}")
    return url


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("Provide --hf-token or set HF_TOKEN.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)

    # Create collection
    print("[1/3] Creating HuggingFace collection …")
    try:
        col = api.create_collection(
            title="Dementor adapters openassistant",
            namespace=HF_ORG,
            description=(
                "Llama-3.1-8B-Instruct LoRA adapters trained on OpenAssistant (oasst1) "
                "to imitate other models — part of the Dementor behavioral-inertia study. "
                "40 SFT adapters + 36 DPO (Direct Preference Optimization) adapters."
            ),
            private=args.private,
            exists_ok=True,
            token=args.hf_token,
        )
        collection_slug = col.slug
        print(f"  Collection: https://huggingface.co/collections/{collection_slug}")
    except Exception as exc:
        print(f"  WARNING: could not create collection: {exc}")
        collection_slug = None

    # Filter existing repos
    from huggingface_hub import HfApi as _HfApi
    _api = _HfApi(token=args.hf_token)
    existing_repos = {m.id for m in _api.list_models(author=HF_ORG)}

    adapters_to_push = []
    for a in ADAPTERS:
        if args.skip_existing and a["repo_id"] in existing_repos:
            print(f"  [skip] {a['repo_id']} (already on HF)")
            continue
        adapters_to_push.append(a)

    if not adapters_to_push:
        print("Nothing to push — all adapters already on HuggingFace.")
        return

    # Push adapters in parallel
    print(f"\n[2/3] Pushing {len(adapters_to_push)} adapters (workers={args.workers}) …")
    pushed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(push_one, a, token=args.hf_token, private=args.private): a
            for a in adapters_to_push
        }
        for fut in as_completed(futures):
            adapter = futures[fut]
            try:
                url = fut.result()
                pushed[adapter["repo_id"]] = url
            except Exception as exc:
                print(f"  ✗ {adapter['repo_id']}: {exc}")

    # Add repos to collection
    if collection_slug and pushed:
        print(f"\n[3/3] Adding {len(pushed)} repos to collection …")
        for repo_id in pushed:
            try:
                api.add_collection_item(
                    collection_slug=collection_slug,
                    item_id=repo_id,
                    item_type="model",
                    token=args.hf_token,
                    exists_ok=True,
                )
                print(f"  ✓ added {repo_id}")
            except Exception as exc:
                print(f"  ✗ {repo_id}: {exc}")

    print("\nDone.")
    print(f"Collection: https://huggingface.co/collections/{collection_slug}")


if __name__ == "__main__":
    main()
