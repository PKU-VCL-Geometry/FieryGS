import numpy as np
import os
import json
import random
import base64
from openai import OpenAI

random.seed(123)

# ---------------------------------------------------------------------------
# API Configuration via environment variables:
#   QWEN_API_KEY       - API key for Qwen/DashScope
#   OPENAI_API_KEY     - API key for OpenAI (or compatible proxy)
#   QWEN_BASE_URL      - (optional) override Qwen endpoint
#   OPENAI_BASE_URL    - (optional) override OpenAI endpoint
#
# Proxy alternatives (for testing behind GFW):
#   OPENAI_BASE_URL=https://hk.xty.app/v1
#   OPENAI_BASE_URL=https://aihubmix.com/v1
# ---------------------------------------------------------------------------

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _get_qwen_client():
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise EnvironmentError("QWEN_API_KEY environment variable is not set. See .env.example for details.")
    return OpenAI(base_url=QWEN_BASE_URL, api_key=api_key)


def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set. See .env.example for details.")
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)


def Qwen(image_path, prompt):
    base64_image = _encode_image(image_path)
    client = _get_qwen_client()
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def GPT4V(image_path, prompt):
    client = _get_openai_client()
    base64_image = _encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a vision assistant. Think step-by-step internally but output only the final answer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def GPTo3(image_path, prompt):
    client = _get_openai_client()
    base64_image = _encode_image(image_path)

    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {
                "role": "system",
                "content": "You are a vision assistant. Think step-by-step internally but output only the final answer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


MATERIALS = [
    "wood", "sand", "metal", "plastic", "glass", "fabric", "foam", "food",
    "ceramic", "paper", "leather", "plant", "stone", "cement",
    "concrete", "soil", "clay", "composite", "sky"
]


def GPTtools(image_path, prompt):
    tools = [{
        "type": "function",
        "function": {
            "name": "report_material",
            "description": "Return caption, material and burnable flag for the masked part",
            "parameters": {
                "type": "object",
                "properties": {
                    "caption":  {"type": "string"},
                    "material": {"type": "string", "enum": MATERIALS},
                    "burnable": {"type": "string", "enum": ["burnable", "unburnable"]}
                },
                "required": ["caption", "material", "burnable"]
            }
        }
    }]

    client = _get_openai_client()
    base64_image = _encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a vision assistant. Think step-by-step internally but output only the final answer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "report_material"}},
    )
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result_str = f"{args['caption']}, {args['material']}, {args['burnable']}"
    return result_str


def GPTtools_supp(image_path, prompt):
    tools = [{
        "type": "function",
        "function": {
            "name": "report_material",
            "description": "Return caption, material, burnable flag, thermal diffusivity ratio (vs. wood), and smoke color for the masked part",
            "parameters": {
                "type": "object",
                "properties": {
                    "caption": {
                        "type": "string",
                        "description": "A brief description of the segmented part"
                    },
                    "material": {
                        "type": "string",
                        "enum": MATERIALS,
                        "description": "Main material of the part, must be from the predefined material library"
                    },
                    "burnable": {
                        "type": "string",
                        "enum": ["burnable", "unburnable"],
                        "description": "Whether the material is considered burnable under normal conditions"
                    },
                    "thermal_diffusivity_ratio_vs_wood": {
                        "type": "number",
                        "description": "Thermal diffusivity of the material divided by that of wood (unitless ratio)"
                    },
                    "smoke_color": {
                        "type": "string",
                        "description": "Typical color of smoke when this material burns, e.g., black, white, gray"
                    }
                },
                "required": ["caption", "material", "burnable", "thermal_diffusivity_ratio_vs_wood", "smoke_color"]
            }
        }
    }]

    client = _get_openai_client()
    base64_image = _encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a vision assistant. Think step-by-step internally but output only the final answer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "report_material"}},
    )
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result_str = f"{args['caption']}, {args['material']}, {args['burnable']}, {args['thermal_diffusivity_ratio_vs_wood']}, {args['smoke_color']}"
    return result_str


def get_image_files(directory):
    image_files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.png')]
    return image_files

