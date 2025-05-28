import os
import json
import base64
import zipfile
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import requests
from PIL import Image as PILImage
from IPython.display import Image, display
import matplotlib.pyplot as plt

from openai import OpenAI

OPENAI_API_KEY = "api_key"

client = OpenAI(api_key=OPENAI_API_KEY)

def get_response(client, model, r_format, messages):
    completion = client.beta.chat.completions.parse(
        model=model,
        temperature=0.1,
        stop=['</json>',],
        response_format=r_format,
        messages=messages
    )
    return completion

def get_contents(response):
    return response.choices[0].message.content

def show_img(img_path, width=None, height=None):
    img = PILImage.open(img_path)
    display(img)
    return img

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


example_paths = [
    r"Enter the img1_paths",
    r"Enter the img2_paths",
    r"Enter the img3_paths",
    r"Enter the img4_paths",
    r"Enter the img5_paths",
]

example_queries = [
    "Enter the guery1",
    "Enter the guery2",
    "Enter the guery3",
    "Enter the guery4",
    "Enter the guery5",
]

example_images = []
for img_path in example_paths:
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    example_images.append(f"data:image/png;base64,{b64}")

examples = list(zip(example_images, example_queries))

system_content = """\
[System Prompt]
You are a chart question-answering expert.  
I will provide you with a query and a chart image; based on these, output according to the following format and do not output anything else.  
Only fill the OCR Result and QA Result fields.  
Do not add any extra text, headings, examples, explanations, formatting, or units. 

[Output Examples Prompt]
It is an example of OCR Result and QA Result output. Based on the Image and Query provided in the actual prompt, output the OCR Result and QA Result in the same format.
  - Example 1
  Image: {example_img1}
  Query: {q_img1}
  OCR Result : [example OCR Result 1]
  QA Result : [example  QA Result 2]

  - Example 2
  Image: {example_img2}
  Query: {q_img2}
  OCR Result : [example OCR Result 2]
  QA Result : [example  QA Result 2]

  - Example 3
  Image: {example_img3}
  Query: {q_img3}
  OCR Result : [example OCR Result 3]
  QA Result : [example  QA Result 3]

  - Example 4
  Image: {example_img4}
  Query: {q_img4}
  OCR Result : [example OCR Result 4]
  QA Result : [example  QA Result 4]

  QA Result : Lombardy

  - Example 5
  Image: {example_img5}
  Query: {q_img5}
  OCR Result : [example OCR Result 5]
  QA Result : [example  QA Result 5]
"""

dataset_root   = r"Enter the root dataset path"
json_train     = os.path.join(dataset_root, "train", "train_human.json")
json_val       = os.path.join(dataset_root, "val",   "val_human.json")
json_test      = os.path.join(dataset_root, "test",  "test_human.json")

with open(json_train, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(json_val, 'r', encoding='utf-8') as f:
    val_data   = json.load(f)
with open(json_test, 'r', encoding='utf-8') as f:
    test_data  = json.load(f)

train_df = pd.DataFrame(train_data, columns=['imgname', 'query', 'label'])
val_df   = pd.DataFrame(val_data,   columns=['imgname', 'query', 'label'])
test_df  = pd.DataFrame(test_data,  columns=['imgname', 'query', 'label'])

for df, split in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
    img_dir = os.path.join(dataset_root, split, "png")
    df['imgname'] = df['imgname'].apply(lambda fn: os.path.join(img_dir, fn))
    tbl_dir = os.path.join(dataset_root, split, "tables")
    df['tabname'] = df['imgname'] \
        .str.replace(os.path.join(img_dir), tbl_dir, regex=False) \
        .str.replace(r'\.png$', '.csv', regex=True)

train_df = train_df[['imgname', 'tabname', 'query', 'label']]
val_df   = val_df[['imgname', 'tabname', 'query', 'label']]
test_df  = test_df[['imgname', 'tabname', 'query', 'label']]

predictions = []

for _, sample in test_df.iterrows():
    img_path    = sample["imgname"]
    query_text  = sample["query"]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    image_data_url = f"data:image/png;base64,{b64}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text",      "text": f"""\
Image: <start_of_image>
Query: {query_text}
OCR Result:
QA Result:
    """}
        ]},
    ]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
    )

    pred = response.choices[0].message.content.strip()

    predictions.append({
        "imgname":    img_path,
        "query":      query_text,
        "prediction": pred,
        "label":      sample.get("label", None)
    })

results_df = pd.DataFrame(predictions)
output_path = "Enter the output_path"
results_df.to_csv(output_path, index=False, encoding="utf-8-sig")    
