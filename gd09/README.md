# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 사재원

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

드림부스를 학습하고 개라는 텍스트를 잘 이해시켰고 lora 모델을 통하여 강아지 이미지를 생성시켰다.

- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
네 이미 가이드 코드 내에 전체적인 흐름 단계별 설명이 작성되어있어 이해하기 쉬웠습니다.
 ```python
# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "digiplay/hellofantasytime_v1.22"

unet = UNet2DConditionModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

 ```
 >

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
아니요 문제없이 잘 돌아갔습니다
 ```python

 ```
 >

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  네 학습을 왜 하는지 학습 이후에 어떻게 추론을 하고 부가적인 모델의 가중치를 가져오는 것 까지 잘 이해한 것 같습니다.
 ```python
!wget https://civitai.com/api/download/models/116417 -O lora_example.safetensors
 ```

```

from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "digiplay/hellofantasytime_v1.22"

unet = UNet2DConditionModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

```
 >

- [x] **5. 코드가 간결한가요?**  
  네 코드 자체는 이미 만들어진 bash 파일이나 py파일을 이용하기에 어떤 역할을하는지 한눈에 알아보고 괜찮았습니다.
 ```python
!sh train_dreambooth.sh
 ```
 >

## **참고링크 및 코드 개선 여부**  
------------------  
- 
    
