# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 이효준

## **PRT(PeerReviewTemplate)**  
------------------  
- [⭕] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [⭕] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
# 입력으로 받은 text를 self.dict에 각 charater들이 어떤 index에 매핑 되는지 저장
        # character, index 정보를 통해 모델이 학습할 수 있는 output 생성
        # character='ABCD' -> label : {'A' : 1, 'B' : 2,,,}
        self.character = "-" + character
        self.label_map = dict()
        for i, char in enumerate(self.character):
            self.label_map[char] = i
 ```
 > ``label_map[char]``이 어떻게 구성되었는지 한 번에 이해되었습니다.

- [❌] **3. 코드가 에러를 유발할 가능성이 있나요?**
 > 에러 유발할 가능성이 없습니다.

- [⭕] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
K.ctc_batch_cost()  
<img src="https://camo.githubusercontent.com/1ef95f053c3f4d71fb5e2bb84e13f3db4e2f11f63d6c6e66a65e139ddd7dbbce/68747470733a2f2f696d6775722e636f6d2f7244456d504a382e706e67" width="500" height="300"></img>

입력의 길이 T와 라벨의 길이 U의 단위가 일치하지 않을 때,
ex) label : APPLE → output : AAAPPPPLLLLEE

추론 결과가 APPLE이 되게 하려면 이미지의 라벨은 AP-PLE로 보정해 주어야 함
모델이 AAAPP-PPLLLEE로 출력을 한다면 추론 결과는 APPLE이 되는 것
<< 이전 스텝에서 LabelConverter.encode() 메소드에 공백문자 처리로직을 포함한 이유
 > 입출력 길이가 서로 다를때 어떻게 처리가 되어야하는지 잘 이해하고 있습니다.

- [⭕] **5. 코드가 간결한가요?**  
 ```python
def check_inference(model, dataset, index = 5):
    for i in range(index):
        inputs, outputs = dataset[i]
        img = dataset[i][0]['input_image'][0:1,:,:,:]
        output = model.predict(img)
        result = decode_predict_ctc(output, chars="-"+TARGET_CHARACTERS)[0].replace('-','')
        print("Result: \t", result)
        display(Image.fromarray(img[0].transpose(1,0,2).astype(np.uint8)))
 ```
 > 추론 결과와 이미지가 보기 쉽게 정리되어있습니다.

## **참고링크 및 코드 개선 여부**  
------------------  
### 출력 결과 뒤에 99999 숫자가 나오는것을 개선(제거)하는 방법

#### 개선전(뒷 문자 99999 출력)
결과 : SLINKING9999999999999999
```python
def decode_predict_ctc(out, chars = TARGET_CHARACTERS):
    # 생략
    for index in indexes:
        text += chars[index]
    results.append(text)
    return results
```
#### 개선후
결과 : SLINKING
```python
def decode_predict_ctc(out, chars = TARGET_CHARACTERS):
    # 생략
    for index in indexes:
        if index != -1: # 개선 부분
            text += chars[index]
    results.append(text)
    return results
```
