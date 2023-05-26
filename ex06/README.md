# AIFFEL Exploration
----  
## **Code Peer Review Templete**
------------------
- 코더 : 김설아
- 리뷰어 : 사재원

## **PRT(PeerReviewTemplate)**  
------------------  
- [&#x2B55;] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  네, 주어진 프로젝트의 요구사항들은 전반적으로 충족시켰다고 봅니다. 
- [&#x2B55;] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
```python
# encoder_input, decoder_input, decoder_target -> np.array
encoder_input = np.array(df['text']) # 인코더의 입력
decoder_input = np.array(df['decoder_input']) # 디코더의 입력
decoder_target = np.array(df['decoder_target']) # 디코더의 레이블
```

```python
# 데이터 전처리 함수
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (headlines)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```

  네, nlp에 대한 지식이 없더라도 주석을 보고 각각 어떤 역할을 수행하고있는지, 어떻게 데이터를 가공하고있는지 충분히 이해할 만한 코드입니다

- [&#x274C;] **3. 코드가 에러를 유발할 가능성이 있나요?**
  없는 것 같습니다. 


- [&#x2B55;] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  네, 데이터 전처리 과정이라든지 학습 모델 설계 흐름을 잘 파악하고 있는 것 같습니다.



- [&#x2B55;] **5. 코드가 간결한가요?**  
  네, 주석이나 적절한 들여쓰기로 인해 읽기 편한 코드였습니다.
  ```python
  # empty sample 확인 및 제거
drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]

print('삭제할 훈련 데이터의 개수 :', len(drop_train))
print('삭제할 테스트 데이터의 개수 :', len(drop_test))

encoder_input_train = [sentence for index, sentence in enumerate(encoder_input_train) if index not in drop_train]
decoder_input_train = [sentence for index, sentence in enumerate(decoder_input_train) if index not in drop_train]
decoder_target_train = [sentence for index, sentence in enumerate(decoder_target_train) if index not in drop_train]

encoder_input_test = [sentence for index, sentence in enumerate(encoder_input_test) if index not in drop_test]
decoder_input_test = [sentence for index, sentence in enumerate(decoder_input_test) if index not in drop_test]
decoder_target_test = [sentence for index, sentence in enumerate(decoder_target_test) if index not in drop_test]

print('훈련 데이터의 개수 :', len(encoder_input_train))
print('훈련 레이블의 개수 :', len(decoder_input_train))
print('테스트 데이터의 개수 :', len(encoder_input_test))
print('테스트 레이블의 개수 :', len(decoder_input_test))
  ```



## **참고링크 및 코드 개선 여부**  
------------------  
- 결과 출력은 일부만 가져오게 해주시면 좋을 것 같습니다 혹은 수치로 변환하는 방식도 괜찮고요
    
