# AIFFEL Exploration 08
----  
## **Code Peer Review 07**
------------------
- 코더 : 김설아
- 리뷰어 : 이동익

## **PRT(PeerReviewTemplate)**  
------------------  
- [⭕] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
 >트랜스포머 모델을 통한 챗봇을 구현하고 주어진 문장에 대한 한글 출력을 얻었습니다.
- [⭕] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 >하이퍼파라미터에 대한 주석과 코드의 라인별 주석을 달아 잘 이해되었습니다.
 ```python
 # 하이퍼파라미터
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
DROPOUT = 0.1 # 드롭아웃의 비율
 ```
 ```python
 # 전처리 동일하게 
def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)

    # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
    sentence = tf.expand_dims(
    START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
    # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
    output_sequence = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 인퍼런스 단계
    for i in range(MAX_LENGTH):
        # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

        # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 예측한 단어들은 지속적으로 output_sequence에 추가
        # output_sequence는 다시 디코더의 입력이 됨
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)
 ```

- [❌] **3. 코드가 에러를 유발할 가능성이 있나요?**
 > 없는 것 같습니다. 다만, 학습횟수를 늘려서 accuracy를 올려보면 출력 결과가 더 잘 나올 것 같습니다.

- [⭕] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
 >제가 lookahead making에 대해 이해를 잘 못한 부분을 잘 설명해주셨습니다.   
 >또한 해당 논문에서 표현하는 수식과 내용을 통해 코드가 어떤식으로 구현되는지 이해하고 계셨습니다.

- [⭕] **5. 코드가 간결한가요?**  
 >대체적으로 간결합니다.

## **참고링크 및 코드 개선 여부**  
------------------  
- 
    
