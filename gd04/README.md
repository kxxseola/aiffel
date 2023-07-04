# AIFFEL GoingDeeper
----
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 김창완

## **PRT(PeerReviewTemplate)**  
------------------
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  - 네 정확히 3가지의 모든 과제를 해결 했습니다. 또한 나아가 각  상황에서의 예측은 어떻게 될지 적어 주셨습니다
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  - 네 특히 데이터 파이프라인 구축때 세세하게 달려있어서 보기 편했습니다.
 ```python
 # x, y 좌표 위치 교체
 def swap_xy(boxes):
     return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
 
 # 무작위로 수평 뒤집기
 def random_flip_horizontal(image, boxes):
     if tf.random.uniform(()) > 0.5:
         image = tf.image.flip_left_right(image)
         boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
         )
         
     return image, boxes
 
 # 이미지 크기 조정 및 패딩 추가
 # 이미지의 비율 << 그대로 유지
 # 이미지 최대/최소 크기 제한
 def resize_and_pad_image(image, training=True):
     
     min_side = 800.0
     max_side = 1333.0
     min_side_range = [640, 1024]
     stride = 128.0
     
     image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
     if training:
         min_side = tf.random.uniform((), min_side_range[0], min_side_range[1], dtype=tf.float32)
     # ratio == min side를 image_shape 중 가장 작은 값으로 나눈 값
     ratio = min_side / tf.reduce_min(image_shape)   # tf.reduce_min : 가장 작은 값
     # (ratio * image_shape 중 가장 큰 값)이 max_side 보다 크다면
     if ratio * tf.reduce_max(image_shape) > max_side: 
         # ratio == max side를 image_shape 중 가장 큰 값으로 나눈 값  
         ratio = max_side / tf.reduce_max(image_shape)
     image_shape = ratio * image_shape
     image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
     # 최종적으로 모델에 입력되는 이미지의 크기 : stride의 배수
     # 모델에 입력되는 이미지 : 검정 테두리가 있음
     padded_image_shape = tf.cast(
         tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
     )
     image = tf.image.pad_to_bounding_box(
         image, 0, 0, padded_image_shape[0], padded_image_shape[1]
     )
     return image, image_shape, ratio
 
 # [x_min, y_min, x_max, y_max] -> [x_min, y_min, width, height] 로 수정
 def convert_to_xywh(boxes):
     return tf.concat(
         [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
         axis=-1,
     )
 ```
 >

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
  - 아니오, Warning도 없어서 편하게 가능할것 같습니다
 ```python
 
 ```
 >

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  - 네. 데이터 파이프 라인. 모델의 가중치를 어떻게 가져왔는지 등을 질문했는데 잘 대답해 주셨습니다.
 ```python
 # 학습된 모델 불러오기
 model_dir = '/Users/yena/Desktop/python_study/AIFFEL/GOINGDEEPER/checkpoints/'
 latest_checkpoint = tf.train.latest_checkpoint(model_dir)
 model.load_weights(latest_checkpoint)
 ```
 >

- [x] **5. 코드가 간결한가요?**  
  - 네. 지금 제 단계에서는 더 줄일만한 개선점은 찾지 못하였습니다. 
 ```python
 
 ```
 >

## **참고링크 및 코드 개선 여부**  
------------------
- 이번것은 저도 배울게 많아 코드의 개선 여부보다는 배울게 더 많았던것 같습니다.
- 특히 마지막의 Threshold 비교는 상세하게 되어 있어 보기도 편하고 결과값이 무엇을 도출하고 의미하는지 알기 편했습니다.
