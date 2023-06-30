# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 사재원

## **PRT(PeerReviewTemplate)**  
------------------  
- [&#11093;] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- 네 프로젝트의 조건을 충족시키고 주어진 문제를 잘 해결하였습니다.
- [&#11093;] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
-    모델 설계 전 이미지와 간단한 구조를 설명하고 이후 모델 파이프 라인 설계를 통하여 쉽게 이해가 가능 했습니다.
 ```
ex)
CNN → 특성맵 → GAP → softmax layer(softmax를 가지고 bias가 없는 fully connected layer)를 적용
 ```
 >

- [❌] **3. 코드가 에러를 유발할 가능성이 있나요?**
  에러 유발 가능성은 없어 보입니다. 그렇지만 
    grad_cam_image = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2]) 이부분이 살짝 헷갈립니다. 해당 shape 크기에서 배치사이즈까지 구하는 이유가 있을까요??
 ```python
def generate_grad_cam(model, activation_layer, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # 특정 레이어의 output이 필요함 -> 모델의 input과 output을 새롭게 정의
    # 어떤 레이어든 CAM 이미지를 뽑을 수 있음.
    # 관찰 대상 레이어의 이름으로 activation_layer를 찾은 후 output으로 추가
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(activation_layer).output, model.output])
    
    # Gradient를 얻기 위해 tape를 사용
    # : 원하는 레이어의 output과 특정 클래스의 prediction 사이의 그래디언트 grad_val을 얻고 이를 weights로 활용하기 때문
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(tf.expand_dims(img_tensor, 0))
    
        loss = pred[:, class_idx] # 원하는 class(정답으로 활용) 예측값
        output = conv_output[0] # 원하는 layer의 output
        grad_val = tape.gradient(loss, conv_output)[0] # pred와 output 사이의 gradient << 가중치로 활용

    weights = np.mean(grad_val, axis=(0, 1)) # gradient의 GAP으로 weight를 구함
    grad_cam_image = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        # output의 k번째 채널과 k번째 weight를 곱하고 누적 : class activation map
        grad_cam_image += w * output[:, :, k]
        
    grad_cam_image = tf.math.maximum(0, grad_cam_image)
    grad_cam_image /= np.max(grad_cam_image)
    grad_cam_image = grad_cam_image.numpy()
    grad_cam_image = cv2.resize(grad_cam_image, (width, height))
    return grad_cam_image
 ```
 >

- [&#11093;] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  네 모델 구조와 각각의 시각화 방법 그리고  iou 계산 방식 까지 이해하고 적절하게 작성하였습니다.
 ```python
def get_iou(boxA, boxB):
    y_min = max(boxA[0], boxB[0])
    x_min= max(boxA[1], boxB[1])
    y_max = min(boxA[2], boxB[2])
    x_max = min(boxA[3], boxB[3])
    
    interArea = max(0, x_max - x_min) * max(0, y_max - y_min)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = iou.round(4)
    
    return iou
 ```
 >

- [&#11093;] **5. 코드가 간결한가요?**  
  네 굉장히 깔끔하고 블럭단위로 잘 나누어 주셨습니다.
 ```python
 ```
 >
