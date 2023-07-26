# AIFFEL GoingDeeper
### [Peer Review 이후 파일 수정 완료]
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 이성주

## **PRT(PeerReviewTemplate)**  
------------------  
- [o] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

|평가문항|상세기준|완료여부|
|-------|---------|--------|
|1. tfrecord를 활용한 데이터셋 구성과 전처리를 통해 프로젝트 베이스라인 구성을 확인하였다.|MPII 데이터셋을 기반으로 1epoch에 30분 이내에 학습가능한 베이스라인을 구축하였다.|![image](https://github.com/kxxseola/aiffel/assets/29011595/6d6e8ed4-4f1a-4010-af58-9e2b91189952) 에폭당 학습시간이 10분 정도라고 합니다.|
|2. simplebaseline 모델을 정상적으로 구현하였다.|simplebaseline 모델을 구현하여 실습코드의 모델을 대체하여 정상적으로 학습이 진행되었다.|![image](https://github.com/kxxseola/aiffel/assets/29011595/2bd54b28-4b55-4272-906c-71a865a0b329) 학습이 잘 진행되었습니다.|
|Hourglass 모델과 simplebaseline 모델을 비교분석한 결과를 체계적으로 정리하였다.|두 모델의 pose estimation 테스트결과 이미지 및 학습진행상황 등을 체계적으로 비교분석하였다.| 진행중 입니다.|

- [o] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
# 얼마나 많은 TFRecord를 만들지 결정할 함수
# 전체 데이터를 몇 개의 그룹으로 나눌지 결정
def chunkify(l, n):
    size = len(l) // n
    start = 0
    results = []
    for i in range(n):
        results.append(l[start:start + size])
        start += size
    return results
 ```
 > 네 주석을 보고 이해가 되었습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 ```python
#모델과 이미지 경로를 입력하면 이미지와 keypoint를 출력하는 함수
def predict(model, image_path):
    encoded = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(encoded)
    inputs = tf.image.resize(image, (256, 256))
    inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)
    outputs = model(inputs, training=False)
    if type(outputs) != list:
        outputs = [outputs]
    heatmap = tf.squeeze(outputs[-1], axis=0).numpy()
    kp = extract_keypoints_from_heatmap(heatmap)
    return image, kp
def draw_keypoints_on_image(image, keypoints, index=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    joints = []
    for i, joint in enumerate(keypoints):
        joint_x = joint[0] * image.shape[1]
        joint_y = joint[1] * image.shape[0]
        if index is not None and index != i:
            continue
        plt.scatter(joint_x, joint_y, s=10, c='red', marker='o')
    plt.show()

def draw_skeleton_on_image(image, keypoints, index=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    joints = []
    for i, joint in enumerate(keypoints):
        joint_x = joint[0] * image.shape[1]
        joint_y = joint[1] * image.shape[0]
        joints.append((joint_x, joint_y))
    
    for bone in MPII_BONES:
        joint_1 = joints[bone[0]]
        joint_2 = joints[bone[1]]
        plt.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], linewidth=5, alpha=0.7)
    plt.show()

test_image = os.path.join(PROJECT_PATH, 'test_image.jpg')

image, keypoints = predict(hourglass_model, test_image)
draw_keypoints_on_image(image, keypoints)
draw_skeleton_on_image(image, keypoints)
 ```
![image](https://github.com/kxxseola/aiffel/assets/29011595/a07b6052-1b86-4c30-b792-329346ba3f6f)

 > 에러를 고치고있어요

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 > ㅎㅎ;; 

- [o] **5. 코드가 간결한가요?**  
  
 ```python
implebaseline_model_file = train(epochs, learning_rate, num_heatmap, 
                                  batch_size, train_tfrecords, val_tfrecords, is_baseline=True)
 ```
 > is_baseline으로 함수를 재사용하여 코드를 간결하게 작성하였습니다.

## **참고링크 및 코드 개선 여부**  
------------------  
- 
