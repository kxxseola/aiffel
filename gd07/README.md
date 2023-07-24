# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 이동익

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
def put_sticker(img, boxes, box_index):
    img_height = img.shape[0]
    img_width = img.shape[1]
    num = 40

    x_min = int(boxes[box_index][0] * img_width)
    y_min = int(boxes[box_index][1] * img_height)
    x_max = int(boxes[box_index][2] * img_width)
    y_max = int(boxes[box_index][3] * img_height)
    face = img[y_min-num:y_max+num, x_min-num:x_max+num]

    # landmark 찾기
    detector_hog = dlib.get_frontal_face_detector()
    dlib_rect = detector_hog(face, 1)
    landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
    landmarks = []
    
    for rect in dlib_rect:
        points = landmark_predictor(img, rect)
        print(points)
        list_points = list(map(lambda p:(p.x, p.y), points.parts()))
        landmarks.append(list_points)
        
    img_sticker = cv2.imread(STICKER_PATH)

    # face landmark 인식 불가면 pass
    if len(landmarks)==0:
        pass
    
    else:
        landmark = landmarks[0]
        # 스티커 좌표 설정   
        w = h = landmark[8][1] - landmark[19][1]
        x = (landmark[30][0] - (num//2))
        y = (landmark[30][1] - (num//3))
        image_sticker = cv2.resize(img_sticker, (w,h))
 
        # 스티커 위치 설정
        sticker_area = face[y:y+image_sticker.shape[0], x:x+image_sticker.shape[1]]

        # 스티커 중 검은 수염 부분을 이미지의 해당 위치에 대체
        face[y:y+image_sticker.shape[0], x:x+image_sticker.shape[1]] = \
            np.where(image_sticker==255, sticker_area, image_sticker).astype(np.uint8)


 ```
 > 주석을 잘달아주셔서 이해가 잘 되었습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**

 > 없는 것 같습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
# bounding box info : [x, y, w, h] -> [x_min, y_min, x_max, y_max] 형태의 꼭짓점 좌표 정보로 변환
# dataset의 image file, bounding box info -> example을 serialize하여 TFRecord 파일로 생성
# default box를 생성하기 위해 기준이 되는 feature map을 생성   
# 4가지 유형의 feature map을 생성
# feature map 별로 default box 생성
 ```
 > 과정 별로 주석을 잘달아주셔서 프로세스가 잘 이해되었습니다. 고찰 부분에서도 과정을 잘 이해하고 계신 것 같습니다.

- [x] **5. 코드가 간결한가요?**
```python
model.load_weights(FILEPATH)

img_raw = cv2.imread(TEST_IMAGE_PATH)
img_raw = cv2.resize(img_raw, (IMAGE_WIDTH, IMAGE_HEIGHT))
img = np.float32(img_raw.copy())

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img, pad_params = pad_input_image(img, max_steps=max(BOX_STEPS))
img = img / 255.0

boxes = default_box()
boxes = tf.cast(boxes, tf.float32)

predictions = model.predict(img[np.newaxis, ...])

pred_boxes, labels, scores = parse_predict(predictions, boxes)
pred_boxes = recover_pad(pred_boxes, pad_params)

for box_index in range(len(pred_boxes)):
    put_sticker(img_raw, pred_boxes, box_index)
plt.figure(figsize=(15, 15))
plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
plt.show()
```

 > 간결합니다!!

## **참고링크 및 코드 개선 여부**  
------------------  
- 
