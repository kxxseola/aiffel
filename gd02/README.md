# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 이효준

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
# imagenet으로 훈련된 resnet50
resnet50 = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,  # imagenet과 데이터셋이 서로 다른 클래스를 가짐 
                            # : 마지막에 추가해야하는 fully connected layer의 구조(뉴련의 개수)가 다름
        weights='imagenet',
        input_shape=(224,224,3),
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation='softmax')
])
 ```
 > `include_top=False` 내용에 주석을 추가로 달아두어 모델 설계에 대한 이해가 훨씬 더 잘 되었습니다.

- [] **3. 코드가 에러를 유발할 가능성이 있나요?**
 > 에러를 유발하는 부분은 없었습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
train = apply_normalize_on_dataset(ds_train)                                            # No Augmentation
aug_train = apply_normalize_on_dataset(ds_train, with_aug=True)                         # 기본 Augmentation
cutmix_train = apply_normalize_on_dataset(ds_train, with_aug=True, with_cutmix=True)    # 기본 + cutmix
mixup_train = apply_normalize_on_dataset(ds_train, with_aug=True, with_mixup=True)      # 기본 + mixup

test = apply_normalize_on_dataset(ds_test, is_test=True)
 ```
 > 검증에 활용할 `test`데이터는 증강처리 등을 하지 않아 각 데이터셋간 모델의 학습 수렴에 대한 옳은 비교가 가능하였다.

- [x] **5. 코드가 간결한가요?**  
  
 ```python
plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
plt.plot(history_resnet50_aug.history['val_accuracy'], 'g')
plt.plot(history_resnet50_cutmix.history['val_accuracy'], 'b')
plt.plot(history_resnet50_mixup.history['val_accuracy'], 'k')

plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Augmentation', 'With Augmentation', 'Aug+Cutmix', 'Aug+Mixup'], loc='upper left')
plt.show()
 ```
 > Validation 결과 시각화를 위한 군더더기 없는 간결한 코드가 매우 인상깊었습니다.

## **참고링크 및 코드 개선 여부**  
------------------  
- 
    
