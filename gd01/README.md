# AIFFEL Exploration
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
        for cnn_num in range(num_cnn):  # [3, 4, 6, 3]
            
            identity = x
            x = keras.layers.Conv2D(filters=channel,kernel_size=(1,1),strides=1,padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)       
                 
            x = keras.layers.Conv2D(filters=channel,kernel_size=(3,3),strides=1,padding='same')(x)  
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)       
            
            x = keras.layers.Conv2D(filters=4*channel,kernel_size=(1,1),strides=1,padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
        
            if is_plane==False:
                identity_channel = identity.shape.as_list()[-1]
                
                if identity_channel != channel:
                    identity = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, padding='same')(identity)
                    identity = keras.layers.BatchNormalization()(identity)
                    # skip connections
                    x = keras.layers.Add()([x, identity])
            else:
                pass    
                      
            x = keras.layers.Activation('relu')(x) 
 ```
 > block layer를 한 줄 한 줄 잘 구분하여 작성해주어 읽기에 많은 도움이 되었습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 > 에러를 허용하지 않는 완벽..

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
assert len(num_cnn_list) == len(channel_list) #모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.
 ```
 > 디버깅을 위한 `assert`구문이 인상적이었습니다.  
 > 동작 순서를 잘 이해하고 있음을 확인할 수 있는 좋은 부분이었습니다.

- [x] **5. 코드가 간결한가요?**  
  ### 데이터셋 불러오기
 ```python
dataset=tf.keras.preprocessing.image_dataset_from_directory('/Users/yena/Desktop/python_study/AIFFEL/GOINGDEEPER/dogs-vs-cats/train',
                                                            shuffle=True,
                                                            image_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                            batch_size=BATCH_SIZE)
 ```
 ### 데이터 증강
 ```python
resize_and_rescale=tf.keras.Sequential([
    keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    keras.layers.experimental.preprocessing.Rescaling(1.0/255)])

data_augmentation = tf.keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomRotation(0.2)])
 ```
 > Dataset 불러오고 증강 및 전처리 내용이 깔끔하게 잘 정리되어 있었습니다. 많이 배워갑니다.  
 > `data_augmentation` 

## **참고링크 및 코드 개선 여부**  
------------------  
- 
    
