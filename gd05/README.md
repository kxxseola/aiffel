# AIFFEL GoingDeeper
----
## **Code Peer Review**
------------------
- 코더 : 김설아
- 리뷰어 : 김창완

## **PRT(PeerReviewTemplate)**  
------------------
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  - 네. 세그멘테이션 작업이 정상적으로 진행 되었고, U-Net++ 모델이 성공적으로 구현되었으며 정량/ 정성적 평가가 제대로 이루어져 있었습니다.
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
  - 네, 특히 KittiGenerator에서 상세한 주석으로 어떻게 subclassing을 구현했는지 정확히 알았습니다
 ```python
 # tf.keras.utils.Sequence 상속
 # Sequence를 커스텀해서 사용
 class KittiGenerator(tf.keras.utils.Sequence):
 
     def __init__(self, dir_path, batch_size=8, img_size=(224, 224, 3), 
                  output_size=(224, 224), is_train=True, augmentation=None):
         self.dir_path = dir_path 
         self.batch_size = batch_size
         self.is_train = is_train
         self.augmentation = augmentation    # 적용할 augmentation 함수를 인자로 받음
         self.img_size = img_size            # preprocess에 사용할 input image size
         self.output_size = output_size      # ground_truth를 만들기 위한 size
 
         # load_dataset()을 통해서 kitti dataset의 directory path에서 라벨과 이미지를 확인
         self.data = self.load_dataset()
 
     def load_dataset(self):
     # kitti dataset에서 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드
     # test set을 분리해서 load
         input_images = glob(os.path.join(self.dir_path, "image_2", "*.png"))
         label_images = glob(os.path.join(self.dir_path, "semantic", "*.png"))
         input_images.sort()
         label_images.sort()
         assert len(input_images) == len(label_images)
         data = [ _ for _ in zip(input_images, label_images)]
 
         if self.is_train:
             return data[:-30]
         return data[-30:]
     
     def __len__(self):
         # Generator의 length
         # 전체 dataset을 batch_size로 나누고 소숫점 첫째자리에서 올림한 값을 반환
         return math.ceil(len(self.data) / self.batch_size)
 
     def __getitem__(self, index):
         # input, output 만듦
         # input : resize및 augmentation이 적용된 input image
         # output : semantic label
         batch_data = self.data[
                             index*self.batch_size:
                             (index + 1)*self.batch_size
                             ]
         inputs = np.zeros([self.batch_size, *self.img_size])
         outputs = np.zeros([self.batch_size, *self.output_size])
             
         for i, data in enumerate(batch_data):
             input_img_path, output_path = data
             _input = imread(input_img_path)
             _output = imread(output_path)
             _output = (_output==7).astype(np.uint8)*1
             data = {"image": _input,"mask": _output,}
             augmented = self.augmentation(**data)
             inputs[i] = augmented["image"]/255
             outputs[i] = augmented["mask"]
             return inputs, outputs
 
     def on_epoch_end(self):
         # 한 epoch가 끝나면 실행되는 함수
         # 학습중인 경우에 순서를 random shuffle하도록 적용
         self.indexes = np.arange(len(self.data))
         if self.is_train == True :
             np.random.shuffle(self.indexes)
             return self.indexes
 ```
 >

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
  - 없습니다. 변수명도 깔끔하게 짜여 있었습니다.
 ```python
 ```
 >

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  

  - 네, 서로 토론하며 전부 명확히 이해하고 있는것을 확인했습니다.  

    특히 논문에서 말한대로 구현하신게 인상깊었습니다.
 ```python
 # backbone 없이
 def u_net_plus_plus(input_shape=(224, 224, 3)):
 
     num_conv = [32, 64, 128, 256, 512]
     input_layer = keras.Input(shape=input_shape)
     conv0_0, pool0_0 = conv_2_pool(input_layer, 32)
     conv1_0, pool1_0 = conv_2_pool(pool0_0, 64)
     conv2_0, pool2_0 = conv_2_pool(pool1_0, 128)
     conv3_0, pool3_0 = conv_2_pool(pool2_0, 128)
     conv4_0 = conv_2(pool3_0, 128)
     
     # 규칙
     # up = keras.layers.Conv2DTranspose(num_conv, 2, activation='relu', strides=(2,2), kernel_initializer='he_normal')(conv(i+1)_(j-1))
     # merge = keras.layers.concatenate([conv(i)_(j-1),up], axis = 3)
     # conv(i)_(j) = conv_2(merge, num_conv)
 
     # 3 (1번)
     up3_1 = keras.layers.Conv2DTranspose(256, 2, activation='relu', strides=(2,2), kernel_initializer='he_normal')(conv4_0) 
     merge3_1 = keras.layers.concatenate([conv3_0,up3_1], axis = 3)
     conv3_1 = conv_2(merge3_1, 256)
     
     # 2 (2번)
     up2_1 = keras.layers.Conv2DTranspose(128, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv3_0)
     merge2_1 = keras.layers.concatenate([conv2_0, up2_1], axis = 3)
     conv2_1 = conv_2(merge2_1, 128)
     
     up2_2 = keras.layers.Conv2DTranspose(128, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv3_1)
     merge2_2 = keras.layers.concatenate([conv2_0, conv2_1, up2_2], axis = 3)
     conv2_2 = conv_2(merge2_2, 128)
     
     # 1 (3번)
     up1_1 = keras.layers.Conv2DTranspose(64, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv2_0)
     merge1_1 = keras.layers.concatenate([conv1_0, up1_1], axis = 3)
     conv1_1, pool1_1 = conv_2_pool(merge1_1, 64)
     
     up1_2 = keras.layers.Conv2DTranspose(64, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv2_1)
     merge1_2 = keras.layers.concatenate([conv1_0, conv1_1, up1_2], axis = 3)
     conv1_2, pool1_2 = conv_2_pool(merge1_2, 64)
     
     up1_3 = keras.layers.Conv2DTranspose(64, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv2_2)
     merge1_3 = keras.layers.concatenate([conv1_0, conv1_1, conv1_2, up1_3], axis = 3)
     conv1_3, pool1_3 = conv_2_pool(merge1_3, 64)
     
     # 0 (4번)
     up0_1 = keras.layers.Conv2DTranspose(32, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv1_0) 
     merge0_1 = keras.layers.concatenate([conv0_0, up0_1], axis = 3)
     conv0_1 = conv_2(merge0_1, 32)
     
     up0_2 = keras.layers.Conv2DTranspose(32, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv1_1) # (56, 56, 32) << (pool1_1)
     merge0_2 = keras.layers.concatenate([conv0_0, conv0_1, up0_2], axis=3)
     conv0_2 = conv_2(merge0_2, 32)
     
     up0_3 = keras.layers.Conv2DTranspose(32, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv1_2) # (56, 56, 32) << (pool1_2)
     merge0_3 = keras.layers.concatenate([conv0_0, conv0_1, conv0_2, up0_3], axis=3)
     conv0_3 = conv_2(merge0_3, 32)
     
     up0_4 = keras.layers.Conv2DTranspose(32, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv1_3) # (56, 56, 32) << (pool1_3)
     merge0_4 = keras.layers.concatenate([conv0_0, conv0_1, conv0_2, conv0_3, up0_4], axis=3)
     conv0_4 = conv_2(merge0_4, 32)
     
     # output 
     output1 = keras.layers.Conv2D(1, 1, padding = "same", activation = "sigmoid")(conv0_1)
     output2 = keras.layers.Conv2D(1, 1, padding = "same", activation = "sigmoid")(conv0_2)
     output3 = keras.layers.Conv2D(1, 1, padding = "same", activation = "sigmoid")(conv0_3)
     output4 = keras.layers.Conv2D(1, 1, padding = "same", activation = "sigmoid")(conv0_4)
     output = (output1 + output2 + output3 + output4) / 4
     
     model = keras.models.Model(inputs = input_layer, outputs=output)
     return model   
 ```
 >

- [x] **5. 코드가 간결한가요?**  
  - 네, 케라스기반이라 그런지 딱히 복잡하고 그런부분은 없었습니다.  
 ```python
 
 ```
 >

## **참고링크 및 코드 개선 여부**  
------------------
- 제가 봤던 U-Net++에서는 출력층이 하나였는데 여기서는 4개의 output으로 구한다음 평균을 낸게 신기했습니다. 
- 코드의 모듈화와 재 사용성에 있어서는 이쪽이 더 나은거 같습니다
- Conv2DTranspose로 upsampling을 구현하신 반면 저는  UpSampling2D 레이어를 사용했습니다. 이때 정확성면에선 Conv2DTranspose가, 연산량 효율성 면에선 UpSampling2D 사용이 나은것 같습니다.

