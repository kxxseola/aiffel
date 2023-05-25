# AIFFEL Exploration
----  
## **Code Peer Review Templete**
------------------
- 코더 : 김설아
- 리뷰어 : 사재원

## **PRT(PeerReviewTemplate)**  
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
네, 다양한 객체들에 대한 결과물 도출, 크로마키 적용, 문제점 솔루션 제시 및 실제 구현까지 프로젝트의 요구사항들을 모두 충족시켰다고 생각합니다.  
- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**   
네 코드 중간중간 마다 주석을 달아주셔서 어떤의도로 작성했는지 쉽게 알 수 있었으며 전체적인 코드의 흐름도 파악하기 쉬웠습니다.   
  

- ['x] **3. 코드가 에러를 유발할 가능성이 있나요?** 
  순차적으로 코드를 실행한다면 에러를 발생한 요소는 없습니다.
  

- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  네, 프로젝트의 요구를 모두 충족하였고 그 과정에서 함수들에 대한 설명 주석을 작성해줌으로써 오히려 리뷰어입장에서도 전반적으로 쉽게 코드 이해가 가능했습니다.


- ['x] **5. 코드가 간결한가요?**  
  코드를 보면 비슷한 내용인 함수를 두번씩 작성하셨는데 하나로 합칠 수 있어보입니다.  
  
```
#예시로 두 코드를 가져와봤는데 함수를 호출하기 이전 seg_color미리 계산하고 
#함수 인자로 만들게된다면 하나의 함수로 만들 수 있어보입니다. 
def make_seg_mask(image_list, output_list, seg_color):
    img_mask_list = []
    # img_show_list = image_list.copy()
    
    for i in range(len(image_list)):
        output = output_list[i]
        seg_map = np.all(output==seg_color, axis=-1) 
        # True과 False인 값을 각각 255과 0으로 바꿔줍니다
        img_mask = seg_map.astype(np.uint8) * 255
        img_mask_list.append(img_mask)
        
    return img_mask_list
    
def make_seg_mask(image_list, output_list, segvalues_list, label_num):
    img_mask_list = []
    # img_show_list = image_list.copy()
    
    for i in range(len(image_list)):
        output = output_list[i]
        segvalues = segvalues_list[i]
        seg_color = segvalues['class_colors'][segvalues['class_ids'].index(label_num)]
        seg_color = seg_color[2::-1]    # bgr -> rgb
        seg_map = np.all(output==seg_color, axis=-1) 
        # True과 False인 값을 각각 255과 0으로 바꿔줍니다
        img_mask = seg_map.astype(np.uint8) * 255
        img_mask_list.append(img_mask)
        
    return img_mask_list

```

## **참고링크 및 코드 개선 여부**  
------------------  
- 문제점 솔루션을 제시하셨을때 이미지 밝기를 통한 작업이 있었는데
  만약에 이미지의 전체적의 밝기 분포를 조정하고 싶으시다면 아래 링크도 살짝 읽어보시면 좋을 것 같습니다.  
  https://bkshin.tistory.com/entry/OpenCV-10-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8
  
  혹은 특정영역만 어두운 경우 그림자를 제거하는 Water-Filling 알고리즘 방식도 있다고 들었습니다.
