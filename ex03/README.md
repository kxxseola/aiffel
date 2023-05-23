# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김설아
- 리뷰어 : 김동규

----------------------------------------------

# PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

다양한 사진을 예시로 사용하여 작성자가 최대한 많은 경우의 수를 파악하려고 노력했음을 알 수 있으며, 적절한 결과 또한 도출했음.

- [O] 주석을 보고 작성자의 코드가 이해되었나요?

코드에 대한 설명이 첫줄에 포함되어 있어서 해당 셀의 기능에 대한 이해가 쉬움
```python
# 여러 장에 대해 진행해서 오류가 없는지 확실하게 확인하고 진행
df = pd.DataFrame(dlib_rects)
df['list_landmarks'] = list_landmarks
df = df.rename(columns={0:'dlib_rects', 1:'list_landmarks'}) 
df
```

관련 자료를 첨부하여 노트북을 읽어보는 독자가 쉽게 참고할 수 있도록 배려함
```md
## 3. 고양이 수염 필터 적용 위치 계산
![]()
...
```

함수 위에 기능에 대한 설명을 부착하여 해당 함수가 어떤 역할을 하는지 파악하기 쉬움
```python
# 고양이 수염 sticker 위치 설정 및 최종 이미지 출력/저장
def sticker_save(image_list):
...
```

- [X] 코드가 에러를 유발할 가능성이 있나요?

for 내부에서 사용하는 함수임에도 불구하고, 외부 스코프에 생성함.
이와 유사하게 for 내부에서 생성하고 소멸시키는 실수를 저지르지 않음.
```python
list_landmarks = []    # 여러 장의 사진들에 대한 landmark 정보 저장

for i in range(len(image_list)):
    image = image_list[i]
    list_points = []    
    ...
```

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

global 키워드 사용시 장점과 사용 안할경우 문제점이 무엇인가
```python
def bgr2rgb(images):
    global image_list
    image_list=[]
    
    for image in images:
    ...
```
답변: 명시적으로 전역변수를 사용 가능하고, 선언하지 않으면 'image_list=[]' 실행시 새로운 변수가 생성됨.

회전행렬에 대해서 설명할 수 있는가?
각 a도에 대한 회전을 다음과 같이 표현 가능하다.
\[\[y'],\[x']] = \[\[cos a, -sin a]\[sin a, cos a]]\[\[y], \[x]]
이러한 행렬 연산 과정을 통해 이루어 지는 것으로 알고 있고, openCV에서는 getRotationMatrix2D 함수를 호출하여 회전을 구현할 수 있다.

- [O] 코드가 간결한가요?

라인 수 자체는 길지만, 라인당 표현식을 최소화하여, 코드를 간결하게 작성함.
아래는 여러 사례들중 
```python
# 스티커 좌표 설정   
    print(landmark[30])
    w = h = dlib_rect[0].width() 
    x = landmark[30][0] - w//2
    y = ((landmark[40][1] + landmark[50][1])- h)//2 
    
    # 스티커 좌표 저장
    site = (x, y, w)
    sticker_site.append(site)
    print (f'{i}번째 필터 (x,y) : ({x},{y})')  
    print (f'{i}번째 필터 (w,h) : ({w},{h})')  
    
    # 스티커 회전
    left_eye = landmark[36] 
    right_eye = landmark[45] 
    angle_x = right_eye[0] - left_eye[0]
    angle_y = -(right_eye[1] - left_eye[1])
    print(angle_x, angle_y)
    
    angle = math.atan2(angle_y,angle_x)
    angle = int(angle * 180 / math.pi)
    print('angle : ',angle)
```

----------------------------------------------

# 참고 링크 및 코드 개선
- https://blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221357923005&parentCategoryNo=&categoryNo=198&viewDate=&isShowPopularPosts=true&from=search
- 주석을 활용한 함수 설명도 너무 좋지만, 거대 라이브러리를 구성하는 상황에서는 읽을 수 없는 경우가 있어요. 
Docstring을 한번 활용해보시는 것도 좋은 경험이 되실 것 같아요.
참고 링크: https://www.programiz.com/python-programming/docstrings
