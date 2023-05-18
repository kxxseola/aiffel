# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김설아
- 리뷰어 : 김동규

----------------------------------------------

# PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
컬럼에 대한 설명을 기입하여 앞으로 다룰 데이터에 대해 빠르게 파악됨
```md
- datetime - hourly date + timestamp
- season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
...
```
미리 표시하여 어디서 변수가 추가되는지 파악됨
```python
# w, b 준비
import numpy as np
w = np.random.rand(10)
b = np.random.rand()
```

- [X] 코드가 에러를 유발할 가능성이 있나요?
w와 b가 해당 코드 블록에 정의 되어 있지 않아서 오해의 여지가 있지만, 모든 변수가 미리 정의 및 초기화가 이루어짐
```python
learning_rate = 0.05
losses = []

for i in range(1, 20001):
  dw, db = gradient(X_train, w, b, y_train)
  w -= learning_rate * dw
  b -= learning_rate * db
  L = loss(X_train, w, b, y_train)
  losses.append(L)
  if i % 100 == 0:
    print('Iteration {} : Loss {}'.format(i, L))
```
아래의 두 코드는 각 메트릭스의 크기를 확인한다.
이를 통해 행렬 연상 상의 문제를 미리 예방했다.
```python
print(df_X.shape)
print(df_y.shape)
```
----
```python
print(df_X_drop_bmi.shape)
```
함수의 정의 순서와 호출 순서가 일치해서 함수 hoisting 문제가 발생하지 않는다.
```python
# MSE 선언
def MSE(x, y):
  mse = ((x - y)**2).mean()
  return mse

# MSE 첫 사용
def loss(X, w, b, y):
    predictions = model(X, w, b)
    L = MSE(predictions, y)
    return L
     
```
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
drop과 [[]]을 썼을 때 차이를 아시나요?
```
df = pd.DataFrame(diabetes['data'], index=diabetes['target'], columns=diabetes['feature_names'])
df_X_drop_bmi = df.drop('bmi', axis=1)
```
답변: drop은 자기 자신을 변경하고, 대괄호 2개 쓰는 표현은 새로운 데이터프레임을 반환한다.

- [O] 코드가 간결한가요?
아래의 항목들의 경우 충분히 간결하게 잘 작성되었다고 생각함.
```python
plt.scatter(X_test[:, 4], y_test, label="true")
plt.scatter(X_test[:, 4], predictions, label="pred")
plt.legend()
plt.show()
```

```python
def model(X, w, b):
  predictions = 0
  for i in range(10):
    predictions += X[:, i] * w[i]
  predictions += b
  return predictions
```

```python
def MSE(x, y):
  mse = ((x - y)**2).mean()
  return mse
```

----------------------------------------------

# 참고 링크 및 코드 개선
- https://developer.mozilla.org/ko/docs/Glossary/Hoisting
- from import 는 최상단에 넣는 것이 좋습니다. (또는 함수 내부에서 가장 윗줄)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # <- 이동됨

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

```
- 테스트는 필요하지만, 반복되는 코드를 줄일 수 도 있어요. 시간이된다면 리펙토링에 재미를 가지시는 것도..
```python
axs = []
for i in range(6):
    axs.append(fig.add_subplot(2, 3, i+1))
    
for i, col in enumerate(['year', 'month', 'day', 'hour', 'minute', 'second']):
    sns.barplot(x=df[col], y=df['count'], ax=axs[i])
```

