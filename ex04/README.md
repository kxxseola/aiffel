# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김설아
- 리뷰어 : 박재영

----------------------------------------------

# PRT(PeerReview)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

    > 코드가 정상적으로 작동하며, 주어진 문제를 해결함(Accuracy : 0.8512)  
    > Word2Vec+BestModel 조합으로 최적의 성능을 발견함

- [O] 주석을 보고 작성자의 코드가 이해되었나요?

    > 주요 코드에 대한 주석이 적시되어 있어 코드에 이해가 쉬웠습니다.  
    > 주요 기능별로 모듈화 되어 있어 모델 결과를 쉽게 파악할 수 있었습니다.
    ```python
     # 테스트셋으로 평가
      rnn_results = rnn_model.evaluate(x_test,  y_test, verbose=2)
      cnn_results = cnn_model.evaluate(x_test,  y_test, verbose=2)
      gloMP_results = gloMP_model.evaluate(x_test,  y_test, verbose=2)

      print('='*70)
      print('TEST SET 평가 결과')
      print('RNN : ', rnn_results)
      print('1-D CNN : ', cnn_results)
      print('GlobalMaxPooling1D : ', gloMP_results)
      print('='*70)```

- [O] 코드가 에러를 유발할 가능성이 있나요?
  
     > 에러를 유발 시킬만한 코드를 발견하지 못했습니다.

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

     > 각 모델별 성능과 차이를 이해하고, 상황에 맞게 적용
     ```
      # 가장 성능이 좋은 rnn으로 진행
      model = rnn_model
      embedding_layer = model.layers[0]
      weights = embedding_layer.get_weights()[0]
      print(weights.shape)    # shape: (vocab_size, embedding_dim)
     ```
----------------------------------------------

참고 링크 : 인터뷰 중 궁금해 하셨던.
  - [FastText](https://fasttext.cc/) 
