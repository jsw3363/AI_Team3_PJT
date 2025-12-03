###  AI_Team3_PJT
## 인공지능 응용 3조 팀 프로젝트

---
- [팀 프로젝트 Notion](https://www.notion.so/2025-2-293cdb36440180e28603e3f1dd4d7672)
- [PPT](https://www.canva.com/design/DAG6X99Y5-g/g2xnIffP5ZEbrTEKx8CS0w/edit)
- [Report](https://www.canva.com/design/DAG6YACiZK0/lWD9whFNOlBcU3m48QcSOA/edit)
---
#### 논문 선정
**Feng, S., Keung, J., Yu, X., Xiao, Y., & Zhang, M. (2021). Investigation on the stability of SMOTE-based oversampling techniques in software defect prediction. Information and Software Technology, 139, 106662.**

<요약>
- **목적** : SMOTE 기반 오버샘플링 기법들이 랜덤성 때문에 불안정(unstable)한 성능을 보이는 문제를 분석하고, 이를 개선한 Stable-SMOTE 계열(Stable SMOTE, S-Borderline, S-ADASYN 등)을 제안·평가.
- **주된 주장** : Stable 버전은 합성 데이터 생성의 랜덤요소를 줄여 분산(성능의 변동성)을 낮추고, 평균 성능도 향상시킨다(이론적/경험적 증거 제시).
- **데이터·실험** :  PROMISE 리포지토리의 SDP(software defect prediction) 데이터셋들(결함비율 < 40%인 것들), 여러 분류기와 반복 실험(bootstrap + 오버샘플링 반복)으로 안정성 비교.

#### Reference

- Paper(pdf) : https://yanxiao6.github.io/papers/stability_IST21.pdf

- Code(Git) : https://github.com/ShuoFENG0527/stability-of-smote/tree/main

--- 
#### 프로젝트 진행 방향

1.  사용할 데이터셋 선정
    - [x] 논문에서 제시한 데이터셋을 그대로 사용
    - [ ] 필요에 따라 클래스 불균형 데이터셋으로도 테스트

2.  레퍼런스 코드 수정 및 시행
    - [x] 개발환경 생성 및 Migration
    - [x] 주석 설명 추가

3.  결과 비교 분석
    - [ ] 평가 지표 분석
    - [ ] 결과 분석

4.  시각화
    - [ ] 시각화

--- 
