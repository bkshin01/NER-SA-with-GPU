# 🔖NER-SA-with-GPU

NER 모델과 SA 모델을 Flask와 GPU 등을 사용해 서빙

### 1. 가상환경 설정
`python -m venv venv`로 생성, 
- Window: `source venv/Scripts/activate`


### 2. 패키지, CUDA Toolkit 설치
- GPU에 맞는 CUDA Toolkit 설치
- *torch는 GPU와 호환되는 버전으로 다운받아야 함*
- requirements.txt 추가 예정
1) 직접 다운로드

```
pip install flask
pip install pandas
pip install tensorflow
pip install transformers
pip install tqdm
pip install kss

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

2) requirements.txt 사용
- 디렉토리 주의
※ torch는 위의 방법으로 다운로드 권장
```
pip install -r requirements.txt
```

### 3. 모델 경로 입력
`model_loader.py`안의 `ner_model_path`와 `sa_model_path` 입력.
- NER은 "NER_model" 폴더 경로 입력.
- SA는 klue_large_fold3_s.pth의 경로 입력.

### 4. GPU 사용 확인 (선택)
`cuda_check.py`로 GPU를 사용할 수 있는지 확인.
torch의 버전 + GPU와 True가 나와야 함.
- `+GPU`가 나오지 않는 경우: uninstall current torch and download torch with GPU
- `True`가 나오지 않는 경우: CUDA Toolkit 설치 여부와 버전, 호환 여부를 확인

### 5. app.py 실행
String 형태의 본문 입력, 다음은 본문 예시 
- 출처: 세계일보(https://www.segye.com/newsView/20240321512455)

    공정거래위원회는 메가스터디교육이 에스티유니타스(공단기) 주식 95.8%를 1030억원에 취득하는 기업 결합에 대해 인수 금지 조치를 부과했다고 21일 밝혔다.

    공정위에 따르면 현재 공무원시험 학원 시장 1위 사업자인 공단기는 지난 2012년 한 번의 구매로 모든 강의를 수강할 수 있는 패스 상품을 출시했다. 단과 상품 위주였던 공무원 학원 시장은 패스 상품 출시 이후 급변했다. 공단기가 패스 상품을 저가로 공급하면서 여러 학원에 분산돼 있던 인기 강사들이 유입됐고, 공단기는 2019년까지 독점적 지위를 유지했다. 공단기의 시장점유율이 올라가면서 출시 당시 30만원대에 불과했던 패스 상품 가격은 2019년 최고 285만원까지 비싸졌다.

    하지만 메가스터디가 2019년 시장에 진입하면서 공무원 학원 시장은 공단기와 메가스터디 양사 경쟁체제로 변화됐고, 공단기 패스 상품의 평균가격도 2019년 166만원에서 2022년 111만원까지 하락했다.

    공정위는 이런 상황에서 양사의 결합이 ‘7·9급 공무원 및 군무원시험의 온라인 강의 시장’과 ‘소방공무원시험 온라인 강의 시장’에서 경쟁하는 기업 간 수평적 결합에 해당한다고 보고 경쟁제한성 여부를 판단했다.

    검토 결과 두 회사 결합 이후 각 시장에서 합산 점유율이 75%(소방공무원 시장)까지 높아지고, 2위 사업자와의 격차도 60%포인트 이상 벌어지는 것으로 조사됐다. 또 결합 후 인기 강사와 수강생이 집중되면서 수강료 인상 등 수험생들의 피해가 발생할 우려가 큰 것으로 분석됐다. 실제 공정위가 20개 과목의 인기 강사 40명의 소속 현황을 분석한 결과 공단기와 메가스터디에 각각 23명, 13명이 속해 있는 것으로 파악됐다. 전속계약 및 수험생 수에 비례해 강사료를 지급받는 구조 등을 감안하면 기업 결합 후 인기 강사들이 결합 회사에 남을 유인이 크다고 공정위는 설명했다. 공정위는 “인기 강사 외에 전체 강사진, 개설 과목 수 등 측면에서도 경쟁사 대비 매우 우월해져 결합 후 가격인상 등을 시도하더라도 적시에 대항할 경쟁사가 없다”면서 “인기 강사에게 지급하는 고정 강사료가 상품 가격으로 전이되고 시잠점유율이 증가할수록 (패스 등) 가격이 인상되는 것으로 분석됐다”고 밝혔다.

    공정위는 이에 인기 강사의 경쟁사 분사 등 행태적 조치나 자산매각 조치만으로는 경쟁제한 우려를 근본적으로 불식시킬 수 없다고 판단, 인수 금지조치를 부과했다. 전원회의 심의 이후 메가스터디 측은 기업결합 신고를 철회했다. 이번 불허 결정은 2016년 SK텔레콤의 CJ헬로비전 인수·합병 불허 이후 8년 만이다.


### 6. 결과 확인

![image](https://github.com/imsinusinu/NER-SA-with-GPU/assets/118626963/fd4afbbc-466b-4320-9bd7-7617d2ae8492)

