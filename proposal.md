# 제3회 KRX 금융 언어모델 성능평가 경진대회

## 1. 참가자 인적사항

1) 정헌수
2) 개인
3) hunsoo.jung@buenaworks.com
4) DoumAI

## 2. 모델 세부 정보
1) 모델 링크: [https://huggingface.co/DoumAI/Krx-Bench-2024-Qwen2.5-7B-Instruct-1104](https://huggingface.co/DoumAI/Krx-Bench-2024-Qwen2.5-7B-Instruct-1104)
2) 베이스 모델: [unsloth/Qwen2.5-7B-Instruct](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)
3) 데이터셋 설명
    - AI Hub > 금융 분야 다국어 병렬 말뭉치 데이터 > 라벨링데이터 > 영어
4) 학습 방법론 요약
    1) AI Hub 에서 데이터셋 다운로드    
    2) Q&A 생성에 맞게 파싱 (`parse_raw.py`)
    3) Q&A 데이터셋 생성 (`generate_qnas.py`)
    4) 생성된 데이터셋을 병합 (`merge_generated.py`)
5) 총 학습시간
    - 학습: 약 10시간 (Tesla T4 15 GB, CUDA 12.2)
    - 모델 저장 및 HF 업로드: 약 30분

## 학습 데이터셋

1) 학습 데이터셋 링크

    - [AI Hub 금융 분야 다국어 병렬 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71782) > 02.라벨링데이터 > `*_영어.zip` > `*.json`
    - [alvanlii/finance-textbooks](https://huggingface.co/datasets/alvanlii/finance-textbooks)

2) 데이터셋의 총 크기

    - 병합 후: 11.5 MB (75,538 rows, CSV format)

3) 데이터셋의 구성

    - 원본

        ```json
        {"meta": {"doc_no": "", "domain": "", "category": "", "license": "", "source_language": "", "taget_language": ""}, "sents": {"page": 0, "sn": "", "source_original": "", "source_cleaned": "", "mt": "", "mtpe": ""}}
        ```
    - 파싱 후
        ```csv
        content,_is_related
        ```
    
    - Q&A 데이터셋 생성 후
        ```csv
        question,answer
        ```

4) 데이터셋의 출처
    - [AI Hub](https:/www.aihub.or.kr)
    - [alvanlii/finance-textbooks](https://huggingface.co/datasets/alvanlii/finance-textbooks)
5) 데이터 제작/처리/검수에 사용 된 방법론에 대한 기술
    1) (제작) 원본 AI Hub 해당 데이터셋 중 "영어" 데이터만 선택하여 다운로드 합니다.
    2) (제작) 압축 해제 후 JSON 파일들을 읽어 ["sents"]["source_cleaned"]를 붙여 한 개의 컨텐츠로 만듭니다.
    3) (제작) `question`과 `answer` 컬럼이 있는 Q&A 셋을 생성합니다.

        - 위 파싱된 `content`를 문장 단위의 chunk로 랜덤 샘플링 한 뒤 GPT-4o-mini로 Q&A를 생성하였습니다. 이 때, `chunk_size`를 4, 8, 12, 16으로 주어 샘플링 하였습니다. 또한 GPT 요청 시 구조화된 응답을 사용하여 파싱이 용이하도록 하였습니다.

        - 단, stock은 샘플에서 inc-N은 주가의 N일 전 종가 - 현재 종가로 추정하였습니다. 10일 이상 데이터가 없으므로, inc-5만 활용 가능하도록 window size를 5로 설정하여 수동으로 생성하였습니다.

    4) (처리) 동일한 질문은 병합 시 중복 제거 하였습니다. `merge_generated.py`
    5) (검수) 선택적으로 GPT-4o-mini를 이용해 사업, 경제, 재무, 회계, 투자 관련 컨텐츠만 선별하였습니다.

        ```python
        prompt: str = (
            "Answer Y only if the following content is related to business, "
            "economics, finance, accounting, investment, or related area.\n"
            f"Content: {parsing_sr['content'][:500]}"
        )
        messages = [
            {
                "role": "system",
                "content": "You're an exprt discriminator fluent in Korean.",
            },
            {"role": "user", "content": prompt},
        ]
        response = generate_gpt(
            messages=messages, model="gpt-4o-mini", temperature=0.1
        )
        ```
6) 데이터셋에 포함된 금융 지식에 대한 기술
    - 주가 예측 컬럼의 의미 추정

        - `open`, `high`, `low`, `close`: 해당 주식의 전일 종가 대비 변동률(%) (소수점 2째 자리에서 반올림)
        - `adj-close`: 컬럼명으로는 수정 종가로 추정되나 학습 데이터에는 `close`와 동일하게 취급
        - `inc-N`: N일 전 종가 - 당일 종가 (따라서 `dec-N`이 더 설득적)
7) 데이터셋 라이선스
    - "본 AI데이터 등은 인공지능 기술 및 제품·서비스 발전을 위하여 구축하였으며, 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다." [링크](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)

## 학습 방법론

1) 학습에 사용한 코드 링크

    - [Colab drive](https://drive.google.com/drive/folders/1ZYCssC5B1kRTrao9lK8EKTu-odwv6wZT?usp=sharing)

    - [GitHub repo.](https://github.com/jung-hunsoo/krx-finllm-2024) (11/7일 이후 공개 전환)
    
2) 하이퍼파리미터 기술

    - PEFT
        - target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        - lora_alpha = 16
        - lora_dropout = 0

3) 학습 중 사용된 방법론에 대한 기술

    - Supervised Fine-tuning

## 성능 지표

1) 대회 벤치마크 성능 지표 기록

    - 

2) 기타 벤치마크를 사용했다면 추가 기록

    - 

3) 향후 성능 지표 향상을 위한 학습 개선 계획

    - 

## 기타 포함되면 좋은 정보

1) 타 금융 LLM들과 차별화되는 독특한 특징

    - [한국어] 한국어 데이터를 주로 활용
    - [균형] 고성능의 `gpt-4o`와 가성비의 `gpt-4o-mini`를 적절히 혼용
    - [확장성] 약간의 코드 수정으로 새로운 공개 데이터셋을 추가로 활용 가능
    - [성능] 고성능의 `polars` 사용

2) 모델 개발 과정에서 참고한 논문 및 연구

    - 

3) 해당 모델의 실제 금융 환경에서 활용 가능성

    - 

4) 해당 모델의 한계점 및 잠재적 부정적 영향

    - 
