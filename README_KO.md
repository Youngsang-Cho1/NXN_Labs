# OutfitTransformer: 패션 코디 아이템 추천 AI

주어진 **부분 코디(Partial Outfit)**를 보고 빠진 아이템을 예측하는 AI 패션 추천 시스템입니다.  
단순 이미지 유사도 검색이 아닌, **여러 아이템 간의 조합·조화(Context)**를 이해하여 최적의 아이템을 추천합니다.

---

## 핵심 아이디어

SigLIP만으로는 **개별 옷 한 벌씩**밖에 이해할 수 없습니다.  
우리가 만든 **OutfitTransformer**는 여러 옷을 동시에 받아 Self-Attention으로 아이템 간의 관계를 모델링하고, 코디를 완성하는 데 필요한 아이템의 벡터를 예측합니다.

```
[SigLIP]              -> 개별 옷의 특징 추출 (번역기, Frozen -- 학습 안 함)
[OutfitTransformer]   -> 코디 전체를 보고 빠진 아이템 예측 (탐정, 학습 대상)
```

> **왜 SigLIP을 Frozen 시키나요?**  
> SigLIP은 이미 수억 개의 패션 이미지-텍스트 쌍으로 학습된 강력한 모델입니다.  
> 이를 다시 학습하면 Catastrophic Forgetting이 발생할 수 있고, 학습 시간도 수백 배 늘어납니다.  
> 따라서 SigLIP은 "번역기"로만 활용하고, 모든 학습은 OutfitTransformer에 집중합니다.

---

## 모델 아키텍처 (`model.py`)

```
아이템 이미지 / 텍스트
      |
[SigLIP Vision/Text Encoder]   <- Frozen (항상 고정)
      | 768차원 벡터
      v
[TransformerEncoder]            <- 학습 (Self-Attention으로 코디 관계 파악)
      |
[MLP Projection Head]           <- 학습 (Linear->GELU->Linear, SigLIP 공간에 정렬)
      |
[L2 Normalize]
      |
   예측 벡터 (768D)     <->   정답 아이템의 SigLIP 이미지 벡터와 비교
```

**학습되는 파라미터 3가지:**

| 컴포넌트 | 역할 | 비유 |
|---|---|---|
| `TransformerEncoder` | 아이템 간의 관계(Self-Attention) 학습 | 탐정 |
| `MLP Projection Head` | Transformer 출력을 SigLIP 이미지 공간에 매핑 | 통역사 |
| `logit_scale` | InfoNCE Loss의 온도(Temperature) 자동 조절 | 채점 기준 |

---

## 데이터셋 & 메타데이터 (`dataset.py`)

**사용 데이터셋 (HuggingFace):**
- `owj0421/polyvore`: **251,008개** 개별 패션 아이템 (이미지 + 메타데이터)
- `owj0421/polyvore-outfits`: **53,306개** 코디 세트 (아이템 ID 조합)

**각 아이템의 메타데이터 필드:**

| 필드 | 내용 | 활용 방식 |
|---|---|---|
| `title` | "Black Slim Fit Leather Jacket" | SigLIP 텍스트 인코딩 (최우선) |
| `category` | "outerwear" | Hard Negative Mining (같은 카테고리 오답 추출) |
| `description` | 상세 설명 | title 없을 때 fallback |
| `url_name` | URL 기반 이름 | 최후 fallback |
| `image` | PIL Image | SigLIP 이미지 인코딩 |

**학습 문제지 생성 방식 (Hard Negative Mining):**
```
코디: [재킷, 바지, 신발]
        | 랜덤으로 하나 숨김
Context:   [재킷, 바지]       <- 힌트
Target:    [신발]             <- 정답
Hard Neg:  [다른 신발들]      <- 같은 카테고리 오답 (진짜 헷갈리는 것)
Easy Neg:  [완전 다른 아이템] <- 랜덤 오답
```

> **Hard Negative Mining이 왜 중요한가요?**  
> 오답으로 완전히 다른 아이템(예: 신발 vs 가방)을 쓰면 모델이 너무 쉽게 맞춥니다.  
> 같은 카테고리(신발 vs 다른 신발)에서 오답을 골라야 모델이 더 세밀한 패션 감각을 학습합니다.

---

## 속도 최적화: 벡터 사전 캐싱 (`vectorize_data.py`)

| 방식 | 1 Epoch 소요 시간 |
|---|---|
| 기존 (매 배치마다 SigLIP 재실행) | ~2시간+ |
| 개선 (사전 캐싱 벡터 활용) | **~7분** |

25만 개 아이템을 **딱 한 번만** SigLIP에 통과시켜 `polyvore_embeddings.pt`로 저장합니다.  
이후 학습/평가는 이미지 처리 없이 벡터만 불러와서 처리합니다.

---

## 성능 향상 전략 (Fine-Tuning)

| 전략 | 설명 | ML 분류 |
|---|---|---|
| **Learnable Temperature** | InfoNCE loss의 `logit_scale`을 고정값 대신 학습 파라미터로 | 최적화 |
| **MLP Projection Head** | Transformer 출력에 필터 레이어(Linear->GELU->Linear) 추가 | 아키텍처 |
| **Text Dropout** | 학습 중 20% 확률로 텍스트 메타데이터를 제거 - 시각적 조화 강제 학습 | Regularization |
| **Grid Search** | LR, 레이어, Dropout, Epoch 등 하이퍼파라미터 자동 탐색 | 튜닝 |

> **Text Dropout이 Regularization인 이유:**  
> 모델이 항상 텍스트 힌트("검은 가죽 재킷")에 의존하면 텍스트 패턴만 암기하게 됩니다.  
> 힌트를 20% 확률로 차단하면 모델이 어쩔 수 없이 **시각적 조화**를 직접 학습해야 합니다.

---

## 파일 구조 (실행 순서)

```
1. vectorize_data.py  -> 25만 개 아이템 벡터화 (최초 1회, ~2시간)
2. dataset.py         -> 학습용 문제지 생성 (train.py 내부에서 호출)
3. model.py           -> OutfitTransformer 아키텍처 정의
4. train.py           -> 모델 학습 (기본 3 에폭)
5. eval_fitb.py       -> FITB 정확도 평가 (4지선다 500문제)
6. grid_search.py     -> 하이퍼파라미터 자동 최적화
7. demo.py            -> 코디 자동 완성 시연 (자기회귀 방식)
```

---

## 빠른 시작

```bash
# 1. 벡터 사전 처리 (단 한 번만 실행, ~2시간)
python vectorize_data.py

# 2. 모델 학습 (~7분/에폭)
python train.py

# 3. 성능 평가 (수초)
python eval_fitb.py

# 4. (선택) 하이퍼파라미터 그리드 서치
python grid_search.py

# 5. (선택) 코디 완성 데모
python demo.py
```

**CLI 파라미터 직접 조절 가능:**
```bash
python train.py --lr=1e-3 --epochs=5 --text_dropout=0.1 --num_layers=6
```

---

## 평가 지표

**FITB (Fill-In-The-Blank)**
- 코디에서 아이템 하나를 제거하고 4지선다로 맞추는 방식
- Random Baseline: **25%** (4지선다 무작위 선택)
- 현재 모델: **~35%**
- 논문 SotA (복잡한 Metric Learning 없이): ~50%대
