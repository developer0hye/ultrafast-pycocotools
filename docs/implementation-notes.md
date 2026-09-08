# Implementation notes

Historical measurements and design decisions. These include different synthetic prediction recipes, thread counts and timing scopes; they are not the new public reproduction run. See [README](../README.md) and [public_benchmarks.json](../bench/results/public_benchmarks.json) for the current measured recipe.

# ultrafast-pycocotools

pycocotools를 그대로 대체하는 Rust 구현이다. `pip install` 한 줄로 바꿔 끼울 수 있고,
**AP가 pycocotools와 비트 단위로 같다**. 같은 입력에서 `precision` / `recall` /
`scores` 배열 전체가 바이트까지 동일하며, 그걸 tolerance가 아니라 바이트 비교로
테스트한다.

> **상태: 아직 PyPI에 올라가 있지 않다.** 소스에서 빌드해야 한다
> ([§직접 해보기](#직접-해보기)). Rust toolchain이 필요하고, wheel을 배포하면
> 그때는 필요 없어진다.

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools
cd ultrafast-pycocotools
pip install maturin && maturin develop --release
```

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("instances_val2017.json")
dt = gt.loadRes("detections.json")
ev = COCOeval(gt, dt, "bbox")
ev.run()          # evaluate() + accumulate() + summarize()
```

이미 pycocotools를 쓰는 코드가 있으면 import 이름만 넘겨받으면 된다:

```python
from ultrafast_pycocotools import init_as_pycocotools
init_as_pycocotools()          # torchvision / detectron2 / mmdetection 이 import 하기 전에
```

---

## 왜 이걸 쓰나

pycocotools를 바꿀 이유는 보통 셋 중 하나다. 자기 증상에서 출발해 해당 절로 가면 된다.

| 겪는 증상 | 원인 | 절 |
|---|---|---|
| eval 한 번이 epoch보다 오래 걸린다 | `evaluate()`가 (image × category) Python 루프 | [§속도](#속도) |
| 대형 데이터셋 eval에서 메모리가 터진다 | `evalImgs`가 K×A×I개 dict + numpy 배열을 전부 들고 있음 | [§메모리](#메모리) |
| 빠른 구현으로 바꿨더니 AP가 소수점 5~6자리에서 달라졌다 | threshold grid를 `np.linspace`가 아닌 방식으로 재구성 | [§동치 보장](#동치-보장) |
| 구현을 바꿨더니 서브클래스나 후처리 코드가 깨졌다 | 공개 메서드가 빠졌거나 반환 타입이 다름 | [§drop-in 호환](#drop-in-호환) |
| Windows에서 `pip install pycocotools`가 컴파일 에러 | Cython + MSVC 빌드 필요 | prebuilt wheel |
| detection이 tie score를 가질 때 구현마다 AP가 다르다 | greedy matcher가 정렬 순서에 민감한데 stable sort가 아님 | [§동치 보장](#동치-보장) |

**역할별로 어디까지 읽으면 되나**

- 그냥 빠르게 돌리고 싶다 → 위 설치/사용 예제까지면 충분하다.
- 논문 수치를 재현해야 한다 / CI에서 AP를 회귀 테스트한다 → [§동치 보장](#동치-보장)을 읽어야 한다.
- `COCOeval`을 상속했거나 `evalImgs`/`ious`를 직접 읽는다 → [§drop-in 호환](#drop-in-호환).
- 자체 metric을 붙이거나 오류 분석을 한다 → [§확장 기능](#확장-기능).
- 구현을 고치거나 기여한다 → [`DESIGN.md`](../DESIGN.md).

---

## 동치 보장

"거의 같다"가 아니라 **같은 비트**다. 테스트는 12개 요약 수치가 아니라
`eval["precision"]` (COCO 기준 T×R×K×A×M ≈ 80만 개 double) 전체를 `tobytes()`로
비교한다. 요약 수치는 차이를 평균으로 지워버리기 때문이다 — curve가 백 군데 틀려도
0.065로 똑같이 반올림된다.

실제 COCO val2017과 Objects365 val에서 측정한 결과:

YOLO11m 예측(431,145 detection)으로 잰 결과다. **요약 수치가 아니라 배열 전체**를
비교한다:

| 구현 | 다른 precision cell | 최대 cell 오차 | 최대 stat 오차 |
|---|---|---|---|
| **ultrafast-pycocotools** | **0** | **0.0e+00** | **0.0e+00** |
| faster-coco-eval 1.7.2 | 5,744 | 2.2e-16 | **0.0e+00** |
| hotcoco 0.5.0 | 6,069 | 8.5e-01 | 3.9e-05 |

**faster-coco-eval의 stat 오차가 0인 것이 함정이다.** 969,600개 cell 중 5,744개가
1 ULP씩 틀렸는데, 12개 요약으로 평균 내니 같은 double로 반올림된다. 요약만 보면
"완전히 일치"로 읽히고, 그건 이 저장소가 처음부터 경고해 온 바로 그 착시다 —
우리도 한동안 그렇게 잘못 판정했다.

실무적으로 AP를 소수점 3자리로 보고하면 1e-5는 안 보인다. 하지만 **개별 cell은
0.85까지 틀린다** — 평균이 감춰줄 뿐이다. YOLO11m 예측에서 hotcoco의 차이를 셀 단위로
분해하면:

| 크기 | cell 수 | 원인 |
|---|---|---|
| ~2e-16 (1 ULP) | 5,744 | `np.spacing(1)` 누락 |
| 1e-6 ~ 1e-2 | 84 | threshold grid 1 ULP |
| **> 1e-2** | **241** | threshold grid 1 ULP |

두 번째 원인의 메커니즘은 완전히 재현된다(`bench/diagnose_divergence.py`):

```
pycocotools recThrs[70] = 0x1.6666666666667p-1  (0.7000000000000001)
hotcoco     recThrs[70] = 0x1.6666666666666p-1  (0.7)
recall 7/10            = 0x1.6666666666666p-1  (0.7)

7/10 >= pycocotools thr : False  ->  샘플 없음  ->  precision 0.0
7/10 >= hotcoco thr     : True   ->  curve 샘플 ->  precision 0.85
```

GT가 10개인 category에서 recall이 정확히 7/10에 떨어지면 그 지점이 0이 되느냐 0.85가
되느냐가 갈린다. 희귀 category일수록 심하다.

### 어디서 어긋나는가 — 실제로 밟은 두 지점

**1. threshold grid를 재구성하면 1 ULP 어긋난다.**

pycocotools는 grid를 `np.linspace`로 만든다. 소스에 경고 주석까지 달려 있다:

```python
# np.arange causes trouble.  the data point on arange is slightly larger than the true value
self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
```

`np.linspace(0.5, 0.95, 10)`은 `[0.5 + 0.05*i]`와 **같지 않다**. 10개 중 2개가 1 ULP
다르고, `np.linspace(0, 1, 101)`은 101개 중 10개가 다르다:

```
iouThrs[7]  linspace 0.85               (0x1.b333333333333p-1)
            naive    0.8500000000000001 (0x1.b333333333334p-1)
recThrs     0.35 0.41 0.47 0.57 0.69 0.70 0.82 0.83 0.94 0.95  ← 10개가 1 ULP 차이
```

accumulate의 `np.searchsorted(rc, recThrs, side='left')`는 strict `<` 비교다.
recall은 `tp/npig`라서 0.35 같은 값에 **정확히** 떨어지는 일이 흔하고, 그 순간 두 grid는
서로 다른 precision을 샘플링한다. 그래서 hotcoco는 AP@0.50과 AP@0.75(grid index 0, 5는
양쪽이 동일)는 정확히 맞는데 AP@[0.50:0.95] 평균만 어긋난다.

우리는 grid를 Python에서 `np.linspace`로 만들어 engine에 **넘긴다**. Rust 쪽에서 다시
만들지 않는다.

**2. tie가 있으면 stable sort가 아니면 매칭이 달라진다.**

pycocotools는 `np.argsort(-scores, kind='mergesort')`로 정렬한다. score가 같은
detection들의 순서가 annotation 순서로 고정된다는 뜻이고, greedy matcher는 순서에
민감하다 — 먼저 온 detection이 GT를 가져간다.

score를 소수 3자리로 반올림한(현실적인) detection set에서 hotcoco는 AR까지 6.5e-7
어긋났고, tie를 없앤 set으로 다시 돌리니 AR 차이가 1e-16(noise)로 떨어졌다. 매칭 자체가
달라졌다는 뜻이다.

우리는 engine의 모든 정렬이 stable이고, `_prepare()`가 `loadAnns(getAnnIds(...))` 순서
그대로 넘겨서 "annotation 순서"의 정의를 양쪽에서 일치시킨다.

**3. precision의 epsilon.**

pycocotools는 `pr = tp / (fp + tp + np.spacing(1))`이다. `np.spacing(1)`을 빼면
`fp + tp == 1`인 지점 — 모든 curve의 첫 점 — 에서 1 ULP 어긋난다. 우리는 `f64::EPSILON`을
그대로 더한다.

### 검증 범위

`pytest tests/` 139개와 `cargo test` 57개가 전부 실제 pycocotools와 비교한다
(golden 파일이 아니라 live 비교).

### 통과 여부가 아니라 mutation score로 잰다

테스트가 초록인 것은 증거가 아니다. **구현을 일부러 망가뜨렸을 때 실제로 빨개지는가**가
증거다. `bench/mutation_check.py`가 21가지 mutation을 core에 넣었다 빼면서 그걸 잰다:

```bash
python bench/mutation_check.py
# caught 21/21
```

처음 돌렸을 때는 **21개 중 절반 가까이가 살아남았다.** 대표적으로:

| 살아남았던 mutation | 왜 안 잡혔나 |
|---|---|
| `bb_iou` 출력을 **transpose** | transpose를 잡으려고 만든 테스트의 fixture가 transpose에 대칭이었다 (`[[1,0],[0,0]]`) |
| `from_str`의 **sign extension 삭제** | round-trip corpus에 음수 delta가 하나도 없어 해당 분기에 진입조차 안 했다 |
| precision **envelope 통째로 삭제** | 모든 fixture가 detection 1~2개라 curve가 점이었다 |
| **crowd 재매칭 예외 삭제** | crowd 테스트의 detection들이 서로 다른 GT에 붙어서 "여러 개 흡수"를 안 밟았다 |
| OKS의 **평균 나눗셈 삭제** | keypoint에 Rust 테스트가 아예 없었다 |

전부 fixture를 고쳐서 잡히게 만들었다. 특히 sign extension은 corpus에 `vec![10, 5, 3, 2]`
한 줄을 넣는 것으로 끝났다 — 실제 마스크에서는 run이 짧아지는 게 흔한 일인데
corpus가 우연히 전부 증가하는 값이었다.

**Rust 57개** — Python 없이 도는 crate 단위 검증. 손으로 답을 낼 수 있는 크기의
시나리오로 규칙을 하나씩 못박는다. 실패했을 때 "AP가 움직였다"가 아니라 **어느 규칙이
깨졌는지**가 나온다.

```rust
assert_eq!(res.precision[0], 1.0 / (1.0 + EPS));
assert_ne!(res.precision[0], 1.0, "the epsilon was dropped");
```

`c_i32`가 Rust가 아니라 하드웨어 의미론을 따르는지(NaN → `INT_MIN`), tie가 annotation
순서로 풀리는지(개수가 아니라 **id**를 확인 — 총 TP/FP는 같으면서 배정만 바뀔 수 있다),
crowd가 detection을 **여러 개** 흡수하는지, IoU가 threshold에 **정확히** 걸릴 때 매칭되는지,
precision envelope이 오른쪽에서 왼쪽으로 전파되는지(그래서 fixture가 *올라가는* curve여야
한다), `0.0`과 `-1.0` sentinel 구분, OKS의 나눗셈 순서.

**Python 128개**

- **mask API 47개** — `encode`/`decode`/`merge`/`area`/`toBbox`/`iou`/`frPyObjects`를
  1×1 이미지, 빈 마스크, 꽉 찬 마스크, 이미지 밖으로 나간 polygon, **꼭짓점이 중복된
  polygon**(`rleFrPoly`가 0으로 나눠 NaN을 int로 캐스팅하는 지점), crowd flag까지.
- **실제 COCO 8개** — 커밋된 93 이미지 slice(`tests/data/`, 440 kB)로 fresh clone과 CI에서도 **항상 돈다**. 예전에는 20 MB annotation 파일이 있는 한 대의 머신에서만 돌고 나머지에서는 조용히 skip됐다. 실제 polygon은 꼭짓점이 수십 개에 ring이 여러 개고 crowd는 임의 형태의 uncompressed RLE라, 합성 데이터가 근사만 하는 부분이다.
- **fixture 구성 8개** — 다른 모든 테스트가 전제하는 어려운 케이스가 fixture에 **실제로 들어 있는지**. crowd 확률이 0으로 바뀌어도 parity 테스트는 전부 통과한다(crowd 없는 데이터에서는 양쪽이 완벽히 일치하니까). 스위트가 조용해지는 것이 parity 스위트의 최악의 실패 모드라 따로 못박았다. **이 가드가 바로 생성기의 실제 결함을 찾았다** — keypoint 가시성을 독립적으로 뽑느라 `num_keypoints == 0` 인스턴스가 (1/5)^17 확률이라 사실상 없었고, OKS의 해당 분기가 한 번도 안 밟히고 있었다.
- **평가 parity 16개** — 배열 전체를 바이트 비교. synthetic bbox/segm, 실제 COCO
  val2017 subset, keypoints, `useCats=0`, custom areaRng/maxDets/iouThrs,
  image/category subset, detection이 하나도 없는 경우, **모든 score가 동점인 경우**,
  `derive_segmentation=False`, `evalImgs` 전체.
- **drop-in 34개** — [§drop-in 호환](#drop-in-호환) 참조.
- **JSON loader 16개** — `json.load`와 float 비트까지 같은지. 커밋된 실제 COCO
  파일로 확인한다(이 테스트가 `serde_json`의 기본 float 파서가 1 ULP 틀리는 것을
  잡았다 — skip되는 테스트는 아무것도 못 잡는다).
- **확장 API 15개** — per-class AP가 mAP로 되돌아오는지, confusion matrix가 AP 회계와
  화해되는지 같은 불변식.
- **결정성 2개** — rayon 스레드 수(1 vs 8)가 결과 바이트를 바꾸지 않는지. 한 머신에서
  pycocotools와만 비교해서는 절대 못 잡는 실패 모드다.

---

## drop-in 호환

AP가 같은 것과 **코드가 그대로 도는 것**은 다른 문제다. 같은 숫자를 내면서도 `bytes`
대신 `str`을 돌려주거나, 서브클래스가 오버라이드하는 메서드가 없어서 깨질 수 있다.

그래서 `bench/audit_api.py`가 pycocotools의 공개 표면을 **열거해서** 대조한다. 우리가
손으로 적은 목록이 아니다 — 손으로 적은 목록엔 "기억한 것"만 들어간다. 현재 결과는
`0 incompatibilities`이고, 처음 돌렸을 때는 `COCOeval.computeIoU`,
`computeOks`, `evaluateImg` 셋이 없다고 나왔다. mmdetection 계열이 `evaluateImg`를
오버라이드하는 게 흔한 패턴이라, 없으면 오버라이드가 **조용히 아무것도 안 한다.**

`tests/test_dropin.py` 34개가 확인하는 것:

| 무엇을 | 왜 |
|---|---|
| 공개 메서드·속성 전부 존재 | 레퍼런스에서 열거하므로 우리가 몰랐던 것도 잡힌다 |
| 시그니처에서 인자가 빠지거나 필수 인자가 늘지 않음 | 기본값 있는 추가 kwarg는 가산적이라 허용 |
| `getAnnIds` 13가지 필터 조합의 **순서까지** | 순서가 score tie를 가른다 |
| `loadAnns`가 저장된 dict **그 자체**를 반환 (`is` 비교) | 호출자가 그걸 수정한다 |
| `counts`가 `str`이 아니라 `bytes` | drop-in이 깨지는 가장 흔한 방식 |
| `summarize()` 출력이 **문자 단위로 동일** | 사람들이 이걸 grep한다 |
| `computeIoU`/`computeOks`/`evaluateImg`/`ious`/`_gts`가 같은 값 | 서브클래스가 여기 의존한다 |
| torchvision `CocoEvaluator` 루프, numpy Nx7 `loadRes` | 실제 호출 패턴 |
| 서브클래스의 `evaluateImg` 오버라이드가 **실제로 호출됨** | 안 불리면 조용히 망가진다 |
| `init_as_pycocotools()` 후 `from pycocotools.coco import COCO` 전체 실행 | 서드파티가 밟는 경로 |
| **의도한 차이도 차이로 assert** | 어느 쪽이든 조용히 바뀌면 알려준다 |

성능 비용은 없다. `computeIoU`/`evaluateImg`/`ious`/`_gts`는 **처음 접근할 때** 만들어져서
안 쓰면 0이다 (호환 작업 전후 실측: bbox 0.1043s → 0.1059s, segm 0.1999s → 0.1979s,
메모리 동일 — 전부 노이즈 범위).

---

## 속도와 메모리

전부 실측이다. 재현은 `bench/compare.py`.

**바쁜 머신에서 wall-clock만 비교하면 불공정하다.** pycocotools는 단일 스레드라 배경
부하를 거의 안 타는데, rayon을 쓰는 구현은 같은 코어를 두고 경쟁하다 자기 효율과
무관한 wall을 잃는다. 그래서 이렇게 잰다:

- **`--threads 1`** — 전 구현을 단일 스레드로. 코어 수와 배경 부하를 질문에서 빼고
  알고리즘 차이만 남긴다. 이 수치가 머신을 건너서도 통한다.
- **CPU 시간을 wall 옆에 같이** — 경합에서 wall은 부풀지만 CPU는 안 부푼다. 둘이
  벌어지면 측정이 방해받았다는 신호다.
- **반복을 교차 실행** — A를 다 돌리고 B를 돌리면 드리프트가 뒤에 실행된 쪽에 몰린다.
- **머신 부하를 같이 기록**하고 spread를 보고한다. 0.1초대 측정은 단발 분산이 쉽게
  15%라 한 번 재고 "빨라졌다"를 판정하면 노이즈를 쫓게 된다.

**단일 스레드, YOLO11m 실제 예측** (5회, 부하 15–48%):

| 구현 | wall best | median | spread | cpu best | wall × | cpu × | peak RSS | 동치 |
|---|---|---|---|---|---|---|---|---|
| pycocotools | 22.135s | 22.478s | 22% | 21.672s | 1.0× | 1.0× | 1.29 GB | (기준) |
| faster-coco-eval | 3.527s | 3.559s | 4% | 3.516s | 6.3× | 6.2× | 1.24 GB | 배열 1 ULP |
| hotcoco | 1.168s | 1.349s | 33% | 1.156s | 18.9× | 18.7× | 1.25 GB | 최대 3.9e-05 |
| **ultrafast-pycocotools** | **0.548s** | 0.551s | 16% | **0.531s** | **40.4×** | **40.8×** | **641 MB** | **bit-identical** |

**단일 스레드에서 hotcoco보다 2.1× 빠르다.** wall ×와 cpu ×가 40.4/40.8로 일치하니
이 측정은 부하에 크게 흔들리지 않았다. 병렬을 켜면 격차가 더 벌어지지만 그 수치는
측정 머신의 유휴 코어 수에 좌우되므로, 아래 표들은 참고용으로 읽어야 한다.

아래는 기본(병렬) 설정이다. 각 구현은 별도 프로세스에서 돌렸고(peak RSS가 섞이지
않도록), 구현마다 3회 돌려 **best**를 적었다. `eval`은
`evaluate + accumulate + summarize` 합계다.

### 실제 모델 출력으로 검증

합성 detection은 GT에서 유도된다 — box가 GT와 상관되어 있고 score는 3자리로 반올림해
동점이 수천 건이다. 동점은 stable sort가 걸리는 지점이라 일부러 어렵게 만든 것이지만,
실제 모델 출력은 **정반대**다: float32 score라 동점이 거의 없고, box는 detector 자신의
prior에서 나오며, 이미지당 개수가 훨씬 많다. 둘 다에서 맞아야 한 생성기의 우연이
아니다.

세 개의 구조적으로 다른 detector를 COCO val2017에 직접 돌려 확인했다
(`bench/predict_coco*.py`):

| 모델 | 구조 | iouType | detection | 이미지당 | 측정 AP | 공식 |
|---|---|---|---|---|---|---|
| YOLO11m | anchor-free + NMS | bbox | 431,145 | 86.2 | 0.507 | 51.5 |
| YOLO11m-seg | + prototype mask | segm | 431,006 | 86.2 | — | |
| RF-DETR base | DETR query, NMS 없음 | bbox | 1,500,000 | 300.0 | 0.532 | ~53–54 |
| Mask R-CNN | two-stage, per-RoI mask | segm | 171,031 | 34.2 | **0.346** | **34.6** |
| Keypoint R-CNN | two-stage, OKS | keypoints | 74,143 | 14.8 | 0.600 | 61.1 |

AP가 공식 수치와 맞으므로 파이프라인 자체가 옳다(Mask R-CNN은 34.6에 정확히 일치).
**다섯 경우 모두 bit-identical**이다.
keypoints가 특히 의미 있는데, OKS는 `exp()`를 쓰는 유일한 경로라 numpy의 벡터화된
`exp`와 Rust libm의 `exp`가 갈릴 수 있다고 처음부터 위험으로 적어뒀던 곳이다.

### 두 대의 머신에서

"pycocotools와 같다"는 한 머신 안의 이야기다. 그 수치가 플랫폼을 건너서도 같은지는
별개 질문이고, 실제로 갈릴 수 있는 자리가 둘 있다 — `rleToString`이 쓰는 C `long`은
gcc에서 64비트, MSVC에서 32비트고, OKS의 `exp`는 libm 구현마다 다르다.

| | |
|---|---|
| Windows 11 / MSVC / AMD64 | pycocotools는 MSVC 빌드(32비트 `long`) |
| Ubuntu 24.04 / glibc 2.39 / EPYC 9554 | pycocotools는 gcc 빌드(64비트 `long`) |

같은 예측 파일을 양쪽에서 돌려 `precision`/`recall`/`scores` 배열 전체의 digest를
비교했다(`bench/cross_platform.py`). bbox·segm·keypoints 모두, 네 구현 모두
**두 머신에서 동일**했다. 그리고 각 머신에서 따로 계산한 pycocotools 대비 판정도
동일하다 — 우리는 양쪽에서 bit-identical, hotcoco는 양쪽에서 3.7e-05~3.9e-05.

---

**COCO val2017** — 5,000 images / 36,781 GT / 37,504 detections

| iouType | 구현 | eval | speedup | peak RSS | 동치 |
|---|---|---|---|---|---|
| bbox | pycocotools 2.0.11 | 6.494s | 1.0× | 644 MB | (기준) |
| bbox | faster-coco-eval 1.7.2 | 1.839s | 3.5× | 636 MB | bit-identical |
| bbox | hotcoco 0.5.0 | 0.149s | 43.4× | 477 MB | max \|diff\| 1.0e-05 |
| bbox | **ultrafast-pycocotools** | **0.104s** | **62.3×** | **244 MB** | **bit-identical** |
| segm | pycocotools 2.0.11 | 7.484s | 1.0× | 636 MB | (기준) |
| segm | faster-coco-eval 1.7.2 | 3.704s | 2.0× | 685 MB | bit-identical |
| segm | hotcoco 0.5.0 | 0.220s | 34.0× | 482 MB | max \|diff\| 2.0e-06 |
| segm | **ultrafast-pycocotools** | **0.200s** | **37.4×** | **341 MB** | **bit-identical** |

**Objects365 val** — 80,000 images / 1,240,587 GT / 1,170,984 detections / 365 categories

| 구현 | eval | speedup | peak RSS | 동치 |
|---|---|---|---|---|
| pycocotools 2.0.11 | 384.7s | 1.0× | 24.89 GB | (기준) |
| faster-coco-eval 1.7.2 | 157.6s | 2.4× | 28.81 GB | bit-identical |
| hotcoco 0.5.0 | 4.22s | 91.1× | 10.74 GB | max \|diff\| 1.8e-06 |
| **ultrafast-pycocotools** | **2.75s** | **139.8×** | **2.41 GB** | **bit-identical** |

스케일이 커질수록 메모리 차이가 벌어진다. O365에서 **pycocotools의 10분의 1,
hotcoco의 4.5분의 1**이다. pycocotools의 메모리는 대부분 `evalImgs`다 —
(category × areaRange × image)개의 dict를 만들고 각각 `T×D` float64 배열을 담는다.
우리는 그 중간 산출물을 아예 만들지 않고 (category, areaRange) 하나씩 처리하며
accumulate까지 끝낸다. 필요하면 `COCOeval(..., store_eval_imgs=True)`로 pycocotools와
동일한 `evalImgs`를 받을 수 있다.

**남은 2.40 GB의 89%는 Python annotation dict다** (`bench/profile_memory.py` 계측:
GT 867 MB + DT 828 MB + Rust engine 197 MB). `coco.anns[id]`를 dict로 유지하는 설계의
대가이고, detectron2·mmdetection·torchvision이 전부 그 dict를 직접 읽고 수정하기 때문에
Rust view로 바꿀 수는 없다.

그중 되돌릴 수 있는 낭비가 하나 있다. pycocotools의 `loadRes`는 box detection마다
`segmentation`에 네 꼭짓점 polygon을 만들어 넣는다 — detection당 376 B, O365에서
**320 MB**다. `iouType="bbox"`면 아무도 안 읽고, `iouType="segm"`이어도 우리 engine은
box를 직접 rasterise하므로 결과가 같다:

```python
dt = gt.loadRes("detections.json", derive_segmentation=False)   # -320 MB, loadRes -2.1s
```

수치가 안 바뀐다는 건 테스트가 지킨다(`test_derive_segmentation_off_changes_nothing`,
bbox·segm 양쪽에서 pycocotools와 바이트 비교). 기본값은 호환을 위해 `True`다 — 직접
`ann["segmentation"]`을 읽는 코드가 있다면 그대로 두면 된다.

annotation 파일 로딩도 Rust로 한다(`json.load`와 **비트까지 동일한** 결과를 낸다 —
`tests/test_json_loader.py`가 실제 COCO 파일로 확인한다). O365 val 269 MB 기준
4.07s → 2.52s.

## 측정 도구

수치를 못 재면 최적화는 추측이다. 두 harness가 저장소에 들어 있고, 위 표의 모든
숫자가 이걸로 나왔다.

```bash
# 공개 API 표면이 pycocotools와 어긋나는 곳이 있는가
python bench/audit_api.py

# 어느 단계에 시간이 가는가 — engine 내부 phase timer를 직접 읽는다
python bench/profile_engine.py --gt gt.json --dt dt.json --iou-type segm

# 어느 단계가 메모리를 쓰는가 — Rust global allocator를 감싸서 정확히 센다
maturin develop --release --features alloc-stats
python bench/profile_memory.py --gt gt.json --dt dt.json --iou-type segm
```

`profile_engine.py`가 내는 것:

- **extraction** — `read`(GIL 필요)와 `rasterise`(GIL 불필요)는 **동시에 돈다.** 그래서
  합이 안 맞는 게 정상이다. `read_blocked`가 크면 rasteriser가 병목, 0이면 읽기가 병목.
- **evaluation** — phase별 시간이 worker 스레드에 걸쳐 **합산**된다. wall과 비교하면
  그 phase가 실제로 병렬화됐는지가 보인다(`parallel speedup`).

`profile_memory.py`가 내는 것: phase별 RSS delta / Python 할당(`tracemalloc`, 선택) /
Rust live·peak 바이트 / 할당 횟수. `alloc-stats` feature 없이 빌드하면 Rust 열은 0이
아니라 `n/a`로 나온다 — 0으로 보이면 "할당을 안 한다"로 잘못 읽히기 때문이다.

이 harness가 실제로 잡아낸 것들:

| 계측이 지목한 것 | 고친 방법 | 효과 |
|---|---|---|
| annotation dict lookup마다 Python 문자열 생성 | `intern!`으로 key 캐시 | O365에서 1,440만 개 문자열 제거 |
| 읽기와 rasterisation이 번갈아 실행 | worker 스레드로 파이프라인화 | rasterisation이 읽기 뒤로 완전히 숨음 |
| `abi3`에서 `PyFloat_AS_DOUBLE`이 함수 호출 | per-version wheel로 전환 | polygon 읽기 0.098s → 0.067s |
| bbox마다 `Vec<f64>` 힙 할당 | 고정 배열로 직접 읽기 | O365에서 240만 할당 제거 |
| area range마다 match 버퍼 재할당 | category 안에서 버퍼 재사용 | O365 evaluate 할당 2,176만 → 666만 |
| RLE `cnts`의 capacity 여유분 | `shrink_to_fit` | 마스크 메모리 최대 2× → 1× |
| GT rasteriser가 비우는 동안 DT 읽기가 대기 | 양쪽이 worker 하나를 공유 | segm 추출 0.127s → 0.115s |
| category 편중으로 병렬 효율 4.5×/12코어 | category 안에서도 이미지 단위 병렬화 | segm evaluation 0.052s → 0.037s |

---

## 확장 기능

pycocotools API를 전부 지원하면서, AP 하나로는 알 수 없는 것들을 추가로 제공한다.
전부 **같은 매칭 결과에서** 나오므로 옆에 있는 AP와 모순되지 않는다.

```python
ev.run()

ev.stats_as_dict            # {"AP": ..., "AP_50": ..., "AR_small": ...}
ev.per_category_stats()     # class별 AP/AP50/AP75/AR — mAP를 끌어내리는 class 찾기
ev.pr_curve(cat_id=1, iou_thr=0.5)   # {"recall", "precision", "score"}
ev.matches(iou_thr=0.5)     # 매칭된 (dt, gt) 쌍: image_id/category_id/dt_id/gt_id/score/iou
ev.confusion_matrix()       # background row/column 포함
ev.mean_iou()               # 매칭된 쌍의 평균 IoU = localisation 품질
```

`matches()`는 dict 리스트가 아니라 컬럼별 numpy 배열을 준다. COCO 규모에서 매칭이
수십만 건이라 dict로 만들면 평가보다 비싸진다.

평가 격자도 전부 바꿀 수 있고, 바꿔도 pycocotools와 동치가 유지된다(테스트가 그걸
확인한다):

```python
ev.params.areaRng    = [[0, 1e10], [0, 500], [500, 5000], [5000, 1e10]]
ev.params.areaRngLbl = ["all", "tiny", "mid", "big"]
ev.params.maxDets    = [3, 25, 300]
ev.params.iouThrs    = np.linspace(0.3, 0.9, 7)
```

추가 `iouType`:

- `"boundary"` — Boundary IoU (Cheng et al., CVPR 2021). mask IoU와 boundary IoU의
  min을 쓴다. 경계 품질에 민감한 segmentation 비교용. pycocotools에는 없다.

---

## 의도적 차이

수치를 바꾸는 차이는 **없다**. 아래는 전부 동작/부작용 수준이다.

| 차이 | 이유 |
|---|---|
| `evaluate()`가 매칭·누적까지 하고 `accumulate()`는 결과를 게시만 한다 | 그 사이에 K×A×I개 dict를 만드는 것이 pycocotools 메모리의 정체다. API 분리는 유지하되 메모리 프로파일은 따라가지 않는다 |
| annotation dict를 수정하지 않는다 | pycocotools는 `ann['segmentation']`을 RLE로, `ann['ignore']`를 in-place로 덮어쓴다. 남의 데이터를 망가뜨리지 않는 쪽이 맞다 |
| `self.ious` / `_gts` / `_dts`가 처음 접근할 때 만들어진다 | 전부 materialise하면 메모리 이점이 사라진다. 값은 동일하고, 안 쓰면 비용이 0이다 |
| `frPyObjects`가 list-of-boxes와 단일 object 형태를 받는다 | pycocotools는 `len(pyobj[0])`을 float에 호출해서 `TypeError`를 낸다(문서화된 분기가 죽은 코드다). 우리는 문서대로 동작한다 |
| `COCO(..., verbose=False)`로 진행 로그를 끌 수 있다 | pycocotools에는 방법이 없다 |

**`ignore` 필드 주의.** pycocotools의 `_prepare()`는

```python
gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']     # ← 윗줄을 무조건 덮어쓴다
```

라서 annotation의 `ignore` 필드가 **아무 효과가 없다**. CrowdHuman처럼 `ignore: 1`을
쓰는 데이터셋에서 놀라는 지점이다. 동치 보장이 목적이므로 우리도 그대로 재현한다.

---

## 직접 해보기

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools
cd ultrafast-pycocotools
python -m venv .venv && .venv/Scripts/pip install -e ".[test]" maturin
.venv/Scripts/python -m maturin develop --release

# 실제 COCO로 검증하려면 annotation을 놓고 detection을 만든다
.venv/Scripts/python bench/make_dets.py --gt path/to/instances_val2017.json \
    --out bench/data/dt.json --segm

.venv/Scripts/python -m pytest tests/ -q
```

```
61 passed in 5.11s
```

벤치마크:

```bash
.venv/Scripts/python bench/run_impl.py --impl ufcoco \
    --gt bench/data/instances_val2017.json --dt bench/data/dt.json --iou-type bbox
```

---

## 더 볼 것

- [`DESIGN.md`](../DESIGN.md) — 자료구조, 병렬화, SIMD 정책, 등가성을 지키는 규칙
- [pycocotools](https://github.com/cocodataset/cocoapi) — 원본. `common/maskApi.c`와
  `PythonAPI/pycocotools/cocoeval.py`가 우리가 맞춰야 하는 기준이다
- [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) — C++ 재구현,
  LVIS/CrowdPose 확장
- [hotcoco](https://github.com/derekallman/hotcoco) — Rust 재구현. TIDE, calibration,
  OBB, Open Images 등 기능 폭이 넓다

## 한 줄 요약

**AP를 바꾸지 않고 빨라지는 것이 목표라면, 바뀌지 않았다는 것을 요약 수치가 아니라
배열 바이트로 증명해야 한다.** 이 프로젝트는 그걸 계약으로 걸고 테스트로 강제한다.

## License

BSD-2-Clause. `pycocotools`(Piotr Dollár, Tsung-Yi Lin, BSD-2-Clause)의 알고리즘을
이식했다.
