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
| Windows에서 `pip install pycocotools`가 컴파일 에러 | Cython + MSVC 빌드 필요 | prebuilt wheel |
| detection이 tie score를 가질 때 구현마다 AP가 다르다 | greedy matcher가 정렬 순서에 민감한데 stable sort가 아님 | [§동치 보장](#동치-보장) |

**역할별로 어디까지 읽으면 되나**

- 그냥 빠르게 돌리고 싶다 → 위 설치/사용 예제까지면 충분하다.
- 논문 수치를 재현해야 한다 / CI에서 AP를 회귀 테스트한다 → [§동치 보장](#동치-보장)을 읽어야 한다.
- 자체 metric을 붙이거나 오류 분석을 한다 → [§확장 기능](#확장-기능).
- 구현을 고치거나 기여한다 → [`DESIGN.md`](DESIGN.md).

---

## 동치 보장

"거의 같다"가 아니라 **같은 비트**다. 테스트는 12개 요약 수치가 아니라
`eval["precision"]` (COCO 기준 T×R×K×A×M ≈ 80만 개 double) 전체를 `tobytes()`로
비교한다. 요약 수치는 차이를 평균으로 지워버리기 때문이다 — curve가 백 군데 틀려도
0.065로 똑같이 반올림된다.

실제 COCO val2017과 Objects365 val에서 측정한 결과:

| 구현 | COCO bbox | COCO segm | O365 bbox | 비고 |
|---|---|---|---|---|
| **ultrafast-pycocotools** | **bit-identical** | **bit-identical** | **bit-identical** | 전체 배열도 바이트 동일 |
| faster-coco-eval 1.7.2 | bit-identical | bit-identical | bit-identical | |
| hotcoco 0.5.0 | 최대 1.0e-5 | 최대 2.0e-6 | 최대 1.8e-6 | 아래 원인 |

실무적으로 1e-5는 소수점 3자리 리포팅에서 안 보인다. 하지만 "AP가 바뀌지 않는다"를
계약으로 걸려면 요약 수치가 아니라 배열로 증명해야 하고, 그러면 얘기가 달라진다.

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

`pytest tests/` 61개가 전부 실제 pycocotools와 비교한다 (golden 파일이 아니라 live 비교).

- mask API 47개: `encode`/`decode`/`merge`/`area`/`toBbox`/`iou`/`frPyObjects`를
  1×1 이미지, 빈 마스크, 꽉 찬 마스크, 이미지 밖으로 나간 polygon, **꼭짓점이 중복된
  polygon**(`rleFrPoly`가 0으로 나눠 NaN을 int로 캐스팅하는 지점), crowd flag까지.
- 평가 14개: synthetic bbox/segm, 실제 COCO val2017 subset bbox/segm, keypoints,
  `useCats=0`, custom areaRng/maxDets/iouThrs, image/category subset, detection이 하나도
  없는 경우, **모든 score가 동점인 경우**, `evalImgs` 전체.

---

## 속도와 메모리

전부 실측이다. 재현 절차는 [§직접 해보기](#직접-해보기)에 있다. 각 구현은 별도
프로세스에서 돌렸고(peak RSS가 섞이지 않도록), `eval`은
`evaluate + accumulate + summarize` 합계다.

**COCO val2017** — 5,000 images / 36,781 GT / 37,504 detections

| iouType | 구현 | eval | speedup | peak RSS | 동치 |
|---|---|---|---|---|---|
| bbox | pycocotools 2.0.11 | 6.650s | 1.0× | 644 MB | (기준) |
| bbox | faster-coco-eval 1.7.2 | 2.338s | 2.8× | 636 MB | bit-identical |
| bbox | hotcoco 0.5.0 | 0.176s | 37.9× | 479 MB | max \|diff\| 1.0e-05 |
| bbox | **ultrafast-pycocotools** | **0.160s** | **41.4×** | **244 MB** | **bit-identical** |
| segm | pycocotools 2.0.11 | 9.413s | 1.0× | 636 MB | (기준) |
| segm | faster-coco-eval 1.7.2 | 4.142s | 2.3× | 686 MB | bit-identical |
| segm | hotcoco 0.5.0 | **0.213s** | **44.3×** | 481 MB | max \|diff\| 2.0e-06 |
| segm | **ultrafast-pycocotools** | 0.302s | 31.1× | **346 MB** | **bit-identical** |

**Objects365 val** — 80,000 images / 1,240,587 GT / 1,170,984 detections / 365 categories

| 구현 | eval | speedup | wall | peak RSS | 동치 |
|---|---|---|---|---|---|
| pycocotools 2.0.11 | 384.7s | 1.0× | 392.9s | 24.89 GB | (기준) |
| faster-coco-eval 1.7.2 | 157.6s | 2.4× | 166.6s | 28.81 GB | bit-identical |
| hotcoco 0.5.0 | 4.21s | 91.4× | 8.35s | 10.74 GB | max \|diff\| 1.8e-06 |
| **ultrafast-pycocotools** | **3.66s** | **105.2×** | 10.18s | **2.40 GB** | **bit-identical** |

스케일이 커질수록 메모리 차이가 벌어진다. **pycocotools의 10분의 1, hotcoco의 4분의 1**이다.
pycocotools의 메모리는 대부분 `evalImgs`다 — (category × areaRange × image)개의 dict를
만들고 각각 `T×D` float64 배열을 담는다. 우리는 그 중간 산출물을 아예 만들지 않고
(category, areaRange) 하나씩 처리하며 accumulate까지 끝낸다. 필요하면
`COCOeval(..., store_eval_imgs=True)`로 pycocotools와 동일한 `evalImgs`를 받을 수 있다.

**segm에서 hotcoco보다 느린 이유는 숨기지 않겠다.** 우리는 annotation을 Python dict에서
읽어 Rust로 넘기는데, segm에서는 그 추출 + polygon rasterisation이 engine 시간의 81%를
차지한다(0.217s / 0.269s). `COCO`가 Python dict를 그대로 유지하는 설계의 대가다
([§의도적 차이](#의도적-차이) 참조). detectron2·mmdetection·torchvision이 전부
`coco.anns[id]`를 직접 읽고 수정하기 때문에 그 dict를 Rust view로 바꿀 수는 없다.

annotation 파일 로딩도 Rust로 한다(`json.load`와 **비트까지 동일한** 결과를 낸다 —
`tests/test_json_loader.py`가 실제 COCO 파일로 확인한다). O365 val 269MB 기준
4.07s → 2.52s.

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
| `self.ious`가 기본으로 채워지지 않는다 | 전부 materialise하면 메모리 이점이 사라진다 |
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

- [`DESIGN.md`](DESIGN.md) — 자료구조, 병렬화, SIMD 정책, 등가성을 지키는 규칙
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
