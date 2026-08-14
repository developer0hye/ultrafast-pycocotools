# 설계 노트

이 문서는 구현을 고치거나 기여할 사람을 위한 것이다. "무엇을 하는가"는
[`README.md`](README.md)에 있고, 여기는 **왜 그렇게 되어 있는가**만 적는다. 특히
"이렇게 하면 더 깔끔한데 왜 안 그랬지" 싶은 자리마다 이유가 있고, 대부분 등가성이
그 이유다.

---

## 왜 알아야 하나

| 하려는 일 | 먼저 읽을 것 |
|---|---|
| 산술을 건드리는 최적화(벡터화, 재배치, FMA)를 넣고 싶다 | [§등가성 규칙](#등가성-규칙-3개) |
| 정렬을 바꾸거나 `sort_unstable`로 교체하고 싶다 | [§규칙 2](#규칙-2-모든-정렬은-stable) |
| threshold를 Rust 쪽에서 생성하도록 리팩터링하고 싶다 | [§규칙 3](#규칙-3-격자는-caller가-만든다) |
| 대형 데이터셋에서 메모리가 안 잡힌다 | [§자료구조](#자료구조-dense-테이블은-o365에서-죽는다) |
| 평가가 왜 이 순서로 도는지 모르겠다 | [§루프 구조](#루프-구조-evaluateaccumulate-융합) |
| 새 metric을 추가하고 싶다 | [§확장 지점](#확장-지점) |

---

## 등가성 규칙 3개

이 프로젝트의 가치는 속도가 아니라 **속도 × 동치**다. 동치를 잃으면 남는 게 없으므로
아래 셋은 협상 대상이 아니다. 셋 다 실제로 다른 구현이 어긴 지점이고, 어긴 결과를
측정했다.

### 규칙 1: 부동소수점 재배치 금지

Rust는 기본적으로 fast-math 플래그를 켜지 않는다. LLVM이 마음대로 reassociate 하거나
FMA로 fuse 하지 않는다는 뜻이고, 그래서 `Cargo.toml`의 release profile에는
`opt-level` / `lto` / `codegen-units`만 있다 — 전부 값을 보존하는 옵션이다.

허용되는 것과 아닌 것:

| 종류 | 예 | 판정 |
|---|---|---|
| 정수 SIMD | RLE run 카운팅, decode | ✅ 정의상 exact |
| element-wise FP SIMD | `bb_iou`의 (d,g) 쌍별 계산 | ✅ 스칼라와 연산 순서 동일 |
| exact-op reduction | precision의 backward `max` | ✅ `max`는 rounding이 없다 |
| horizontal FP sum | OKS의 `sum(exp(-e))` | ❌ 순서가 바뀌면 결과가 바뀐다 |
| `f64::mul_add` | 어디든 | ❌ 중간 rounding이 사라진다 |

구체적으로 지켜야 하는 것:

- `pr = tp / (fp + tp + EPS)`에서 `EPS`(= `np.spacing(1)` = `f64::EPSILON`)를 빼지 마라.
  `fp + tp == 1`인 지점 — 모든 curve의 첫 점 — 에서 1 ULP 어긋난다.
  faster-coco-eval의 C++ 경로와 hotcoco 둘 다 이 항이 없다.
- OKS의 `(dx*dx + dy*dy) / vars[t] / (area + EPS) / 2.0`은 나눗셈을 하나씩 한다.
  곱해서 한 번에 나누면 다른 값이 나온다.
- `rle_iou`는 bbox IoU가 **strictly positive**인 쌍에서만 정확한 run 교집합을 계산한다.
  이건 최적화가 아니라 의미다 — bbox가 어긋난 쌍은 마스크를 보지 않고 정확히 `0.0`이
  되어야 한다.

### 규칙 2: 모든 정렬은 stable

pycocotools는 `kind='mergesort'`를 쓴다. score가 같은 detection의 순서가 annotation
순서로 고정된다는 뜻이고, greedy matcher는 순서에 민감하다 — 먼저 온 detection이 GT를
가져가고, 그 결과가 뒤 detection의 운명을 바꾼다.

그래서:

- `Vec::sort_by` / `sort_by_key`만 쓴다. `sort_unstable_*`는 값이 전부 서로 다른
  `rle_fr_poly`의 `a.sort_unstable()` 한 곳에만 있다(같은 값끼리는 구분이 불가능하므로
  안전하다).
- `cmp_desc_score`는 `-score` 오름차순 정렬과 같은 순열을 만든다. `-0.0`과 `0.0`이
  같다고 비교되는 것까지 numpy와 일치하고, NaN은 마지막으로 간다.
- 정렬의 입력 순서도 계약이다. `_prepare()`가 `loadAnns(getAnnIds(...))` 결과를 그대로
  넘기고, `group.rs`가 그 순서를 보존한다. 여기서 정렬·중복제거·필터를 추가하면
  "annotation 순서"의 정의가 달라진다.

### 규칙 3: 격자는 caller가 만든다

`iouThrs` / `recThrs`는 Python에서 `np.linspace`로 만들어 engine에 넘긴다. Rust 쪽에서
다시 만들지 않는다.

`np.linspace(0.5, 0.95, 10)`은 `[0.5 + 0.05*i]`와 다르다(10개 중 2개가 1 ULP),
`np.linspace(0, 1, 101)`은 101개 중 10개가 다르다. accumulate의
`searchsorted(rc, recThrs, side='left')`가 strict 비교라서, recall이 threshold에 정확히
떨어지는 순간(recall은 `tp/npig`이라 흔하다) 다른 precision을 샘플링한다. hotcoco에서
측정한 COCO AP 차이 1.7e-6의 원인이 이것이다.

---

## 자료구조: dense 테이블은 O365에서 죽는다

pycocotools는 사실상 (image × category) dense 테이블을 만든다. COCO(5k × 80)에서는
문제없다. Objects365 val은 **80,000 × 365 = 2,920만 칸**이고, 빈 `Vec` 헤더만 24바이트씩
잡아도 annotation을 하나도 담기 전에 **700MB**다.

그래서 `group.rs`는 존재하는 쌍만 저장한다. annotation index를 (group, image)로 한 번
정렬해 CSR 형태로 만든다:

```
order:      annotation index, (group, image, category) 로 stable 정렬
runs:       (image_slot, start, len)  — 한 (group, image)의 연속 구간
group_runs: group마다 runs의 [start, end)
```

메모리는 O(annotation), 조회는 O(1)이다. 그리고 `RunJoin`이 gt/dt의 run을 image 순서로
merge-walk 해서 **둘 다 비어 있는 image는 아예 방문하지 않는다**. pycocotools는 모든
image를 돌면서 `None`을 만들지만, 그 `None`은 어떤 metric에도 기여하지 않으므로
건너뛰는 것이 관측상 동일하다.

## 루프 구조: evaluate/accumulate 융합

pycocotools는 (1) 모든 IoU 계산 → (2) K×A×I개 결과 dict 생성 → (3) accumulate 순서다.
(2)가 메모리의 정체다.

우리는 바깥 루프가 **category**이고, 그 안에서 IoU → 매칭 → accumulate까지 끝낸다:

```
for k in categories (rayon 병렬):
    work = prepare_category(k)            # 이 category의 image별 IoU 행렬
    for a in areaRanges:
        matches = work.map(evaluate_img)  # 이 (k, a)의 매칭
        for m in maxDets:
            accumulate_slice(...)         # 바로 누적하고 버린다
```

살아있는 메모리는 category 하나 분량이다. IoU 행렬을 area range마다 다시 계산하지 않고
재사용하므로 연산량은 pycocotools와 같다.

대가: `evalImgs`가 기본으로 존재하지 않는다. `store_eval_imgs=True`면
`materialise_eval_imgs()`가 pycocotools와 같은 dict를 만든다(없는 image 자리에 `None`을
다시 넣어 positional index까지 맞춘다).

## 병렬화

rayon으로 category 단위 fan-out. 각 작업이 자기 출력 슬롯에만 쓰므로 결정적이다 —
스레드 수가 결과를 바꾸지 않는다. 이건 테스트로 확인해야 할 성질이고,
`accumulate_slice`가 category별 버퍼에만 쓰고 마지막에 한 번 모아 붙이는 이유다.

annotation 추출은 GIL을 잡아야 하므로 단일 스레드다. 대신 chunk(16384개)마다 raw
geometry를 모아 `py.detach()`로 GIL을 놓고 rasterise 한다. peak 메모리가 "전체
polygon + 전체 RLE"가 아니라 "chunk 하나 + 완성된 RLE"가 된다.

## 기각: 안 쓰이는 마스크 건너뛰기

IoU는 (image, category) 셀 안에서만 계산된다. 한쪽이 비어 있는 셀의 마스크는 아무와도
비교되지 않는다 — 매칭 상대가 없는 GT는 픽셀을 몰라도 FN 하나로 세어지고, 그 image에
없는 category의 detection은 볼 것도 없이 FP다. 그런데 pycocotools도 우리도 **모든**
annotation을 RLE로 굽는다. YOLO11m-seg + COCO val2017에서 467,787개 중 **119,564개
(25.6%)** 가 그렇게 구워놓고 안 쓰인다.

건너뛰어도 결과는 그대로다 — AP에도, `evalImgs`에도, `matches`에도 안 들어간다.
그래도 안 했다. 이득이 **전체의 1.8%** 라서다.

`bench/probe_unused_masks.py`가 상한을 잰다. 완벽한 구현이 건너뛸 annotation의
segmentation을 1×1 빈 마스크로 바꿔 engine build를 측정한다:

```
                              build   gt_read  dt_read  rasterise  read_blocked
segm, as-is                   0.288     0.055    0.200      0.182         0.026
segm, unused masks stripped   0.257     0.062    0.167      0.144         0.014
bbox (no masks at all)        0.070     0.011    0.042      0.000         0.000
```

0.030s. 단일 스레드 segm 평가 전체가 1.655s이므로 **1.8%** 이고, 이건 상한이다.
실제 구현은 어느 셀이 비었는지 알아야 건너뛸 수 있고 그건 grouping을 먼저 만들어야
한다는 뜻이라 — annotation을 두 번 걸어야 한다. 그 비용을 빼면 더 내려간다.

25.6%를 지웠는데 왜 1.8%뿐인가: rasterise는 이미 읽기 뒤에 숨어 있다(`read_blocked`
0.026s). 남는 건 "안 쓰일 segmentation을 Python에서 안 읽는 것"뿐이고, 그건 dt_read의
일부다.

### 이 측정에서 두 번 틀렸다 — 둘 다 실험 설계였다

같은 실수를 반복하지 않도록 적어둔다.

1. **stripped arm만 dict를 복사했다.** 일을 25% 줄였는데 **70% 느리게** 나왔다.
   새로 할당한 dict 12만 개는 json이 남긴 것들과 다른 곳에 있고, 추출 루프가 cache
   miss를 전부 물었다. 양쪽 arm이 똑같이 복사하도록 고쳤다.
2. **`segmentation` 키를 지우는 것이 "건너뛰기"가 아니었다.** 키가 없으면 reader가
   bbox로 fallback 하는데, **박스의 RLE가 마스크의 RLE보다 비싸다.** COCO RLE은
   column-major라 폭 400짜리 직사각형은 열마다 run이 생겨 ~800 run이 되고, 뭉쳐 있는
   blob의 압축 문자열은 그보다 훨씬 짧게 decode 된다. rasterise가 0.175 → 0.350으로
   두 배가 됐다. 1×1 빈 마스크로 바꾸니 그제야 읽기 경로와 variant를 유지한 채 기하만
   사라졌다.

"일을 줄였는데 느려졌다"가 나오면 아이디어가 틀린 게 아니라 **측정이 틀렸을 가능성을
먼저** 본다. 두 번 다 그랬다.

## SIMD 정책

넣되, 규칙 1을 어기지 않는 곳에만. 실측 기준 우선순위:

1. ~~**JSON 파싱**~~ — **기각.** 여기가 wall time의 최대 항목인 건 맞지만
   (`bench/profile_loadres.py`: segm 결과 파일에서 51.7%), **파서가 아니라 Python
   객체를 만드는 게 82%다.** `bench/profile_json.py`가 read / parse / build로 쪼갠다:

   | 파일 | read | parse(버림) | + PyObject build | stdlib |
   |---|---|---|---|---|
   | 163 MB DT | 0.050s | +0.087s (11%) | +0.647s (**82%**) | 1.098s |
   | 269 MB O365 | 0.086s | +0.238s (14%) | +1.420s (**81%**) | 2.946s |

   토크나이저가 **무한히** 빨라져도 상한이 14%다. 남은 82%는 annotation마다
   `PyDict` 하나 + 숫자 객체 열 개를 만드는 비용이고, `coco.anns[id]`가 진짜 dict여야
   detectron2·mmdetection이 그걸 읽고 고칠 수 있으므로 없앨 수 없다. 이미 stdlib의
   1.3~1.7배다.
2. **`bb_iou` 내부 루프** — element-wise라 안전. crowded image / `useCats=0`에서 유효.
3. ~~**accumulate의 정렬**~~ — **기각.** 정렬이 이 단계의 비용이 아니다. 먼저
   comparator에서 두 번 포인터를 쫓던 것(`matches[i].dt_scores[d]`)을 없애고 score를
   entry에 인라인으로 실었는데, RF-DETR bbox(150만 detection, accumulate가 최대 항목)에서
   1.490s → 1.447s로 **노이즈 범주**였다. 그 다음 slice 수를 바꿔 스케일링을 봤더니
   maxDets 3개가 1개보다 **0.017s** 더 들 뿐이었다 — 재계산도 비용이 아니다.
   radix sort로 바꿔봐야 가져올 게 없다. (인라인 score는 남겼다. 빨라져서가 아니라
   `scores_sorted` 버퍼와 그 위를 한 번 더 도는 pass가 통째로 없어져서다.)
4. **`rle_iou`의 run 병합** — 데이터 의존 분기라 이득이 거의 없다. 스칼라 유지.
   대신 **아예 안 하는** 쪽으로 줄였다(아래).

SIMD 커널을 넣을 때는 **반드시 스칼라 레퍼런스와 bit-identical 비교하는 테스트를
같이** 넣어야 한다. 규칙을 사람이 기억하게 두면 안 된다.

## IoU를 정확히 구하지 않아도 되는 pair

`rle_iou`는 bbox prefilter를 통과한 pair마다 두 run 리스트를 끝까지 병합한다. 그런데
매처가 그 값으로 하는 일은 threshold와 비교하는 것뿐이고, 가장 낮은 threshold는 보통
0.5다. 실측: **prefilter 생존 pair의 77%가 결국 0.5 미만**이다. 정확히 구해서 "아니오"만
듣는다.

마스크는 자기 tight box 안에 있으므로 마스크 교집합이 box 교집합을 넘을 수 없다:

```
    i <= m = min(area_dt, area_gt, box_inter)
    IoU = i / (a + b - i)   는 i에 대해 증가함수
    => IoU <= m / (a + b - m)          (crowd면 m / area_dt)
```

이 bound가 이미 threshold 아래면 병합은 어떤 판정도 바꿀 수 없다. `0.0`을 쓰고 넘어간다.
COCO val2017 + YOLO11m-seg에서 **pair의 63.5%** 에 걸리고(버려지는 일의 82%),
**bound가 진짜 IoU를 넘은 적은 0번**이다 — 넘을 수 없다, 위가 증명이다.

`min_thr <= 0`이면 끈다. **이건 성능 스위치가 아니다.** threshold가 0이면 매처는 낮은
값을 건너뛰는 게 아니라 **순위를 매기기** 시작한다(threshold를 넘는 것 중 IoU 최대인 GT를
고른다). 그때 진짜 0.37 대신 0.0을 쓰면 어느 GT가 이기는지가 바뀐다.

측정(단일 스레드, segm):

| | 전 | 후 |
|---|---|---|
| `iou` phase (worker 합) | 0.637s | 0.591s |
| eval_total, YOLO11m-seg | 1.655s | 1.512s |
| eval_total, Mask R-CNN | 1.047s | 0.977s |

**7%만 줄었다.** run 병합이 이 단계의 주된 비용이 아니었기 때문이다 — 병합을 통째로
빼고 재보니 `iou`가 0.388s였다. 남은 0.388s는 467k개 RLE의 `to_bbox`(+`area`, `bb_iou`,
할당)이고, 그중 나눗셈은 0.104s다(`area`와 `toBbox`를 같은 입력으로 나란히 재서 뺐다).
즉 **`iou`는 이제 바닥에 가깝다.** 더 줄이려면 `to_bbox`/`area`를 rasterise 시점에
계산해 캐시해야 하는데, 지금 rasteriser는 단일 워커라 그리로 옮기면 직렬 구간이 늘어
병렬 구간이 줄어든 만큼을 까먹는다. **rasteriser를 rayon으로 병렬화하는 것이 선행 조건**이고,
그게 다음 후보다.

## 무게중심은 이제 평가가 아니라 로딩이다

기본 스레드로 COCO val2017 + YOLO11m-seg 전체를 재면:

| | pycocotools | ufcoco |
|---|---|---|
| load (GT + DT) | 1.845s (7%) | 1.880s (**75%**) |
| evaluate + accumulate + summarize | 24.116s (93%) | 0.574s (25%) |
| **total** | **25.961s** | **2.454s** |

평가를 **42배** 빠르게 만들고 나니, 사용자가 기다리는 시간의 3/4이 로딩이다. 그리고 그
로딩의 82%는 파서가 아니라 annotation마다 `PyDict`를 만드는 비용이다(위 SIMD §1).
**즉 남은 큰 덩어리는 알고리즘이 아니라 drop-in 계약 자체다** — `coco.anns[id]`가 진짜
dict여야 한다는 것. 그걸 지키는 한 여기가 바닥이고, 깨면 이 프로젝트의 존재 이유가
없어진다.

한 가지 남은 것: `loadRes`에 **경로를 넘기면** 우리 Rust 로더를 타고(163 MB에서
0.789s vs stdlib 1.098s), 이미 파싱된 리스트를 넘기면 못 탄다. 대부분의 평가 하네스는
후자다.

## 확장 지점

새 metric은 `Evaluator::matches()`에서 시작하는 것이 맞다. 매칭 결과를 그대로 쓰므로
옆에 있는 AP와 모순되지 않는다. AP와 다른 매칭을 쓰는 confusion matrix는 없느니만
못하다.

`MatchRecord`는 (image, category, dt_id, gt_id, score, iou)를 담는다. TP/FP/FN 목록,
per-image 진단, mean IoU, calibration이 전부 여기서 나온다.

`GeomStore`에 variant를 추가하면 새 IoU 종류를 붙일 수 있다(`Boundaries`가 그 예다).
`compute_iou`의 match arm 하나와 추출 경로 하나면 된다.

## 측정 방법

**추측으로 최적화하지 않는다.** 이 저장소의 최적화는 전부 계측이 먼저 지목한 것이고,
계측 도구가 코드와 같이 들어 있다.

```bash
python bench/make_dets.py --gt <gt.json> --out bench/data/dt.json [--segm]

# 비교 (구현마다 별도 프로세스, 3회 돌려 best)
python bench/run_impl.py --impl {pycocotools,faster,hotcoco,ufcoco} \
    --gt <gt.json> --dt bench/data/dt.json --iou-type {bbox,segm} --json-out out.json
python bench/summarize.py bench/out/*.json

# 어디에 시간이 가는가 (engine 내부 timer를 직접 읽음)
python bench/profile_engine.py --gt <gt.json> --dt bench/data/dt.json --iou-type segm

# 어디에 메모리가 가는가 (Rust global allocator 계측)
maturin develop --release --features alloc-stats
python bench/profile_memory.py --gt <gt.json> --dt bench/data/dt.json --iou-type segm

python bench/check_params.py <gt.json> {hotcoco,faster,ufcoco}   # 격자 bit 비교
```

측정할 때 지킬 것:

- **단일 스레드 수치를 같이 낸다** (`bench/compare.py --threads 1`). pycocotools는
  단일 스레드라 배경 부하를 거의 안 타는데 rayon을 쓰는 구현은 코어를 두고 경쟁하다
  자기 효율과 무관한 wall을 잃는다. 부하 37%인 데스크톱에서 잰 병렬 speedup은 그
  머신의 유휴 코어 수를 재는 것에 가깝다. 단일 스레드 수치가 머신을 건너서도 통한다.
- **CPU 시간을 wall 옆에 같이 본다.** 경합에서 wall은 부풀고 CPU는 안 부푼다. 둘의
  비가 어긋나면 측정이 방해받았다는 신호다.
- **판정은 요약이 아니라 배열로 한다.** 이건 남 얘기가 아니다 — 우리는 12개 stat만
  보고 faster-coco-eval을 "bit-identical"이라 여러 번 보고했다가, 배열 digest를
  붙이고 나서야 5,744개 cell이 1 ULP씩 다르다는 걸 알았다. 969,600개로 평균 내면
  같은 double로 반올림된다. `bench/run_impl.py`가 배열 digest를 함께 내는 이유다.

- **구현마다 별도 프로세스.** 같은 프로세스에서 재면 다른 라이브러리의 warm cache와
  allocator 상태가 섞이고 peak RSS가 의미를 잃는다.
- **최소 3회 돌려 best를 쓴다.** 0.1~0.3초대 측정의 분산이 15%다. 한 번만 재고
  "빨라졌다/느려졌다"를 판정하면 노이즈를 쫓게 된다 — 실제로 이 저장소의 버퍼 재사용
  최적화는 단발 측정에서 13% 느려 보였고, 5회 측정에서 노이즈였음이 드러났다.
- **차분 추정과 직접 계측을 구분한다.** `profile_segm.py`는 입력을 바꿔가며 전체
  시간 차이로 비용을 역산한다 — 첫 감을 잡기엔 좋지만 engine 내부를 못 보고
  오버랩도 못 본다. `profile_engine.py`는 engine이 스스로 잰 값을 읽으므로 귀속이
  확실하다.

### 두 종류의 숫자를 섞어 읽지 않기

`profile_engine.py`의 extraction과 evaluation은 의미가 다르다.

- **extraction**은 서로 겹쳐 도는 두 스레드의 wall-clock이다. annotation을 Python에서
  읽으려면 GIL이 필요하고 rasterise에는 필요 없으므로, 둘을 번갈아 하지 않고 동시에
  한다. 그래서 `read`와 `rasterise`는 **합해지지 않는다.** `read_blocked`가 둘 중
  누가 병목인지를 말해준다.
- **evaluation**은 worker 스레드에 걸쳐 **합산된 CPU 시간**이라 wall을 넘는다.
  일부러 그렇게 둔다 — 합은 CPU가 어디로 갔는지를, wall과의 비(`parallel speedup`)는
  그 phase가 실제로 병렬화됐는지를 말한다.

`profile_memory.py`의 RSS delta는 allocator 여유분과 단편화를 포함하므로 Rust/Python
수치보다 크고 서로 합해지지 않는다. 사용자가 OOM으로 맞는 숫자는 RSS 쪽이다.

`alloc-stats` feature가 없으면 Rust 열은 0이 아니라 `n/a`로 나온다. 0으로 보이면
"할당을 안 한다"로 잘못 읽히는데, 그건 정반대의 결론이다.

`bench/make_dataset.py`는 씨앗 고정 synthetic 데이터를 만든다. 일부러 어려운 입력을
넣는다 — crowd, small/medium/large 경계에 정확히 걸치는 area, 동점 score, GT만 있는
image, detection만 있는 image, 꼭짓점이 중복된 polygon.

## 계측이 지목했고 실제로 고친 것

| 계측이 보여준 것 | 원인 | 고친 방법 |
|---|---|---|
| dict lookup이 extraction의 큰 몫 | `get_item("image_id")`가 매번 Python 문자열 생성 | `intern!`로 key 캐시. O365에서 1,440만 개 제거 |
| segm setup의 81%가 extract+rasterise | 읽기와 rasterise를 번갈아 실행 | worker 스레드 + bounded channel로 파이프라인화. rasterise가 읽기 뒤로 완전히 숨음(`read_blocked` 0.000s) |
| polygon vertex 읽기 92 ns | `abi3`에서 `PyFloat_AS_DOUBLE`이 매크로가 아니라 함수 호출 | abi3 포기, per-version wheel. 0.098s → 0.067s |
| bbox마다 힙 할당 | `Vec<f64>::extract` | 고정 배열로 직접 읽기. O365에서 240만 할당 제거 |
| evaluate에서 2,176만 할당 | area range마다 match 버퍼 재할당 | category 안에서 재사용. 666만으로 감소, 해당 phase 2.22s → 0.83s |
| 마스크 메모리가 예상의 2배 | `Vec::push`로 만든 `cnts`의 capacity 여유분 | `shrink_to_fit` |
| O365 `loadRes` 1,148 MB | box detection마다 네 꼭짓점 polygon 생성 | engine이 box에서 직접 rasterise. `derive_segmentation=False`로 320 MB·2.1s 절약 |

**abi3를 포기한 것은 패키징 결정이다.** Python 버전마다 wheel을 만들어야 한다. 대신
limited API에서는 `PyFloat_AS_DOUBLE`, `PyList_GET_ITEM`이 매크로가 아니라 함수 호출이
되는데, annotation을 읽는 것이 이 라이브러리의 가장 뜨거운 루프라 그 비용이 그대로
드러난다. pycocotools 자체도 Cython 확장이라 per-version wheel을 배포하므로 사용자가
겪는 차이는 없다.

부수 효과: `cargo build -p ufcoco-py`가 Python 인터프리터를 찾아야 한다. venv를
활성화했거나 `PYO3_PYTHON`이 설정돼 있으면 된다. 알고리즘과 테스트가 있는
`ufcoco-core`는 PyO3 의존이 없으므로 `cargo test -p ufcoco-core`는 항상 그냥 돈다.

## 한 줄 요약

**빠르게 만드는 방법은 여러 개지만, 빨라지면서 값이 안 변하는 방법은 몇 개 없다.**
이 코드에서 이상해 보이는 선택은 대부분 그 몇 개 중 하나를 고른 흔적이다.
