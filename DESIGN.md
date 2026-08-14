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

## SIMD 정책

넣되, 규칙 1을 어기지 않는 곳에만. 실측 기준 우선순위:

1. **JSON 파싱** — O365 val에서 GT 4.07s + DT 4.57s로 wall time의 최대 항목이다.
   문자열·정수 처리라 등가성 위험이 0이다. (미구현 — 가장 큰 남은 작업)
2. **`bb_iou` 내부 루프** — element-wise라 안전. crowded image / `useCats=0`에서 유효.
3. **accumulate의 정렬** — O365 규모에서는 SIMD보다 stable LSD radix sort가 크다.
   f64 score를 order-preserving u64로 매핑하면 stable하게 정렬된다.
4. **`rle_iou`의 run 병합** — 데이터 의존 분기라 이득이 거의 없다. 스칼라 유지.

SIMD 커널을 넣을 때는 **반드시 스칼라 레퍼런스와 bit-identical 비교하는 테스트를
같이** 넣어야 한다. 규칙을 사람이 기억하게 두면 안 된다.

## 확장 지점

새 metric은 `Evaluator::matches()`에서 시작하는 것이 맞다. 매칭 결과를 그대로 쓰므로
옆에 있는 AP와 모순되지 않는다. AP와 다른 매칭을 쓰는 confusion matrix는 없느니만
못하다.

`MatchRecord`는 (image, category, dt_id, gt_id, score, iou)를 담는다. TP/FP/FN 목록,
per-image 진단, mean IoU, calibration이 전부 여기서 나온다.

`GeomStore`에 variant를 추가하면 새 IoU 종류를 붙일 수 있다(`Boundaries`가 그 예다).
`compute_iou`의 match arm 하나와 추출 경로 하나면 된다.

## 측정 방법

수치는 전부 실측이고, 재현 절차는 다음과 같다.

```bash
python bench/make_dets.py --gt <gt.json> --out bench/data/dt.json [--segm]
python bench/run_impl.py --impl {pycocotools,faster,hotcoco,ufcoco} \
    --gt <gt.json> --dt bench/data/dt.json --iou-type {bbox,segm}
python bench/profile_phases.py --gt <gt.json> --dt bench/data/dt.json --iou-type segm
python bench/check_params.py <gt.json> {hotcoco,faster,ufcoco}   # 격자 bit 비교
```

각 구현은 **별도 프로세스**에서 돈다. 같은 프로세스에서 재면 다른 라이브러리의 warm
cache와 allocator 상태가 섞이고 peak RSS가 의미를 잃는다.

`bench/make_dataset.py`는 씨앗 고정 synthetic 데이터를 만든다. 일부러 어려운 입력을
넣는다 — crowd, small/medium/large 경계에 정확히 걸치는 area, 동점 score, GT만 있는
image, detection만 있는 image, 꼭짓점이 중복된 polygon.

## 한 줄 요약

**빠르게 만드는 방법은 여러 개지만, 빨라지면서 값이 안 변하는 방법은 몇 개 없다.**
이 코드에서 이상해 보이는 선택은 대부분 그 몇 개 중 하나를 고른 흔적이다.
