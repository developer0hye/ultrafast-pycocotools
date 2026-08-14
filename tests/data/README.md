# 테스트 fixture

## `coco_subset_gt.json` / `coco_subset_dt.json`

COCO val2017 annotation에서 잘라낸 93개 이미지(785 annotation)와, 거기서 유도한
780개 detection.

**왜 커밋되어 있나.** 합성 데이터는 실제 annotation을 근사할 뿐이고 일부는 틀리게
근사한다 — `bench/make_dataset.py`가 만드는 polygon은 3~8개 꼭짓점의 단일 ring인데,
실제 COCO의 것은 수십 개 꼭짓점에 ring이 여러 개다. 실제 crowd 영역은 임의 형태의
uncompressed RLE지만 합성 쪽은 축 정렬 사각형이다. `rleFrPoly`가 아무도 만들어볼
생각을 못 한 형태에서 어긋나면 잡을 수 있는 건 실제 데이터뿐이다.

원래는 `bench/data/`의 20 MB 파일을 쓰다 보니 그 파일이 있는 한 대의 머신에서만
돌고 fresh clone과 CI에서는 **조용히 skip**됐다. 한 곳에서만 도는 테스트는 나중에
알게 되는 테스트다.

**구성** (`tests/test_fixture_coverage.py`가 강제한다):

| | |
|---|---|
| crowd annotation | 32 (uncompressed RLE 포함) |
| multi-ring polygon | 145 |
| 가장 긴 polygon | 40+ 꼭짓점 |
| area scale | small 440 / large 130 |
| annotation 없는 이미지 | 3 |
| annotation 없는 category | 있음 (`-1` sentinel 경로) |

**판별력.** 969,600개 precision 셀 중 **한 셀의 1 ULP** 차이도 parity 비교가 잡는다.

**재생성**:

```bash
python bench/make_real_fixture.py --gt path/to/instances_val2017.json
```

선택은 무작위가 아니라 "바이트당 커버리지" 탐욕 선택이고, crowd 이미지에는 별도
할당을 준다 — 순수 효율 순위로는 crowd가 밀려난다(밀도가 높아 비싸다). 처음 뽑았을
때 crowd가 2개뿐이었다.

## 출처와 라이선스

COCO annotation은 [cocodataset.org](https://cocodataset.org) 것이며
**CC BY 4.0**이다. 이미지 파일은 포함하지 않는다 — annotation만 있으면 평가
테스트에는 충분하다.
