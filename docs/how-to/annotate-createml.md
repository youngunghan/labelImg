# How-to: CreateML(JSON) 라벨링

CreateML 포맷은 Apple CreateML 객체 검출 학습용이다. JSON 최상위가 **이미지 객체들의 리스트**라는 점이 특징이다.

> ⚠️ **기본 저장 단위**: 앱의 일반 저장(`Ctrl+S`)은 다른 포맷과 똑같이 **이미지마다 `<이미지이름>.json`** 을 만든다(`save_file`→`save_labels`가 이미지 stem에 `.json`을 붙임, `labelImg.py:943-947`). 따라서 기본적으로는 각 JSON이 **이미지 1개짜리 리스트**다. `CreateMLWriter` 자체는 같은 출력 파일을 재사용하면 여러 이미지를 한 파일에 누적(같은 `image`는 교체, 없으면 append)할 수 있지만, 그건 같은 `output_file`을 반복 지정할 때만이고 기본 `Ctrl+S` 경로는 그렇게 하지 않는다.

## 절차

1. 앱을 실행한다.
2. 툴바 포맷 버튼을 눌러 **CreateML**로 바꾼다(VOC→YOLO→CreateML 순환).
3. `Open`/`Open Dir`로 이미지를 열고 박스를 그린다.
4. `Ctrl+s`로 저장한다.

## 출력

기본 저장 결과 — `img001.json`(이미지 1개짜리 리스트):

```json
[
  {
    "image": "img001.jpg",
    "verified": false,
    "annotations": [
      { "label": "face", "coordinates": { "x": 297.5, "y": 307.5, "width": 105, "height": 115 } }
    ]
  }
]
```

- `coordinates.x`/`y`는 박스 **중심**(절대 픽셀), `width`/`height`는 크기.
- 같은 출력 파일을 재사용하면 리스트에 여러 이미지가 쌓이고, 같은 `image`는 항목이 **통째로 교체**된다(기존 annotations는 사라짐, 병합 아님). 기본 `Ctrl+S`는 이미지별 파일이라 보통 1개 항목만 들어간다.
- 파일은 `indent` 없이 한 줄로 직렬화된다(사람이 읽기엔 불편).

## 주의사항

- **로드 시 모든 박스가 `difficult=True`로 들어온다**(`libs/create_ml_io.py:132`) — 코드상 하드코딩이라 다른 포맷과 동작이 다르다. 학습 전처리에서 difficult를 거른다면 유의.
- Reader는 파일의 **첫 항목** `verified`만 읽어 전체 verified로 삼는다.
- Reader는 JSON 안에서 `image` 값이 현재 이미지 **basename과 정확히 일치**하는 항목만 로드한다(`libs/create_ml_io.py:120`). 이미지 파일명을 바꾸면 기존 JSON의 박스가 로드되지 않는다.
- `CreateMLWriter`는 `verified=False`로 시작하지만 저장 경로(`LabelFile.save_create_ml_format`)가 `writer.verified = self.verified`로 덮어쓰므로 verify 상태는 보존된다.

정확한 구조·좌표식은 [../reference/formats.md](../reference/formats.md#3-createml-json).

관련: [annotate-pascal-voc.md](annotate-pascal-voc.md) · [annotate-yolo.md](annotate-yolo.md) · [../explanation/annotation-formats.md](../explanation/annotation-formats.md)
