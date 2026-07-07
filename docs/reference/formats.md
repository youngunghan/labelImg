# 레퍼런스: 어노테이션 포맷

세 출력 포맷의 **정확한 구조**다. 배경·설계 의도는 [../explanation/annotation-formats.md](../explanation/annotation-formats.md).

공통 전제: **VOC와 YOLO는** 저장 직전 박스가 `(x_min, y_min, x_max, y_max)` 절대 픽셀 정수로 환원되며(`libs/labelFile.py:151-174`, 호출부는 `labelFile.py:78, 108`), `x_min`/`y_min`은 1 미만이면 1로 클램프된다. **CreateML은 이 변환을 거치지 않는다** — 캔버스의 float 좌표가 가공 없이 사용되어 저장값(중심 `x`/`y`, `width`/`height`)에 소수점(예: 297.5)이 나올 수 있고 1-클램프도 적용되지 않는다(`labelFile.py:39-51`, `create_ml_io.py:42-47, 73-93`). 이미지 크기는 `[height, width, depth]` 순서(`depth`=grayscale 1, 그 외 3).

---

## 1. PASCAL VOC (`.xml`)

구현: `libs/pascal_voc_io.py` (`PascalVocWriter` / `PascalVocReader`).

`lxml`로 pretty-print하며 들여쓰기는 **탭**이다(`pascal_voc_io.py:30-32`). 인코딩 `utf-8`(`DEFAULT_ENCODING`).

```xml
<annotation>                      <!-- verified면 <annotation verified="yes"> -->
	<folder>images</folder>
	<filename>img001.jpg</filename>
	<path>/abs/path/img001.jpg</path>   <!-- local_img_path가 있을 때만 -->
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{img_size[1]}</width>    <!-- 너비 -->
		<height>{img_size[0]}</height>  <!-- 높이 -->
		<depth>{img_size[2] if len(img_size)==3 else 1}</depth>
	</size>
	<segmented>0</segmented>            <!-- 항상 0 -->
	<object>                            <!-- 박스마다 반복 -->
		<name>person</name>
		<pose>Unspecified</pose>        <!-- 항상 고정 -->
		<truncated>0</truncated>        <!-- 박스가 이미지 경계에 닿으면 1 -->
		<difficult>0</difficult>        <!-- 0 또는 1 -->
		<bndbox>
			<xmin>60</xmin>
			<ymin>40</ymin>
			<xmax>430</xmax>
			<ymax>504</ymax>
		</bndbox>
	</object>
</annotation>
```

규칙:
- `truncated` = 1 if `ymax==height or ymin==1 or xmax==width or xmin==1` else 0 (`pascal_voc_io.py:93-99`).
- `difficult` = `str(bool(difficult) & 1)` → `'0'`/`'1'` (`pascal_voc_io.py:100-101`).
- 좌표는 **절대 픽셀, 정규화 없음**.
- Reader는 `<object>`의 `name`/`difficult`/`bndbox`와 루트 `verified` 속성만 결과에 반영한다. 단 `<filename>`도 파싱하므로(`pascal_voc_io.py:155`) 이 요소가 없으면 예외가 발생해 (아래 예외 삼킴 규칙에 따라) 파일 전체가 빈 결과가 된다 — `<filename>`은 사실상 필수다. `size`/`pose`/`truncated`/`source`는 읽지 않으므로 라운드트립 시 writer 쪽 현재 상태에서 재생성된다.
- Reader는 파싱 예외를 모두 삼킨다(`try/except: pass`, `pascal_voc_io.py:135-138`) → 손상 파일은 조용히 빈 결과.

---

## 2. YOLO (`.txt` + `classes.txt`)

구현: `libs/yolo_io.py` (`YOLOWriter` / `YoloReader`).

라벨 txt — 한 줄당 박스 하나:

```
<class_index> <x_center> <y_center> <width> <height>
```

- 출력 포맷 `"%d %.6f %.6f %.6f %.6f\n"` (`yolo_io.py:74`).
- 모두 **0~1 정규화**:
  - `x_center = (x_min + x_max) / 2 / img_size[1]`
  - `y_center = (y_min + y_max) / 2 / img_size[0]`
  - `w = (x_max - x_min) / img_size[1]`
  - `h = (y_max - y_min) / img_size[0]` (`yolo_io.py:39-43`)
- 역변환(읽기): `cx ± w/2`, `cy ± h/2`를 `[0,1]`로 클램프 후 너비·높이를 곱하고 `round` (`yolo_io.py:135-143`).

사이드카 `classes.txt` — 라벨 txt와 **같은 디렉터리**에 저장:

```
person
car
dog
```

- 한 줄당 클래스명, **줄 번호(0부터) = `class_index`**.
- 저장 시 처음 보는 라벨은 `class_list`에 append되어 인덱스를 부여받는다(`yolo_io.py:47-50`).
- 읽기 시 `classes.txt`를 `strip('\n').split('\n')`로 로드해 `self.classes[int(class_index)]`로 라벨 조회(`yolo_io.py:109,133`).

주의:
- **difficult는 저장되지 않으며 읽을 때 항상 `False`** (`yolo_io.py:170-171`).
- 라벨 txt는 저장 시 `DEFAULT_ENCODING`으로 열지만 `classes.txt`는 인코딩 지정 없이 열린다(`yolo_io.py:60-68`). 읽기 시에는 `classes.txt`(`yolo_io.py:108`)와 라벨 txt(`yolo_io.py:148`) 모두 인코딩 지정이 없다 → 비ASCII 클래스명은 플랫폼 간 불일치 가능.
- `parse_yolo_format`은 공백 분리 5필드를 검증하고 **불량 라인(필드 수·숫자 변환·NaN/inf·클래스 인덱스 범위 오류)은 건너뛰며** `skipped_lines`로 집계한다(`yolo_io.py:147-168`, 포크 견고화 2026-07-07 — 상류는 단일 공백 5필드 강제 언패킹이라 한 줄만 틀려도 크래시). `classes.txt` 부재 시 `YoloParseError`를 던지며(`yolo_io.py:104-107`), 호출부는 이를 에러 대화상자로 보여준다(`labelImg.py:1908-1917`).

---

## 3. CreateML (`.json`)

구현: `libs/create_ml_io.py` (`CreateMLWriter` / `CreateMLReader`).

최상위가 **이미지 객체들의 리스트**다. `indent` 없이 한 줄로 직렬화된다(`create_ml_io.py:71`). 앱 기본 저장은 이미지별 `<stem>.json`(리스트에 항목 1개); 같은 출력 파일을 재사용하면 여러 이미지를 누적할 수 있다.

```json
[
  {
    "image": "img001.jpg",
    "verified": false,
    "annotations": [
      {
        "label": "face",
        "coordinates": { "x": 297.5, "y": 307.5, "width": 105, "height": 115 }
      }
    ]
  }
]
```

- `coordinates.x`/`y`는 박스 **중심**(절대 픽셀), `width`/`height`는 크기.
  - 저장: `width = x_max-x_min`, `height = y_max-y_min`, `x = x_min + width/2`, `y = y_min + height/2` (`create_ml_io.py:73-93`).
  - 복원: `x_min = x - width/2`, `y_min = y - height/2`, `x_max = x + width/2`, `y_max = y + height/2` (`create_ml_io.py:125-129`).
- writer는 기존 파일을 읽어 같은 `image`가 있으면 **그 항목을 통째로 교체**, 없으면 append한다(병합 아님; `create_ml_io.py:60-69`). 단 기본 저장은 이미지별 파일이라 보통 항목 1개.

주의:
- Reader는 `output_list[0]`(첫 항목)의 `verified`만 읽어 파일 전체 verified로 삼는다(`create_ml_io.py:114-115`).
- **Reader가 만드는 모든 shape의 difficult가 `True`로 하드코딩**돼 있다(5-튜플 마지막 원소, `create_ml_io.py:132`) — 다른 포맷과 다른 동작이니 주의.
- writer `__init__`은 `verified=False`지만(`create_ml_io.py:21`) 저장 경로 `LabelFile.save_create_ml_format`이 `writer.verified=self.verified`로 덮어쓴다(`labelFile.py:49`) → verified는 보존됨.
- Reader는 `ValueError`만 잡아 출력하고(빈 결과) 그 외 예외는 전파한다(`create_ml_io.py:102-105`) — VOC reader의 전부-삼킴과 다름.
- 저장은 utf-8이지만(`create_ml_io.py:71`) 기존 파일 병합 읽기(`create_ml_io.py:27`)와 Reader 파싱(`create_ml_io.py:108`)은 인코딩 지정 없이 열린다 → 비ASCII 라벨은 Windows 등에서 라운드트립이 깨질 수 있다.

---

## 공통 Reader 인터페이스

세 Reader 모두 `get_shapes()`가 **같은 5-튜플 리스트**를 반환한다:

```python
(label, points, line_color, fill_color, difficult)
# points = [(x_min,y_min), (x_max,y_min), (x_max,y_max), (x_min,y_max)]  # 시계방향 4코너
# line_color, fill_color = None  (색은 상위에서 결정)
```

이 통일 인터페이스 덕분에 `MainWindow`의 라벨 로드 경로가 포맷과 무관하게 동작한다. 관련: [modules.md](modules.md) · [../explanation/annotation-formats.md](../explanation/annotation-formats.md)
