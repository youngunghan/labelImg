# 레퍼런스: 어노테이션 포맷

네 출력 포맷의 **정확한 구조**다. 배경·설계 의도는 [../explanation/annotation-formats.md](../explanation/annotation-formats.md).

공통 전제: **VOC·YOLO·COCO는** 저장 직전 박스가 `(x_min, y_min, x_max, y_max)` 절대 픽셀 정수로 환원되며(`LabelFile.convert_points_to_bnd_box`, `libs/labelFile.py:182-205`, 호출부는 `labelFile.py:79`(COCO)`, 109`(VOC)`, 139`(YOLO)), `x_min`/`y_min`은 1 미만이면 1로 클램프된다. **CreateML은 이 변환을 거치지 않는다** — 캔버스의 float 좌표가 가공 없이 사용되어 저장값(중심 `x`/`y`, `width`/`height`)에 소수점(예: 297.5)이 나올 수 있고 1-클램프도 적용되지 않는다(`labelFile.py:41-53`, `create_ml_io.py:42-47, 73-93`). 이미지 크기는 `[height, width, depth]` 순서(`depth`=grayscale 1, 그 외 3).

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
- 읽기 시 `classes.txt`를 `strip('\n').split('\n')`로 로드해 `self.classes[int(class_index)]`로 라벨 조회(`yolo_io.py:111,135`).

주의:
- **difficult는 저장되지 않으며 읽을 때 항상 `False`** (`yolo_io.py:172-173`).
- (2026-07-08 통일) 저장(`yolo_io.py:60-68`)은 라벨 txt와 `classes.txt` 모두 `DEFAULT_ENCODING`(utf-8)으로 고정됐다 — 비ASCII 클래스명이 플랫폼과 무관하게 라운드트립된다. (이전에는 `classes.txt`와 읽기 경로가 OS 로케일을 따라 한국어 Windows(cp949)에서 mojibake/크래시 위험이 있었다.) 읽기는 둘로 나뉜다: 라벨 txt는 `ENCODE_METHOD`(=`DEFAULT_ENCODING`)으로 읽고(`yolo_io.py:150`), `classes.txt`는 하드코딩된 `utf-8-sig`로 읽어(`yolo_io.py:110`) 첫 클래스명 앞의 Windows/Notepad BOM을 제거한다.
- `parse_yolo_format`은 공백 분리 5필드를 검증하고 **불량 라인(필드 수·숫자 변환·NaN/inf·클래스 인덱스 범위 오류)은 건너뛰며** `skipped_lines`로 집계한다(`yolo_io.py:149-173`, 포크 견고화 2026-07-07 — 상류는 단일 공백 5필드 강제 언패킹이라 한 줄만 틀려도 크래시). `classes.txt` 부재 시 `YoloParseError`를 던지며(`yolo_io.py:104-107`), 호출부는 이를 에러 대화상자로 보여준다(`labelImg.py:2200-2209`).

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
  - 복원: `x_min = x - width/2`, `y_min = y - height/2`, `x_max = x + width/2`, `y_max = y + height/2` (`create_ml_io.py:126-130`).
- writer는 기존 파일을 읽어 같은 `image`가 있으면 **그 항목을 통째로 교체**, 없으면 append한다(병합 아님; `create_ml_io.py:60-69`). 단 기본 저장은 이미지별 파일이라 보통 항목 1개.

주의:
- Reader는 현재 이미지와 **매칭되는 항목**의 `verified`를 읽는다(`create_ml_io.py:119-123`; 2026-07-08 수정 — 이전에는 첫 항목에서 읽어 다중 이미지 파일에서 오표시).
- **Reader가 만드는 모든 shape의 difficult가 `True`로 하드코딩**돼 있다(5-튜플 마지막 원소, `create_ml_io.py:133`) — 다른 포맷과 다른 동작이니 주의.
- writer `__init__`은 `verified=False`지만(`create_ml_io.py:21`) 저장 경로 `LabelFile.save_create_ml_format`이 `writer.verified=self.verified`로 덮어쓴다(`labelFile.py:49`) → verified는 보존됨.
- Reader는 `ValueError`만 잡아 출력하고(빈 결과) 그 외 예외는 전파한다(`create_ml_io.py:102-105`) — VOC reader의 전부-삼킴과 다름.
- (2026-07-08 통일) 저장(`create_ml_io.py:71`)·병합 읽기(`create_ml_io.py:27`)·Reader 파싱(`create_ml_io.py:108`) 모두 utf-8로 고정 — 비ASCII 라벨이 안전하게 라운드트립된다.

---

## 4. COCO (`.json`, 데이터셋 레벨)

구현: `libs/coco_io.py` (`COCOWriter` / `COCOReader`).

**CRITICAL**: COCO는 앞의 세 포맷과 달리 이미지별 사이드카가 아니라 **데이터셋 레벨** 포맷이다 — 하나의 json이 여러 이미지의 `images`/`annotations`/`categories`를 함께 담는다(설계 노트, `coco_io.py:11-17`). 기본 타깃은 `<save dir>/annotations.json`(`COCO_DEFAULT_DATASET_NAME`, `coco_io.py:17`; 타깃 결정 로직 `MainWindow.coco_dataset_target`, `labelImg.py:670-687`)이고, File 메뉴의 **Import COCO...** / **Export COCO...**(`labelImg.py:257-260, 2264-2314`)로 다른 파일을 명시적으로 고를 수 있다. `save_labels`가 COCO를 저장할 때는 이 공유 파일을 **read-modify-write**로 병합한다 — 매 저장마다 파일 전체를 읽어 현재 이미지의 `images`/`annotations` 항목만 교체하고 다시 쓴다(`COCOWriter.save`, `coco_io.py:216-256`; 디스패치 `labelImg.py:1058-1069`). VOC/YOLO/CreateML과 달리 이미지별 자동 로드(사이드카 `<stem>.json`)에는 연결되지 않는다 — `<stem>.json`을 COCO로 오인해 자동 로드하는 경로는 없다.

```json
{
  "images": [
    {"id": 1, "file_name": "train/0001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [60, 40, 370, 464],
      "area": 171680,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": ""}
  ]
}
```

- `bbox`는 `[x, y, width, height]`(좌상단 기준) — 나머지 세 포맷의 `(xmin,ymin,xmax,ymax)`/중심 표현과 다르다(`coco_io.py:244-247`, 읽을 때 복원은 `coco_io.py:330-334`).
- `area = width * height`, `iscrowd`는 항상 `0`(`coco_io.py:248-249`).
- `category_id` ↔ 이름 매핑은 병합을 거듭해도 안정적이다: `sync_categories`가 기존 `categories`의 id를 그대로 재사용하고 새 라벨명만 다음 id를 받는다(재넘버링 없음, `coco_io.py:140-161`).

이미지 키잉(상대 경로) — **바스네임 충돌 방지**:
- `images[].file_name`은 데이터셋 json 기준 **상대 경로**(슬래시 통일)이지, 단순 basename이 아니다(`dataset_relative_name`, `coco_io.py:34-61`). labelImg는 디렉터리를 재귀적으로 스캔하므로(`MainWindow.scan_all_images`가 `os.walk`) `train/0001.jpg`와 `val/0001.jpg`처럼 **basename이 같은 두 이미지**가 흔한 워크플로다 — basename만으로 키를 잡으면 하나가 다른 하나의 `images[]` 항목을 덮어쓰고, 읽을 때는 두 이미지의 박스가 합쳐지는 **조용한 데이터 손상**이 발생한다. 상대 경로를 계산할 수 없을 때(상대경로 입력, 또는 Windows에서 서로 다른 드라이브)만 basename으로 폴백한다(`coco_io.py:52-61`).
- 다른 도구가 만든 데이터셋처럼 basename으로만 키가 잡힌 기존 항목은, 그 basename에 해당하는 후보가 **정확히 하나**일 때만 상대 경로로 마이그레이션해 채택한다(쓰기 쪽 `COCOWriter.sync_image`, `coco_io.py:174-214`; 읽기 쪽 동등한 폴백은 `COCOReader.parse_json`, `coco_io.py:292-317`). 후보가 둘 이상이면(바로 그 basename 충돌 상황) 채택하지 않는다.

COCO가 담지 **않는** 것:
- `difficult` — COCO 스키마에 필드가 없어 저장 시 버려지고, 읽으면 항상 `False`로 복원된다(`COCOWriter.add_bnd_box` 주석, `coco_io.py:131-138`; `COCOReader.add_shape`, `coco_io.py:330-336`).
- `verified` — writer/reader 모두 `verified=False`로 고정된다(`coco_io.py:129, 273`). `MainWindow.verify_image`는 COCO 포맷으로 저장할 때 "COCO는 verified/difficult를 저장하지 않는다"는 상태바 안내를 띄운다(`labelImg.py:1606-1612`).

`.json` 콘텐츠 스니핑 (COCO vs CreateML):

COCO와 CreateML은 둘 다 확장자가 `.json`이라 확장자만으로는 리더를 고를 수 없다. 그래서 모든 `.json` 로드 경로는 **먼저 파싱한 내용을 살펴** 리더를 정한다 — 최상위가 `images`/`annotations`/`categories` 키를 가진 dict면 COCO, 리스트면 CreateML(`is_coco_dict`, `coco_io.py:82-91`; 파일 단위 래퍼 `is_coco_json`, `coco_io.py:94-102`). 자동 로드 디스패치는 `MainWindow.load_json_by_filename`(`labelImg.py:2267-2277`)이 맡는다(주석: `CreateMLReader`는 디코드 오류만 잡으므로 이 스니핑 없이는 COCO 데이터셋을 조용히 "박스 0개"로 오독했을 것). `load_coco_json_by_filename`(`labelImg.py:2292-2317`)은 데이터셋에 이 이미지가 없으면(`found_image=False`) 포맷을 바꾸지 않고 조용히 물러난다 — 저장 디렉터리에 우연히 있는 COCO 데이터셋이 무관한 이미지들의 포맷까지 바꾸지 않게 하기 위함이다. **Import COCO...** 다이얼로그도 같은 스니핑으로 고른 파일을 검증한다(`import_coco_dialog`, `labelImg.py:2319-2346`).

---

## 공통 Reader 인터페이스

네 Reader 모두 `get_shapes()`가 **같은 5-튜플 리스트**를 반환한다:

```python
(label, points, line_color, fill_color, difficult)
# points = [(x_min,y_min), (x_max,y_min), (x_max,y_max), (x_min,y_max)]  # 시계방향 4코너
# line_color, fill_color = None  (색은 상위에서 결정)
```

이 통일 인터페이스 덕분에 `MainWindow`의 라벨 로드 경로가 포맷과 무관하게 동작한다. 관련: [modules.md](modules.md) · [../explanation/annotation-formats.md](../explanation/annotation-formats.md)
