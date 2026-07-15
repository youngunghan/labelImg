# 어노테이션 포맷 설계

labelImg는 같은 화면 작업 결과를 **네 가지 포맷**으로 저장할 수 있다: PASCAL VOC(XML), YOLO(txt), CreateML(JSON), COCO(JSON, 데이터셋 단위). 이 글은 "왜 네 포맷인지, 어떻게 전환되는지, reader/writer가 어떻게 대칭을 이루는지"를 설명한다. 바이트 단위 정확한 구조는 [../reference/formats.md](../reference/formats.md)에 있다.

## 공통 데이터 모델: 축 정렬 박스

화면의 한 박스는 `Shape`로, 내부적으로는 **4개 꼭짓점**(`points`)을 가진 (사실상) 사각형이다. 저장 직전 PASCAL VOC·YOLO·**COCO** 경로는 `LabelFile.convert_points_to_bnd_box`(`libs/labelFile.py:182-205`)로 이 꼭짓점들을 **축 정렬 바운딩 박스** `(x_min, y_min, x_max, y_max)`(절대 픽셀, 정수)로 환원한다(호출: `save_coco_format` `libs/labelFile.py:79`, `save_pascal_voc_format` `:109`, `save_yolo_format` `:139`). **CreateML은 예외다**: `save_create_ml_format`(`libs/labelFile.py:41-53`)은 shapes를 `CreateMLWriter`에 그대로 넘기고, writer가 꼭짓점 0/1/2를 직접 읽어(`libs/create_ml_io.py:42-45`) `calculate_coordinates`(`libs/create_ml_io.py:73-93`)로 자체적으로 min/max를 계산한다 — 정수 변환 없이 부동소수점 좌표가 그대로 기록된다. 즉 네 포맷 중 **셋(VOC/YOLO/COCO)은 같은 축소 함수를 공유**하고, CreateML만 독자 경로를 쓴다.

> 세부: `convert_points_to_bnd_box`는 `x_min`/`y_min`이 1 미만이면 **1로 클램프**한다(`libs/labelFile.py:196-203`). 0값 좌표가 Faster R-CNN 학습에서 오류를 내던 문제를 피하기 위한 의도적 처리다(코드 주석). 이 1-클램프는 VOC/YOLO/COCO 경로에 공통 적용되며, CreateML 좌표에는 적용되지 않는다(0에 붙은 좌표도 그대로 기록됨).

이미지 크기는 항상 `image_shape = [height, width, depth]` 순서로 다뤄진다. `depth`는 grayscale이면 1, 아니면 3(`libs/labelFile.py:45-46`). **이 (height, width) 순서가 포맷마다 인덱스 `[0]`/`[1]`로 등장하므로, 호출 측이 순서를 바꾸면 가로/세로가 뒤집힌다.**

## 네 포맷의 좌표 규약 비교

| | PASCAL VOC | YOLO | CreateML | COCO |
|---|---|---|---|---|
| 확장자 | `.xml` | `.txt` (+`classes.txt`) | `.json` | `.json` (CreateML과 충돌 — 내용 스니핑으로 구분) |
| 컨테이너 | 이미지 1개 = 파일 1개 | 이미지 1개 = 파일 1개 | 리스트형(기본은 이미지별 파일; 같은 파일 재사용 시 누적 가능) | **데이터셋 1개 = 파일 1개**(여러 이미지가 `images[]`/`annotations[]`를 공유) |
| 좌표 단위 | 절대 픽셀 | 0~1 정규화 | 절대 픽셀 | 절대 픽셀 |
| 좌표 원점 | 모서리(xmin,ymin,xmax,ymax) | **중심**(cx,cy)+w,h | **중심**(x,y)+w,h | 모서리(x,y)+w,h — `bbox=[x,y,width,height]` |
| 클래스 표현 | 이름 문자열 | 인덱스(↔`classes.txt`) | 이름 문자열 | 이름 문자열 ↔ `category_id`(`categories[]` 매핑) |
| difficult | 보존(0/1) | **폐기** | 코드상 항상 True로 로드(주의) | **폐기**(스키마에 슬롯 없음, 항상 False로 로드) |
| verified | `verified="yes"` 속성 | 없음 | `verified` 필드 | **폐기**(스키마에 슬롯 없음, 항상 False로 로드) |

### PASCAL VOC — 메타데이터가 풍부한 XML

루트 `<annotation>` 아래 `folder`/`filename`/(`path`)/`source`/`size`/`segmented`와 객체별 `<object>`(`name`,`pose`,`truncated`,`difficult`,`bndbox`)를 둔다. `lxml`로 pretty-print하고 들여쓰기는 탭이다(`libs/pascal_voc_io.py:30-32`). `truncated`는 박스가 이미지 경계에 닿으면 1(`libs/pascal_voc_io.py:93-99`), `pose`는 항상 `Unspecified`. 좌표는 정규화 없이 정수 픽셀이라 사람이 읽기 쉽고 ImageNet/VOC 도구와 호환된다.

### YOLO — 학습기 친화적 정규화 txt + classes.txt

한 줄에 박스 하나: `class_index cx cy w h`, 형식은 `"%d %.6f %.6f %.6f %.6f"`(`libs/yolo_io.py:74`). 모두 이미지 크기로 나눈 0~1 정규화 값이다. 클래스 이름 대신 **인덱스**를 쓰므로, 같은 폴더에 `classes.txt`(한 줄당 클래스명, 줄 번호 = 인덱스)를 함께 저장한다. 저장 시 새 라벨은 `class_list`에 append되며 그 인덱스가 부여된다(`libs/yolo_io.py:47-50`).

> **difficult 손실**: YOLO 라인엔 difficult가 없다. 읽을 때 항상 `False`로 들어온다(`libs/yolo_io.py:172-173`). VOC로 difficult를 단 박스를 YOLO로 저장 후 다시 읽으면 difficult가 사라진다.
> **classes.txt 함정**: 이미지 묶음을 처리하는 도중 라벨 목록을 바꾸면 안 된다. `classes.txt`는 저장 때마다 갱신되지만 **이전에 저장한 라벨 파일의 인덱스는 갱신되지 않아** 매핑이 어긋날 수 있다(루트 README).

### CreateML — 리스트형 JSON

최상위가 이미지 객체들의 **리스트**다. 각 이미지 = `{"image": 파일명, "verified": bool, "annotations": [{"label", "coordinates": {x, y, width, height}}]}`. `x,y`는 모서리가 아니라 **박스 중심**(`libs/create_ml_io.py:90-92`). writer는 기존 JSON을 읽어 같은 `image`가 있으면 **교체**, 없으면 append하므로(`libs/create_ml_io.py:60-69`) 같은 출력 파일을 재사용하면 여러 이미지를 누적할 수 있다. 단 앱의 기본 저장(`Ctrl+S`)은 파일명 스템으로 경로를 만들고(`save_file`, `labelImg.py:1716-1740`) `save_labels`가 포맷별 확장자를 붙이므로(CreateML은 `.json`, `labelImg.py:1053-1057`) **이미지별 `<stem>.json`**이 되어 보통 각 파일에 이미지 1개만 들어간다. **COCO는 이 스템 유도 경로를 아예 타지 않는다** — 아래 참조.

### COCO — 데이터셋 레벨 JSON

세 포맷(VOC/YOLO/CreateML)은 모두 **이미지별 사이드카**다: `save_file`이 이미지 stem으로 경로를 유도하고(`labelImg.py:1716-1740`), `save_labels`가 그 경로에 **현재 이미지 하나만** 직렬화한다(`labelImg.py:1019` 이하 포맷 분기, `libs/labelFile.py`의 각 `save_*_format`은 인자로 받은 `shapes`를 그 파일 하나에 통째로 쓴다). COCO 데이터셋은 정의상 그렇게 만들 수 없다 — COCO json 하나는 `images[]`/`annotations[]`/`categories[]`로 **여러 이미지를 함께** 담는 형식이라, "이미지 1개 = 파일 1개"로 저장하면 이미지마다 별도의 `images`/`annotations` 배열을 가진 파일이 나와 애초에 COCO가 아니게 된다. 그래서 COCO만 **데이터셋 레벨 레인**으로 분리했다:

- **고정된 데이터셋 대상**을 하나 갖는다 — `coco_dataset_target()`(`labelImg.py:670-687`)이 사용자가 Import/Export COCO...로 지정한 경로, 없으면 저장 폴더의 `annotations.json`(`COCO_DEFAULT_DATASET_NAME`, `libs/coco_io.py:17`)을 돌려준다. File 메뉴의 **Import COCO...** / **Export COCO...** 액션(`labelImg.py:255-260`)이 이 대상을 명시적으로 바꾼다.
- `save_labels`의 COCO 분기(`labelImg.py:1058-1069`)는 호출자가 만든 이미지별 경로를 **버리고** 데이터셋 경로로 갈아끼운다. `save_file`도 COCO일 때는 같은 데이터셋 경로로 먼저 분기한다(`labelImg.py:1665-1675`) — 자동저장을 포함해 **`<stem>.json` 사이드카는 COCO에서 단 하나도 만들어지지 않는다.**
- 실제 쓰기는 `COCOWriter.save`(`libs/coco_io.py:216-256`)의 **read-modify-write 병합**이다: 기존 데이터셋 json을 읽고, 이 이미지의 `annotations`만 지운 뒤 새로 채우고, 다른 이미지의 항목은 그대로 둔다(`libs/coco_io.py:230-232`). `save_pascal_voc_format`/`save_yolo_format`/`save_create_ml_format`처럼 "파일 하나 = 이 이미지"가 아니라 "파일 하나 = 이 이미지 + 이미 있던 나머지 전부"다.
- 이미지 키는 **basename이 아니라 데이터셋 파일 기준 상대경로**다(`dataset_relative_name`, `libs/coco_io.py:34-61`). labelImg는 폴더를 재귀 스캔하므로 `train/0001.jpg`와 `val/0001.jpg`가 같은 파일명을 가질 수 있는데, basename으로만 키를 잡으면 두 이미지가 서로의 `images[]` 항목을 덮어쓰고 주석을 합집합으로 읽게 된다(collision). 상대경로 키가 이걸 막는다 — 다른 도구가 남긴 bare-basename 항목은 후보가 유일할 때만 채택·마이그레이션한다(`libs/coco_io.py:174-214`, `:306-313`).
- `bbox`는 `[x, y, width, height]`(좌상단 원점, `libs/coco_io.py:236-248`), `area = width * height`, `iscrowd = 0`으로 고정 기록된다(`libs/coco_io.py:244-249`).
- `category_id ↔ name` 매핑은 병합에도 **안정적**이다 — `sync_categories`(`libs/coco_io.py:140-161`)는 파일에 이미 있는 이름은 기존 id를 재사용하고, 새 이름에만 다음 id를 새로 붙인다. 매 저장마다 카테고리를 다시 번호 매기면 이미 저장된 다른 이미지의 주석이 조용히 재라벨링되기 때문이다.
- COCO 스키마에는 `difficult`/`verified` 슬롯이 없다 — 둘 다 저장 시 폐기되고(`add_bnd_box`가 difficult를 받아도 버림, `libs/coco_io.py:131-138`; writer에 verified 필드 자체가 없음) 읽을 때는 항상 `False`로 돌아온다(`libs/coco_io.py:271-273`, `:335-336`). `verify_image`가 COCO 포맷일 때 이 사실을 상태바로 알린다(`labelImg.py:1587-1593`).

`.json` 확장자는 CreateML과 COCO가 공유한다. 로드 경로는 확장자만으로 리더를 고를 수 없으므로 **내용을 먼저 스니핑**한다: 모든 `.json` 디스패치가 `load_json_by_filename`(`labelImg.py:2267-2277`) 하나로 모이고, `is_coco_json`(`libs/coco_io.py:94-102`, 판정 로직은 `is_coco_dict` `:82-91`)이 최상위가 `images`/`annotations`/`categories` 키를 가진 **dict**인지 본다 — 맞으면 COCO, 아니면(최상위가 **list**) CreateML로 분기한다. 이 순서가 필수인 이유는 `CreateMLReader`가 **`ValueError`만** 잡기 때문이다(`libs/create_ml_io.py:102-105`): COCO dict를 그대로 물리면 `parse_json`의 `for image in output_list`(`libs/create_ml_io.py:119-120`)가 dict의 키(문자열)를 순회하다 `image["image"]`에서 `TypeError`를 내는데, 이는 잡히지 않고 앱까지 올라간다. 즉 스니핑 없이는 CreateML 리더가 COCO 파일을 **우아하게 거부하지 못한다.**

거꾸로 COCO 리더도 방어적이다: 데이터셋 json이 열려는 이미지를 **모르면** 포맷을 바꾸지 않고 `False`를 돌려준다(`load_coco_json_by_filename`, `labelImg.py:2292-2317`) — 저장 폴더에 우연히 놓인 남의 `annotations.json`이 앱을 COCO로 끌고 가지 못한다.

## 포맷 전환

현재 포맷은 `MainWindow.label_file_format`(`LabelFileFormat` enum, `libs/labelFile.py:19-24` — `PASCAL_VOC`/`YOLO`/`CREATE_ML`/`COCO`)으로 추적된다. 툴바의 포맷 버튼(`change_format`, `labelImg.py:657`)을 누르면 **`PASCAL_VOC → YOLO → CREATE_ML → COCO → PASCAL_VOC` 순으로 4단계 순환**하고(`labelImg.py:657-667` — COCO는 세 번째가 아니라 네 번째로 추가됐다), `set_format`(`labelImg.py:632`)이 버튼의 텍스트/아이콘과 함께 클래스 변수 `LabelFile.suffix`(`.xml`/`.txt`/`.json`)를 바꾼다. COCO도 `.json`을 쓰므로 `LabelFile.suffix`만으로는 COCO와 CreateML을 구분할 수 없다 — `is_label_file`(`libs/labelFile.py:177-180`)이 확장자만 보는 것은 여전하고, 실제 구분은 위 `.json` 내용 스니핑이 담당한다. 마지막 사용 포맷은 `SETTING_LABEL_FILE_FORMAT` 키로 영속화된다.

## Reader/Writer 대칭과 공통 인터페이스

네 Reader는 모두 **같은 5-튜플** `(label, points, line_color, fill_color, difficult)`을 반환한다(주석으로 명시, 예: `libs/pascal_voc_io.py:130-131`, COCO는 `libs/coco_io.py:262-263`). 색 두 자리는 `None`이라 색 결정은 상위(MainWindow/Shape 기본색·`generate_color_by_text`)에 위임된다. `points`는 `[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]` 시계방향 4코너로 통일돼, MainWindow의 라벨 로드 경로가 포맷과 무관하게 동작한다.

writer 인터페이스는 완전히 대칭이 아니다. `PascalVocWriter`·`YOLOWriter`·`COCOWriter`는 `add_bnd_box(x_min, y_min, x_max, y_max, name, difficult)`로 박스를 모은 뒤 `save()`로 직렬화한다(COCO는 `libs/coco_io.py:131-138`, `:216`). `YOLOWriter.save()`와 `COCOWriter.save()`는 둘 다 `class_list`를 추가로 받지만(`libs/yolo_io.py:54`, `libs/coco_io.py:216`) 쓰임이 다르다 — YOLO는 인덱스를 매기려고, COCO는 `category_id` 매핑을 만들려고(`sync_categories`, `libs/coco_io.py:140-161`) 받는다. 셋 다 `target_file`을 받지만 VOC/YOLO는 `None`이면 `self.filename` 기반 경로로 조용히 대체하는 데 반해(`libs/pascal_voc_io.py:116-118`, `libs/yolo_io.py:59-64`), COCO는 `target_file`이 없으면 **바로 예외**를 던진다(`COCOParseError`, `libs/coco_io.py:219-220`) — 데이터셋 레인에는 "이미지 stem으로 대체" 같은 폴백 경로 자체가 없기 때문이다. 반면 `CreateMLWriter`는 `add_bnd_box`가 없다 — 생성자에서 `shapes` 리스트를 통째로 받아(`libs/create_ml_io.py:14`) `write()`(`libs/create_ml_io.py:25`)로 직렬화하고, `class_list`는 받지 않는다(`LabelFile.save_create_ml_format`이 인자로 받지만 writer에 전달하지 않음, `libs/labelFile.py:41-53`).

## 설계상 함정 / 비대칭 (요약)

- **difficult 비대칭**: VOC만 difficult를 온전히 라운드트립한다. YOLO·COCO는 폐기(둘 다 읽을 때 `False`), CreateML reader는 모든 박스를 `difficult=True`로 로드한다(`libs/create_ml_io.py:133` — 의도와 다를 수 있는 동작).
- **verified 비대칭**: VOC(XML 속성)·CreateML(JSON 필드)만 verified를 저장한다. YOLO는 애초에 슬롯이 없고, COCO는 슬롯이 없어 폐기된다(`verify_image`가 COCO일 때 상태바로 알림, `labelImg.py:1587-1593`).
- **색상은 어떤 포맷에도 안 들어간다**: `format_shape`는 색을 직렬화하지만 어떤 writer도 색을 기록하지 않는다 — 색은 화면 표시용일 뿐 라벨 파일엔 없다.
- **Reader 실패 처리는 포맷마다 다름**: `PascalVocReader`는 `try/except: pass`로 **모든** 예외를 삼켜 그 시점까지 파싱된 shapes만 돌려준다(`libs/pascal_voc_io.py:135-138`) — 파일 수준 파싱 오류면 빈 리스트, object 순회 도중 실패면 부분 결과다(append가 순회 중 즉시 일어나므로, `libs/pascal_voc_io.py:163-170`). `CreateMLReader`는 **`ValueError`만** 잡아 "JSON decoding failed"를 출력하고(빈 결과), 그 외 예외(KeyError/IOError 등)는 전파한다(`libs/create_ml_io.py:102-105`) — 이 좁은 catch가 위 `.json` 스니핑을 필수로 만든 이유다. `YoloReader`는 (포크 견고화 2026-07-07) `classes.txt` 부재 시 `YoloParseError`를 명시적으로 던지고(`libs/yolo_io.py:104-107`) 호출부가 에러 대화상자로 처리하며(`load_yolo_txt_by_filename`, `labelImg.py:2244-2265`), 불량 라인(NaN/inf 좌표 포함)은 건너뛰고 `skipped_lines`로 집계해 상태바에 알린다(`libs/yolo_io.py:149-173`). `COCOReader`는 데이터셋 파싱 실패를 `COCOParseError`(`ValueError` 서브클래스, `libs/coco_io.py:20-27`)로 좁혀 잡아 출력만 하고(`libs/coco_io.py:277-282`) 빈 결과로 남으며, 이 이미지가 데이터셋에 아예 없는 경우와 있지만 박스가 0개인 경우를 `found_image` 플래그로 구분한다(`libs/coco_io.py:276`, `:315-317`).
- **CreateML verified는 보존된다**: `CreateMLWriter.__init__`은 `verified=False`로 시작하지만(`libs/create_ml_io.py:21`), 실제 저장 경로인 `LabelFile.save_create_ml_format`이 `writer.verified = self.verified`로 덮어쓴다(`libs/labelFile.py:51`; VOC/YOLO/COCO도 동일한 패턴). 따라서 화면의 verified 상태가 그대로 기록된다(COCO만 애초에 필드가 없어 이 대입이 저장에 영향을 주지 못한다). 단 CreateML reader 쪽은 비대칭이다 — (2026-07-08 수정) `CreateMLReader.parse_json`은 verified를 현재 이미지와 매칭된 엔트리에서 읽는다(`libs/create_ml_io.py:121`) — 이전에는 리스트 첫 엔트리에서 읽어 다중 이미지 JSON에서 오표시됐다.

관련: [../reference/formats.md](../reference/formats.md) · [architecture.md](architecture.md) · [ml-assist-architecture.md](ml-assist-architecture.md) §리스크 7·8 (COCO 데이터셋 레인·`.json` 충돌의 설계 검증 배경)
