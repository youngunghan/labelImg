# 어노테이션 포맷 설계

labelImg는 같은 화면 작업 결과를 **세 가지 포맷**으로 저장할 수 있다: PASCAL VOC(XML), YOLO(txt), CreateML(JSON). 이 글은 "왜 세 포맷인지, 어떻게 전환되는지, reader/writer가 어떻게 대칭을 이루는지"를 설명한다. 바이트 단위 정확한 구조는 [../reference/formats.md](../reference/formats.md)에 있다.

## 공통 데이터 모델: 축 정렬 박스

화면의 한 박스는 `Shape`로, 내부적으로는 **4개 꼭짓점**(`points`)을 가진 (사실상) 사각형이다. 저장 직전 PASCAL VOC와 YOLO 경로는 `LabelFile.convert_points_to_bnd_box`(`libs/labelFile.py:151-174`)로 이 꼭짓점들을 **축 정렬 바운딩 박스** `(x_min, y_min, x_max, y_max)`(절대 픽셀, 정수)로 환원한다(호출: `labelFile.py:78`, `:108`). **CreateML은 예외다**: `save_create_ml_format`(`libs/labelFile.py:39-51`)은 shapes를 `CreateMLWriter`에 그대로 넘기고, writer가 꼭짓점 0/1/2를 직접 읽어(`libs/create_ml_io.py:42-45`) `calculate_coordinates`(`libs/create_ml_io.py:73-93`)로 자체적으로 min/max를 계산한다 — 정수 변환 없이 부동소수점 좌표가 그대로 기록된다.

> 세부: `convert_points_to_bnd_box`는 `x_min`/`y_min`이 1 미만이면 **1로 클램프**한다(`libs/labelFile.py:165-172`). 0값 좌표가 Faster R-CNN 학습에서 오류를 내던 문제를 피하기 위한 의도적 처리다(코드 주석). 이 1-클램프는 VOC/YOLO 경로에만 적용되며, CreateML 좌표에는 적용되지 않는다(0에 붙은 좌표도 그대로 기록됨).

이미지 크기는 항상 `image_shape = [height, width, depth]` 순서로 다뤄진다. `depth`는 grayscale이면 1, 아니면 3(`libs/labelFile.py:45-46`). **이 (height, width) 순서가 포맷마다 인덱스 `[0]`/`[1]`로 등장하므로, 호출 측이 순서를 바꾸면 가로/세로가 뒤집힌다.**

## 세 포맷의 좌표 규약 비교

| | PASCAL VOC | YOLO | CreateML |
|---|---|---|---|
| 확장자 | `.xml` | `.txt` (+`classes.txt`) | `.json` |
| 컨테이너 | 이미지 1개 = 파일 1개 | 이미지 1개 = 파일 1개 | 리스트형(기본은 이미지별 파일; 같은 파일 재사용 시 누적 가능) |
| 좌표 단위 | 절대 픽셀 | 0~1 정규화 | 절대 픽셀 |
| 좌표 원점 | 모서리(xmin,ymin,xmax,ymax) | **중심**(cx,cy)+w,h | **중심**(x,y)+w,h |
| 클래스 표현 | 이름 문자열 | 인덱스(↔`classes.txt`) | 이름 문자열 |
| difficult | 보존(0/1) | **폐기** | 코드상 항상 True로 로드(주의) |
| verified | `verified="yes"` 속성 | 없음 | `verified` 필드 |

### PASCAL VOC — 메타데이터가 풍부한 XML

루트 `<annotation>` 아래 `folder`/`filename`/(`path`)/`source`/`size`/`segmented`와 객체별 `<object>`(`name`,`pose`,`truncated`,`difficult`,`bndbox`)를 둔다. `lxml`로 pretty-print하고 들여쓰기는 탭이다(`libs/pascal_voc_io.py:30-32`). `truncated`는 박스가 이미지 경계에 닿으면 1(`libs/pascal_voc_io.py:93-99`), `pose`는 항상 `Unspecified`. 좌표는 정규화 없이 정수 픽셀이라 사람이 읽기 쉽고 ImageNet/VOC 도구와 호환된다.

### YOLO — 학습기 친화적 정규화 txt + classes.txt

한 줄에 박스 하나: `class_index cx cy w h`, 형식은 `"%d %.6f %.6f %.6f %.6f"`(`libs/yolo_io.py:69`). 모두 이미지 크기로 나눈 0~1 정규화 값이다. 클래스 이름 대신 **인덱스**를 쓰므로, 같은 폴더에 `classes.txt`(한 줄당 클래스명, 줄 번호 = 인덱스)를 함께 저장한다. 저장 시 새 라벨은 `class_list`에 append되며 그 인덱스가 부여된다(`libs/yolo_io.py:42-45`).

> **difficult 손실**: YOLO 라인엔 difficult가 없다. 읽을 때 항상 `False`로 들어온다(`libs/yolo_io.py:142-143`). VOC로 difficult를 단 박스를 YOLO로 저장 후 다시 읽으면 difficult가 사라진다.
> **classes.txt 함정**: 이미지 묶음을 처리하는 도중 라벨 목록을 바꾸면 안 된다. `classes.txt`는 저장 때마다 갱신되지만 **이전에 저장한 라벨 파일의 인덱스는 갱신되지 않아** 매핑이 어긋날 수 있다(루트 README).

### CreateML — 리스트형 JSON

최상위가 이미지 객체들의 **리스트**다. 각 이미지 = `{"image": 파일명, "verified": bool, "annotations": [{"label", "coordinates": {x, y, width, height}}]}`. `x,y`는 모서리가 아니라 **박스 중심**(`libs/create_ml_io.py:90-92`). writer는 기존 JSON을 읽어 같은 `image`가 있으면 **교체**, 없으면 append하므로(`libs/create_ml_io.py:60-69`) 같은 출력 파일을 재사용하면 여러 이미지를 누적할 수 있다. 단 앱의 기본 저장(`Ctrl+S`)은 **이미지별 `<stem>.json`** 을 만들어(`labelImg.py:930-934`) 보통 각 파일에 이미지 1개만 들어간다.

## 포맷 전환

현재 포맷은 `MainWindow.label_file_format`(`LabelFileFormat` enum)으로 추적된다. 툴바의 포맷 버튼(`change_format`, `labelImg.py:595`)을 누르면 `PASCAL_VOC → YOLO → CREATE_ML → PASCAL_VOC` 순으로 **순환**하고, `set_format`(`labelImg.py:576`)이 버튼의 텍스트/아이콘과 함께 클래스 변수 `LabelFile.suffix`(`.xml`/`.txt`/`.json`)를 바꾼다. 이 suffix는 "이 파일이 라벨 파일인가"를 판정하는 `is_label_file`의 기준이 된다(`libs/labelFile.py:146-149`). 마지막 사용 포맷은 `SETTING_LABEL_FILE_FORMAT` 키로 영속화된다.

## Reader/Writer 대칭과 공통 인터페이스

세 Reader는 모두 **같은 5-튜플** `(label, points, line_color, fill_color, difficult)`을 반환한다(주석으로 명시, 예: `libs/pascal_voc_io.py:130-131`). 색 두 자리는 `None`이라 색 결정은 상위(MainWindow/Shape 기본색·`generate_color_by_text`)에 위임된다. `points`는 `[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]` 시계방향 4코너로 통일돼, MainWindow의 라벨 로드 경로가 포맷과 무관하게 동작한다.

writer 인터페이스는 완전히 대칭이 아니다. `PascalVocWriter`와 `YOLOWriter`는 `add_bnd_box(x_min, y_min, x_max, y_max, name, difficult)`로 박스를 모은 뒤 `save()`로 직렬화하며, YOLO의 `save()`만 `class_list` 추가 인자를 받는다(`libs/yolo_io.py:49`). 반면 `CreateMLWriter`는 `add_bnd_box`가 없다 — 생성자에서 `shapes` 리스트를 통째로 받아(`libs/create_ml_io.py:14`) `write()`(`libs/create_ml_io.py:25`)로 직렬화하고, `class_list`는 받지 않는다(`LabelFile.save_create_ml_format`이 인자로 받지만 writer에 전달하지 않음, `libs/labelFile.py:39-51`).

## 설계상 함정 / 비대칭 (요약)

- **difficult 비대칭**: VOC만 difficult를 온전히 라운드트립한다. YOLO는 폐기, CreateML reader는 모든 박스를 `difficult=True`로 로드한다(`libs/create_ml_io.py:132` — 의도와 다를 수 있는 동작).
- **색상은 VOC에도 안 들어간다**: `format_shape`는 색을 직렬화하지만 어떤 writer도 색을 기록하지 않는다 — 색은 화면 표시용일 뿐 라벨 파일엔 없다.
- **Reader 실패 처리는 포맷마다 다름**: `PascalVocReader`는 `try/except: pass`로 **모든** 예외를 삼켜 그 시점까지 파싱된 shapes만 돌려준다(`libs/pascal_voc_io.py:135-138`) — 파일 수준 파싱 오류면 빈 리스트, object 순회 도중 실패면 부분 결과다(append가 순회 중 즉시 일어나므로, `libs/pascal_voc_io.py:163-170`). `CreateMLReader`는 **`ValueError`만** 잡아 "JSON decoding failed"를 출력하고(빈 결과), 그 외 예외(KeyError/IOError 등)는 전파한다(`libs/create_ml_io.py:102-105`). `YoloReader`는 예외를 전혀 잡지 않는다(생성자의 try/except가 주석 처리됨, `libs/yolo_io.py:108-111`) — `classes.txt` 부재(IOError), 라인 형식 오류, 범위 밖 클래스 인덱스(IndexError)가 모두 호출자로 전파된다.
- **CreateML verified는 보존된다**: `CreateMLWriter.__init__`은 `verified=False`로 시작하지만(`libs/create_ml_io.py:21`), 실제 저장 경로인 `LabelFile.save_create_ml_format`이 `writer.verified = self.verified`로 덮어쓴다(`libs/labelFile.py:49`; VOC/YOLO도 동일). 따라서 화면의 verified 상태가 그대로 기록된다. 단 reader 쪽은 비대칭이다 — `CreateMLReader.parse_json`은 verified를 현재 이미지와 매칭된 엔트리가 아니라 리스트 첫 엔트리에서 읽으므로(`libs/create_ml_io.py:115`), 여러 이미지가 누적된 JSON에서는 다른 이미지의 verified가 표시될 수 있다.

관련: [../reference/formats.md](../reference/formats.md) · [architecture.md](architecture.md)
