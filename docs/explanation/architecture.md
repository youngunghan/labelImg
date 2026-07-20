# 아키텍처

labelImg는 **단일 윈도우 데스크톱 앱**이다. 외부 서버·DB·네트워크가 없고, 모든 상태는 메모리에 있으며 결과는 디스크의 라벨 파일(및 포크의 good/bad 분류 시 이동되는 이미지·라벨 파일)로 떨어진다. 구조는 전형적인 **컨트롤러 + 뷰 + 모델** 변형이다.

```
                ┌──────────────────────────────────────────┐
                │  MainWindow (labelImg.py)  ── 컨트롤러     │
                │  · UI(액션/메뉴/툴바/도크) 구성             │
                │  · 모든 사용자 액션 슬롯                    │
                │  · 파일/디렉터리 탐색, 포맷 선택            │
                │  · 설정 영속화(Settings)                   │
                └───────▲───────────────┬──────────────────┘
        시그널(7종)      │               │ 메서드 호출
   newShape/selection/  │               │ load_pixmap/load_shapes/
   shapeMoved/zoom...   │               │ select_shape/delete_selected...
                ┌───────┴───────────────▼──────────────────┐
                │  Canvas (libs/canvas.py)  ── 뷰/드로잉     │
                │  · pixmap 위 사각형 그리기/선택/이동/편집   │
                │  · 마우스·키보드·휠 이벤트, paintEvent      │
                │  · CREATE / EDIT 모드, 줌·스케일 변환        │
                └───────────────┬──────────────────────────┘
                                │ Shape 인스턴스 생성/조작
                ┌───────────────▼──────────────────────────┐
                │  Shape (libs/shape.py)  ── 모델(박스 1개)  │
                │  · points(최대4) / label / difficult / 색  │
                │  · paint() / contains_point / nearest_vertex│
                └──────────────────────────────────────────┘

  저장:  MainWindow ── LabelFile(libs/labelFile.py) ──┬─ PascalVocWriter
                                                      ├─ YOLOWriter
                                                      ├─ CreateMLWriter
                                                      └─ COCOWriter(데이터셋 레인, 신규)
  로드:  MainWindow ──(직접 생성/호출)──── PascalVocReader / YoloReader / CreateMLReader / COCOReader

  AI:    MainWindow ── 소유·배선만 ──┬─ InferenceService(QObject) ── ModelBackend(StubBackend/YoloOnnxBackend)
                                    └─ AssistController(QObject) ── provisional Shape 수명주기
```

## 3개 핵심 계층

### 1. MainWindow — 컨트롤러(God-object)

`MainWindow`(`labelImg.py:76`)는 `QMainWindow`와 `WindowMixin`을 다중 상속한다. `__init__`에서 **모든** 위젯·액션·메뉴·툴바·도크를 구성하고 `Canvas`·`ZoomWidget`·`LightWidget`·`LabelDialog`·`ColorDialog`·콤보박스들을 임베드한다. 포크 확장으로 `InferenceService`·`AssistController`도 여기서 **생성·소유**된다(`labelImg.py:444-446`) — 자세한 내용은 아래 **AI 어시스트 컨트롤러** 절 참고.

액션은 `partial(new_action, self)`로 생성하고(`labelImg.py:248`), 만들어진 액션들을 `self.actions = Struct(save=..., create=..., ...)`처럼 키워드 인자로 묶어 둔다(`labelImg.py:449`).

책임이 한 클래스에 모여 있다:
- **UI 구성**: `WindowMixin.menu()`/`toolbar()` 헬퍼로 메뉴바·좌측 툴바 생성(`labelImg.py:59-73`).
- **액션 라우팅**: 박스 생성/편집/복사/삭제, 라벨 리스트 상호작용, 포맷 전환, verify, difficult 등.
- **모델 상태 보관**: `items_to_shapes` / `shapes_to_items` 양방향 딕셔너리(`labelImg.py:153-154`)가 라벨 리스트 항목(`HashableQListWidgetItem`)과 `Shape`를 잇는다.
- **파일 I/O 선택**: 어떤 Reader/Writer를 쓸지 포맷에 따라 분기(PASCAL VOC/YOLO/CreateML/**COCO**, 후자는 데이터셋 레벨 — [annotation-formats.md](annotation-formats.md) 참고).
- **설정 영속화**: 시작 시 `Settings().load()`, 종료 시 `closeEvent`에서 모든 키 기록 후 `save()`. 추가로(포크 확장) predefined 클래스 목록도 영속화된다: `get_persistent_classes_file`(`labelImg.py:2177`)이 실제 파일을 결정하는데, frozen exe면 exe 옆 `predefined_classes.txt`를, 소스 실행이면 `class_file` 인자 경로를 쓰고, 파일이 없으면 번들 기본 파일 복사 또는 'person' 한 줄로 자동 생성한다. 이 파일은 Edit Default Classes 메뉴(Ctrl+Shift+E, `edit_default_classes` `labelImg.py:2197`)로 편집·덮어쓰기된다.
- **이미지 분류(포크 확장)**: `g`/`b`로 현재 이미지와 짝 라벨 파일(.xml/.txt/.json)을 형제 폴더 `<열린폴더>_good`/`_bad`로 이동(`classify_current_image`, `labelImg.py:1805`; 충돌 시 stem 리네임), `Ctrl+Z`로 되돌리기(`undo_classify`, `labelImg.py:1964`). 라벨 파일은 `default_save_dir` 우선, 없으면 이미지 옆 폴더에서 찾으며, 이동 이력은 `self.classify_history` 스택(`labelImg.py:116-118`)에 보관.
- **AI 어시스트 컨트롤러 소유·배선(포크 확장, 신규)**: `InferenceService`/`AssistController`를 생성하고, 그 액션을 AI 메뉴와 `onLoadActive`에 등록하는 것까지만 한다 — AI 행위 자체의 코드는 `MainWindow` 안에 없다(아래 **AI 어시스트 컨트롤러** 절).

> 이 "한 클래스가 다 한다" 구조는 **God-object/Controller 패턴**으로, 작은 앱에선 단순하지만 새 포맷·기능을 추가할 때 `MainWindow`와 `LabelFile`/Reader 양쪽을 함께 고쳐야 하는 결합이 있다.

### 2. Canvas — 뷰/드로잉 위젯

`Canvas`(`libs/canvas.py:24`)는 `QWidget`을 상속한 순수 드로잉/편집 위젯이다. GPU·파일을 모르고, 오직 `pixmap` 위에 `Shape`들을 그리고 마우스·키보드로 편집한다. 두 모드 `CREATE`/`EDIT`를 오가며(`libs/canvas.py:33`) 박스를 그리거나 선택·이동·정점편집한다. 자세한 동작은 [canvas-interaction-model.md](canvas-interaction-model.md).

Canvas는 윈도우에 **7개 시그널**로 보고한다(`libs/canvas.py:24-31`):

| 시그널 | 언제 | MainWindow 슬롯 |
|---|---|---|
| `newShape()` | 박스 1개 완성 | `new_shape` → LabelDialog 팝업 후 라벨 부여 |
| `selectionChanged(bool)` | 선택 변경 | `shape_selection_changed` |
| `shapeMoved()` | 박스/정점 이동 | `set_dirty` |
| `drawingPolygon(bool)` | 그리기 시작/종료 | `toggle_drawing_sensitive` |
| `zoomRequest(int)` | Ctrl+휠 | `zoom_request` |
| `lightRequest(int)` | Ctrl+Shift+휠 | `light_request` |
| `scrollRequest(int,int)` | 휠/드래그 패닝 | `scroll_request` |

반대 방향으로 MainWindow는 Canvas의 메서드(`load_pixmap`, `load_shapes`, `set_last_label`, `select_shape`, `delete_selected`, `copy_selected_shape`, `set_editing`, `set_shape_visible` 등)를 직접 호출해 캔버스를 구동한다(예: `load_pixmap` `labelImg.py:1327`, `load_shapes` `:1020`, `set_last_label` `:1158`, `select_shape` `:1118`, `delete_selected` `:2133`). 시그널 연결 자체는 `__init__`의 `labelImg.py:220-237`에서 이뤄진다. 즉 **Canvas→Window는 시그널, Window→Canvas는 직접 호출**이라는 단방향성을 가진 MVC 변형이다.

### 3. Shape — 모델(박스 1개)

`Shape`(`libs/shape.py:23`)는 어노테이션 하나를 표현한다: `points`(꼭짓점, 사각형은 최대 4점 — `reach_max_points`가 `len>=4`로 제한, `libs/shape.py:84-87`), `label`, `difficult` 플래그, 닫힘 상태 `_closed`, 선택/하이라이트 상태, 색상. 자기 자신을 `paint(painter)`로 그리고(`libs/shape.py:104`), `contains_point`·`nearest_vertex`로 히트테스트, `move_by`·`move_vertex_by`로 이동, `copy()`로 복제한다.

색상·`point_size`·`scale`·`label_font_size`는 **클래스 변수**라(`libs/shape.py:36-47`) 한 곳에서 바꾸면 모든 `Shape`에 적용된다. MainWindow는 이를 전역 설정처럼 갱신한다(`Shape.line_color`/`Shape.difficult` 등). 단, 인스턴스에 `line_color`를 대입하면 인스턴스 속성이 클래스 색을 가린다 — `__init__` 인자는 그리기 중인 펜딩 라인 색에 쓰이고(`libs/canvas.py:47`), 라벨별 색(`generate_color_by_text`)은 로드/라벨 부여 시 `shape.line_color = ...` 직접 대입으로 입혀진다(`labelImg.py:1008-1011`, `libs/canvas.py:683-687`).

포크 확장(신규)으로 `Shape`는 `provisional`/`confidence`/`shape_type` 필드도 갖는다(`libs/shape.py:57-64`). `provisional=True`인 도형은 AI가 제안했을 뿐 사용자가 아직 받아들이지 않은 박스로, `paint()`가 점선 윤곽 + 반투명 채움으로 다르게 그린다(`libs/shape.py:110-113`, `:162-164`). 자세한 수명주기는 아래 **AI 어시스트 컨트롤러** 절과 [ml-assist-architecture.md](ml-assist-architecture.md) §provisional Shape 수명주기를 참고.

## 저장/로드 — LabelFile 파사드와 Reader/Writer

`MainWindow.save_labels`(`labelImg.py:1033`)가 입출력 디스패처다. 각 `Shape`를 `format_shape`로 `dict(label, line_color, fill_color, points, difficult)`로 직렬화한 뒤, 현재 포맷에 따라 `LabelFile`(`libs/labelFile.py:30`)의 메서드를 호출한다:

- `LabelFileFormat.PASCAL_VOC` → `save_pascal_voc_format` → `PascalVocWriter`
- `LabelFileFormat.YOLO` → `save_yolo_format(..., label_hist)` → `YOLOWriter`
- `LabelFileFormat.CREATE_ML` → `save_create_ml_format(..., label_hist)` → `CreateMLWriter`
- `LabelFileFormat.COCO` → `save_coco_format(..., label_hist)` → `COCOWriter`(포크 확장, 신규 — 이미지별 파일이 아니라 **데이터셋 json 병합**. `save_labels`의 COCO 분기가 호출자의 이미지별 경로를 버리고 `coco_dataset_target()`으로 갈아끼운다, `labelImg.py:1072-1083`. 근거는 [annotation-formats.md](annotation-formats.md) §COCO)

직렬화 직전, 이 디스패치 자체가 **단일 저장 초크포인트**다: 캔버스의 모든 도형 중 `provisional`(AI가 제안했지만 아직 수락되지 않은 도형, 포크 확장·신규)인 것은 여기서 걸러진다 — `shapes = [format_shape(shape) for shape in self.canvas.shapes if not shape.provisional]`(`labelImg.py:1054`). `Ctrl+S`·Save As·Export COCO...·verify·자동저장 등 **모든** 저장 경로가 `_save_file` → `save_labels`로 수렴하므로, 이 한 줄이 포맷·경로에 상관없이 미확정 AI 추측이 디스크로 새는 것을 막는다. 자세한 수명주기는 아래 **AI 어시스트 컨트롤러** 절.

`LabelFile`은 포맷별 writer를 아는 **얇은 파사드**다. PascalVOC/YOLO/**COCO** 경로에서는 각 writer로 위임하기 전에 `convert_points_to_bnd_box`로 다각형 points를 축 정렬 박스 `(x_min,y_min,x_max,y_max)`로 환원한다(정의 `libs/labelFile.py:182-205`, 호출 `:79`(COCO)·`:109`(VOC)·`:139`(YOLO) — 이때 x_min/y_min<1은 1로 클램프, `:196-203`). 단 CreateML 경로는 예외로, `LabelFile`이 shape dict를 그대로 넘기고 `CreateMLWriter.write()`가 내부에서 꼭짓점 순서를 가정해 중심좌표+폭·높이로 변환한다(`libs/create_ml_io.py:39-58`). 로드는 대칭으로 `PascalVocReader`/`YoloReader`/`CreateMLReader`/`COCOReader`가 라벨 파일을 읽어 `(label, points, None, None, difficult)` 5-튜플 리스트를 돌려주고, MainWindow가 이를 `Shape`로 복원해 Canvas에 싣는다(단 CreateMLReader는 포맷에 difficult 필드가 없어 마지막 요소를 항상 True로 채우고 — `libs/create_ml_io.py:133` — COCOReader는 반대로 항상 False로 채운다, `libs/coco_io.py:335-336`). 포맷별 정확한 구조는 [annotation-formats.md](annotation-formats.md) · [../reference/formats.md](../reference/formats.md).

## AI 어시스트 컨트롤러 — `InferenceService` / `AssistController` (신규)

포크 확장으로 `MainWindow`는 두 개의 얇은 컨트롤러 객체를 **생성·소유·배선**한다(`labelImg.py:444-446`):

- **`InferenceService`**(QObject, `libs/inference/service.py:186`) — 모델 백엔드를 들고 단일 워커 `QThreadPool`에서 추론을 돌리고 결과를 queued 시그널로 낸다.
- **`AssistController`**(QObject, `libs/assist/controller.py:112`) — AI 액션(Auto-label Image = Ctrl+I, Accept All = Ctrl+Return, Reject All = Ctrl+Backspace, 신뢰도 슬라이더), provisional 도형 수명주기를 소유한다. 만들어진 액션은 새 **AI 메뉴**(`labelImg.py:478`, `:504`)와 이미지-의존 액션 집합 `onLoadActive`(`labelImg.py:468-471`)에 등록되고, `toggle_actions()`가 활성화 여부를 정리한 뒤 `AssistController.refresh_actions()`가 모델 유무에 따라 다시 손본다(`labelImg.py:752`).

핵심은 **경계**다: 두 컨트롤러 안의 코드가 `MainWindow`로 새어 들어오지 않는다 — `MainWindow`가 만지는 것은 생성·소유·배선과 위 저장 초크포인트의 필터 한 줄뿐이다. `libs/inference/`의 코어(`types`/`backend`/`stub`/`registry`)는 PyQt5를 import하지 않으며, 모델 백엔드는 `ModelBackend` ABC(`libs/inference/backend.py:36`, `predict`/`segment`/`embed`)라는 하나의 심(seam) 뒤에 있다 — 결정론적인 `StubBackend`와 실제 ONNX YOLO 추론을 하는 `YoloOnnxBackend`(`libs/inference/yolo_onnx.py`)가 같은 인터페이스로 교체 가능하다.

이 설계의 근거(왜 `MainWindow`에 직접 얹지 않았는지, 스레딩 규칙, 좌표계 계약, provisional 수명주기의 세부, 열린 리스크)는 이 페이지에서 중복하지 않는다 — 전개는 [ml-assist-architecture.md](ml-assist-architecture.md)에 있다.

## 기반 모듈

- **`libs/settings.py`** `Settings`: 설정 dict을 `~/.labelImgSettings.pkl`에 pickle로 직렬화. dict 유사 인터페이스(`[]`/`get`/`save`/`load`/`reset`). → [../reference/settings.md](../reference/settings.md)
- **`libs/stringBundle.py`** `StringBundle`: 로케일별 i18n 문자열을 Qt 리소스(`:/strings*`)에서 로드. `get_bundle()` 팩토리로만 생성(생성자 보호). 기본 폴백 `en`.
- **`libs/utils.py`**: `new_action`/`new_button`/`new_icon`/`add_actions`(UI 팩토리), `generate_color_by_text`(라벨→색 해시), `natural_sort`(파일 정렬), `distance`, `label_validator`.
- **`libs/constants.py`**: `SETTING_*` 키, `FORMAT_*` 토큰, `DEFAULT_ENCODING='utf-8'`.
- **소형 위젯**: `ToolBar`/`ToolButton`(텍스트-아이콘 하단 툴바), `ZoomWidget`/`LightWidget`(QSpinBox), `HashableQListWidgetItem`(dict 키용 해시 가능 리스트 아이템).

## 리소스 파이프라인

아이콘 PNG와 다국어 `strings*.properties`는 `resources.qrc` 매니페스트에 묶여 `pyrcc5 -o libs/resources.py resources.qrc`(= `make qt5py3`)로 `libs/resources.py`에 임베드된다. 런타임에 `MainWindow`는 아이콘을, `StringBundle`은 문자열을 `:/...` 경로로 읽는다. → [../how-to/install-and-build.md](../how-to/install-and-build.md)

## 진입점

`main()`(`labelImg.py:2436`) → `get_main_app()`(`labelImg.py:2401`)가 `QApplication`을 만들고(이미 살아 있는 인스턴스가 있으면 재사용 — 테스트에서 여러 번 호출 가능) `argparse`로 `image_dir`/`class_file`/`save_dir`를 파싱한 뒤 `MainWindow`를 생성·`show()`한다. 콘솔 스크립트 진입점은 `setup.py`의 `labelImg=labelImg.labelImg:main`. 단, `class_file` 인자는 그대로 사용되지 않고 `MainWindow.__init__`(`labelImg.py:79`) 안의 `labelImg.py:142`에서 `get_persistent_classes_file`을 거친다 — frozen exe에서는 exe 옆 영속 사본으로 대체되고, 소스 실행에서는 인자 경로가 영속 파일로 사용되며 부재 시 자동 생성된다.
