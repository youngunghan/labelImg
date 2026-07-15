# 레퍼런스: 모듈 인벤토리

파일별 핵심 클래스/함수. 라인 번호는 현재 저장소(`D:\labelImg`) 기준이다.

## `labelImg.py` (2442줄) — 앱 본체

| 심볼 | 위치 | 역할 |
|---|---|---|
| `WindowMixin` | `:57` | `QMainWindow`에 `menu()`/`toolbar()` 생성 헬퍼 믹스인 |
| `MainWindow` | `:76` | 앱 컨트롤러(God-object). `QMainWindow`+`WindowMixin` 다중상속 |
| `inverted(color)` | `:2352` | RGB 반전 `QColor` |
| `read(filename, default)` | `:2356` | `QImageReader`(autoTransform=EXIF 회전)로 `QImage` 읽기 |
| `get_main_app(argv)` | `:2365` | `QApplication`(기존 인스턴스 재사용)+argparse(image_dir/class_file/save_dir)+`MainWindow` |
| `main()` | `:2400` | 엔트리포인트(`get_main_app` → `exec_`) |

`MainWindow` 주요 메서드:

| 메서드 | 위치 | 역할 |
|---|---|---|
| `__init__` | `:79` | Settings 로드, 위젯/액션/메뉴/툴바/도크 전체 구성(AI 메뉴 포함), `AssistController`/`InferenceService` 생성(`:444-446`), Canvas 시그널 연결 |
| `set_format` / `change_format` | `:626` / `:651` | 포맷 표시 갱신 / VOC→YOLO→CreateML→COCO 순환(`:651-661`) |
| `set_dirty` / `set_clean` / `toggle_actions` | `:726` / `:730` / `:735` | 저장 필요 상태·이미지 의존 액션 활성화. `toggle_actions`는 `onLoadActive` 루프 **다음에** `self.assist.refresh_actions()`를 호출한다(`:743-746`) — 이미지가 열렸다고 AI 액션까지 켜지는 건 아니고(백엔드 없으면 계속 꺼짐), 루프가 먼저 켜버린 것을 되돌린다 |
| `reset_state` | `:754` | 도형/라벨리스트/캔버스 초기화 + `self.assist.forget_suggestions()`(`:764-769`)로 AI 추적 상태도 정리 |
| `add_label` / `remove_label` / `load_labels` | `:932` / `:945` / `:963` | Shape↔리스트 항목 동기화. `remove_label`은 모든 제거를 `self.assist.discard_shape(shape)`에 보고한다(`:954-961`) — 사용자가 직접 지운 제안(suggestion)도 AI 레이어가 추적을 놓치지 않도록 |
| `save_labels` | `:1019` | 포맷별 `LabelFile.save_*_format` 호출(입출력 디스패처, COCO 분기 `:1058-1069`). **`shape.provisional`인 도형은 저장 직전 필터링**되어 어떤 포맷에도 기록되지 않는다(`:1040`) — Ctrl+S/Save As/Export COCO.../verify_image/자동저장이 모두 거치는 단일 choke point |
| `new_shape` | `:1120` | `newShape` 슬롯; LabelDialog 팝업 후 라벨 부여 |
| `shape_selection_changed` / `label_selection_changed` / `label_item_changed` | `:917` / `:1100` / `:1109` | 선택/체크 상태 상호 동기화 |
| `zoom_request` / `light_request` / `scroll_request` / `paint_canvas` | `:1175` / `:1227` / `:1159` / `:1388` | 줌·밝기·스크롤·스케일·페인트 |
| `load_file` | `:1255` | 이미지/라벨파일 판별·로드, `canvas.load_pixmap`, 어노테이션 자동 로드 |
| `show_bounding_box_from_annotation_file` | `:1342` | xml > txt > json 우선순위로 어노테이션 자동 탐색(json은 `load_json_by_filename`으로 CreateML/COCO 콘텐츠 스니핑) |
| `is_same_path` / `coco_dataset_target` | `:665` / `:670` | (포크, COCO) 대소문자 무시 경로 비교(staticmethod) · 이 세션이 병합해 쓸 COCO 데이터셋 json 경로 결정 — 명시적으로 고른 `coco_dataset_path` 우선, 없으면 `<save dir>/annotations.json`(save dir도 없으면 이미지 폴더), 그마저 없으면 `None`(`:670-687`) → [formats.md](formats.md) |
| `open_dir_dialog` | `:1541` | 디렉터리 열기 진입점(시작 시 `__init__` `:614`에서 `dir_path`+`silent=True`로 호출). (로컬 수정 2026-07-03) `dir_path` 인자가 명시되면 설정 pkl의 `lastOpenDir`보다 우선(`:1548-1553`) · 다이얼로그 취소 시 상태 불변 조기 반환(`:1560-1562`) · `default_save_dir`를 임포트 전에 확정 — 시작 시 명령줄 `save_dir`가 있으면 그것을 유지, 없으면 연 폴더(`:1568-1571`) |
| `import_dir_images` / `scan_all_images` / `open_next_image` / `open_prev_image` | `:1588` / `:1493` / `:1671` / `:1646` | 디렉터리 스캔·이미지 탐색(재귀 `os.walk` — COCO의 basename 충돌 경로가 여기서 발생) |
| `save_file` / `save_file_as` | `:1716` / `:1742` | 저장 경로 결정 → `save_labels` |
| `classify_current_image` / `undo_classify` / `create_classify_actions` / `edit_classify_categories` / `rebuild_classify_actions` | `:1805` / `:1964` / `:2003` / `:2016` / `:2075` | (로컬 확장) 사용자 정의 카테고리(기본 g→good·b→bad, `SETTING_CLASSIFY_TARGETS`) 단축키로 현재 이미지+라벨(.xml/.txt/.json)을 형제 폴더 `<폴더>_<이름>`으로 이동 후 다음 이미지 표시, Ctrl+Z로 직전 분류 원복. File > Edit Classify Categories로 카테고리를 라이브 편집(메뉴/단축키 재구성, 재시작 불필요). 라벨은 `default_save_dir` 우선, 없으면 이미지 옆에서 탐색(라벨 없이 이미지만 이동되면 상태바 경고). 내부 헬퍼 `_cleanup_dest_dir` `:1837` / `_taken` `:1854` / `_rollback` `:1867` |
| `get_persistent_classes_file` / `edit_default_classes` / `reload_predefined_classes` | `:2177` / `:2197` / `:2216` | (로컬 확장) `predefined_classes.txt` 영속 경로 결정(frozen exe면 exe 옆 파일, 없으면 번들 기본으로 부트스트랩) · Edit Default Classes 다이얼로그(Ctrl+Shift+E, 저장 후 리로드) · 클래스 목록 리로드(`label_hist`·`default_label`·`LabelDialog`·기본라벨 콤보 갱신) |
| `load_pascal_xml_by_filename` / `load_yolo_txt_by_filename` | `:2231` / `:2244` | 각 Reader 인스턴스화 → `load_labels`. YOLO는 로드 실패(`YoloParseError` 등)를 에러 대화상자로 처리하고 불량 라인 수를 상태바에 표시(`:2255-2264`) |
| `load_json_by_filename` / `load_create_ml_json_by_filename` / `load_coco_json_by_filename` | `:2267` / `:2279` / `:2292` | (포크, COCO) `load_json_by_filename`이 `is_coco_json`으로 콘텐츠를 먼저 스니핑해 COCO/CreateML 리더를 고른다(`:2268-2277`) → [formats.md](formats.md). `load_coco_json_by_filename`은 데이터셋에 이 이미지가 없으면(`found_image=False`) 포맷을 바꾸지 않고 `False`를 반환한다(`:2299-2317`) |
| `import_coco_dialog` / `export_coco_dialog` | `:2319` / `:2348` | (포크, COCO) File > Import/Export COCO... 핸들러 — 데이터셋 json을 골라 현재 이미지를 불러오거나(Import) 병합 저장(Export)하고, 고른 경로를 이후 저장의 타깃으로 고정한다(`coco_dataset_path`) |
| `closeEvent` | `:1417` | 대부분의 `SETTING_*` 키(창 크기/위치, 색상, 마지막 포맷 등) 기록 후 `Settings.save()`. `SETTING_MODEL_BACKEND`만 예외로 **조건부** 기록이다 — `self.assist.backend_name`이 실제로 설정된 값일 때만 쓰고, 아무것도 설정되지 않았으면 그 키를 아예 쓰지 않는다(레거시 키가 남아 있으면 지운다, `:1449-1472`) → [settings.md](settings.md) |

주의: 소스가 수정되면 (특히 후반부) 라인 번호가 밀릴 수 있다 — 정확한 위치는 심볼명 검색으로 재확인한다.

## `libs/canvas.py` (748줄) — `Canvas(QWidget)`

드로잉/편집 캔버스. 시그널 7종: `zoomRequest`,`lightRequest`,`scrollRequest`,`newShape`,`selectionChanged`,`shapeMoved`,`drawingPolygon`(`:24-31`). 모드 `CREATE,EDIT`(`:33`), `epsilon=24.0`(정점 근접).

주요 메서드: `mouseMoveEvent`(`:111`)·`mousePressEvent`(`:258`)·`mouseReleaseEvent`(`:278`)·`handle_drawing`(`:322`)·`finalise`(`:574`)·`bounded_move_vertex`(`:400`)·`bounded_move_shape`(`:436`)·`select_shape`/`select_shape_point`(`:355`/`:363`)·`paintEvent`(`:495`)·`transform_pos`(`:557`)·`wheelEvent`(`:605`)·`keyPressEvent`(`:629`)·`load_pixmap`(`:708`)·`load_shapes`(`:713`).

## `libs/shape.py` (246줄) — `Shape`

박스 1개. 상수 `P_SQUARE/P_ROUND`(`:24`), `MOVE_VERTEX/NEAR_VERTEX`(`:26`). 지오메트리 종류 `RECT/POLYGON/POINT/LINE`(포크, ML-assist 대비, `:31-34` — Phase 1은 사각형만 그리지만 나중 단계가 도형 종류를 재정의하지 않고 추가할 수 있게 상수만 미리 마련해둠). 클래스 변수 색상·`point_size=16`·`scale`·`label_font_size=8`(`:38-47`).

`__init__`(`:49-79`)이 받는 `shape_type` 인자와, 생성자에서 설정하는 두 AI 관련 필드(포크, ML-assist, `:57-64`):
- `provisional`(bool, 기본 `False`) — 모델이 제안한 도형(사용자 데이터 아님) 표시. `Shape.paint`가 점선 윤곽+반투명 채움으로 그리고(`:110-113, 132-138, 159-164`), `MainWindow.save_labels`가 저장 직전 이 값이 `True`인 도형을 걸러낸다(`labelImg.py:1040`) — Accept로 `False`가 되기 전까지는 어떤 어노테이션 파일에도 절대 기록되지 않는다.
- `confidence`(float|None, 기본 `None`) — 모델 신뢰도 점수. AI 메뉴의 임계값 슬라이더가 재추론 없이 이 값으로만 화면 표시를 다시 필터링한다(`libs/assist/controller.py`).
- `shape_type`(기본 `Shape.RECT`) — 도형 종류.

`copy()`(`:218-237`)는 필드 화이트리스트 방식이라 `provisional`/`confidence`/`shape_type`도 명시적으로 복제한다(`:234-236`) — 빠뜨리면 Ctrl+D로 복제된 제안이 조용히 "확정된" 박스가 되어 저장되는 버그가 된다는 주석이 코드에 달려 있다.

메서드: `add_point`(`:89`)·`pop_point`(`:93`)·`close`(`:81`)·`is_closed`(`:98`)·`set_open`(`:101`)·`reach_max_points`(`:84`, `len>=4`)·`paint`(`:104`)·`draw_vertex`(`:166`)·`nearest_vertex`(`:184`)·`contains_point`(`:193`)·`make_path`(`:196`)·`bounding_rect`(`:202`)·`move_by`(`:205`)·`move_vertex_by`(`:208`)·`highlight_vertex`/`highlight_clear`(`:211`)·`copy`(`:218`)·`__len__/__getitem__/__setitem__`(`:239`).

## `libs/labelFile.py` (205줄)

| 심볼 | 위치 | 역할 |
|---|---|---|
| `LabelFileFormat(Enum)` | `:19` | `PASCAL_VOC=1`, `YOLO=2`, `CREATE_ML=3`, `COCO=4`(포크 추가) |
| `LabelFileError(Exception)` | `:26` | 라벨 파일 오류 |
| `LabelFile` | `:30` | 포맷별 writer 위임 파사드. `suffix`(클래스변수, 기본 `.xml`) |
| `LabelFile.save_create_ml_format` | `:41` | `CreateMLWriter` 위임(+`class_list`). `convert_points_to_bnd_box`를 거치지 **않음**(float 좌표 그대로) |
| `LabelFile.save_coco_format` | `:56` | (포크) `COCOWriter` 위임 — `filename`은 **데이터셋 json**(이미지별 사이드카 아님). `convert_points_to_bnd_box`로 정수 클램프 후 `writer.save(target_file=filename, class_list=class_list)`로 read-modify-write 병합(`:82`) → [formats.md](formats.md) |
| `LabelFile.save_pascal_voc_format` | `:85` | `PascalVocWriter` 위임 |
| `LabelFile.save_yolo_format` | `:115` | `YOLOWriter` 위임(+`class_list`) |
| `LabelFile.toggle_verify` | `:145` | `verified` 반전 |
| `LabelFile.is_label_file` | `:178` | 확장자 == `suffix` 판별(staticmethod) |
| `LabelFile.convert_points_to_bnd_box` | `:183` | points → `(xmin,ymin,xmax,ymax)` 정수, min좌표 1 클램프(staticmethod). 호출부: COCO(`:79`)·VOC(`:109`)·YOLO(`:139`) — CreateML만 호출하지 않는다 |

## I/O 모듈

| 파일 | 클래스 | 위치 |
|---|---|---|
| `libs/pascal_voc_io.py` (171줄) | `PascalVocWriter` / `PascalVocReader` | `:15` / `:127` |
| `libs/yolo_io.py` (173줄) | `YoloParseError` / `YOLOWriter` / `YoloReader` | `:12` / `:16` / `:86` |
| `libs/create_ml_io.py` (136줄) | `CreateMLWriter` / `CreateMLReader` | `:13` / `:96` |
| `libs/coco_io.py` (339줄, 포크) | `COCOParseError` / `COCOWriter` / `COCOReader` | `:20` / `:120` / `:259` |

상세 포맷 → [formats.md](formats.md).

## `libs/inference/` (포크, ML-assist) — 모델 백엔드 코어, **의존성 0**

`import libs.inference`가 numpy/onnxruntime/Qt를 전혀 끌어오지 않는다는 것이 이 패키지의 불변식이다 — `libs/inference/__init__.py`(78줄)는 실제 서브모듈을 지연 임포트(PEP 562 `__getattr__`)로만 재노출한다(`:50-74`).

| 파일 | 심볼 | 역할 |
|---|---|---|
| `libs/inference/types.py` (149줄) | `Detection`(frozen dataclass, `:56`) / `Mask`(`:77`) / `SegPrompt`(`:91`) / `Prediction`(`:104`) / `least_confidence`(`:118`) | 모델 계층의 유일한 어휘. `Detection.box`는 **원본 이미지 픽셀** `(x1,y1,x2,y2)` — letterbox·줌·정규화를 전부 역변환한 뒤의 좌표(좌표 계약, `:10-28`). `least_confidence`는 상위 k 점수 평균의 여집합(액티브러닝 대비, 미구현) |
| `libs/inference/backend.py` (100줄) | `ModelBackend`(ABC, `:36`) / `MissingDependency`(`:26`) | 백엔드 플러그인 시드. `predict`(추상, `:57`)·`segment`/`embed`(기본 `NotImplementedError`, `:66-85`)·`close`(no-op 기본, `:87-92`). `supports_detection`/`supports_segmentation` 캐퍼빌리티 플래그로 UI가 액션을 켜고 끔 |
| `libs/inference/stub.py` (139줄) | `StubBackend`(`:61`) / `image_size`(`:34`) | 의존성 없는 결정론적 가짜 백엔드 — `predict`(`:105-139`)가 이미지 크기만의 순수 함수로 대각선을 따라 박스를 생성(설계 노트 `:62-74`), 테스트가 정확한 좌표를 assert할 수 있게 함 |
| `libs/inference/registry.py` (186줄) | `build_backend`(`:127`) / `available_backends`(`:118`) / `register_backend`(`:91`) / `DEFAULT_BACKEND=None`(`:39`) | 이름→백엔드 생성 테이블(`:85-88`: `stub`/`yolo_onnx`). `DEFAULT_BACKEND`가 `None`이라 설정에 `model/backend`가 없으면 **아무 백엔드도 자동 선택되지 않는다** — 기본 설치는 백엔드 미설정 상태(`:140-146`). `MissingDependency`/`ImportError`/그 외 예외를 전부 흡수해 `None`을 반환 — AI 없는 머신에서도 앱이 절대 죽지 않는다(`:163-186`) |
| `libs/inference/service.py` (280줄) | `InferenceService`(QObject, `:186`) / `RawImage`(`:56`) / `to_model_image`(`:113`) / `ThreadPoolExecutor`(`:142`) / `SynchronousExecutor`(`:157`, 테스트 전용) | 이 패키지에서 유일하게 Qt를 아는 모듈. **싱글 워커** `QThreadPool.maxThreadCount=1`(ONNX 세션 동시 호출 금지, `:146-148`) · UI 스레드에서 `QImage`를 순수 데이터로 변환 후 워커에 넘김(Qt 객체가 스레드를 넘지 않음, `to_model_image`, `:113-139`) · `predictionReady`/`predictionFailed` 시그널이 이미지 경로를 태깅해 결과-이미지 불일치(stale) 검사를 소비자에게 위임(`:194-197`) |
| `libs/inference/yolo_onnx.py` (984줄, Phase 2) | `YoloOnnxBackend`(`:606`) / `letterbox_params`/`inverse_letterbox`(`:167`/`:182`) / `nms`(`:235`, class-aware) / `detect_layout`(`:296`) / `decode_output`(`:374`) / `postprocess`(`:466`) | onnxruntime 기반 실제 YOLO 추론. 순수-Python 지오메트리(`_letterbox_geometry` `:129`가 유일한 소스여서 순변환/역변환이 서로 다른 값을 계산할 수 없음, `:31-57`의 설계 노트) · YOLOv5 `(1,N,5+nc)`/YOLOv8 `(1,4+nc,N)` 레이아웃을 클래스 수 일치→앵커축 휴리스틱 순서로 자동판별하고 모호하면 **예외로 실패**(`detect_layout`, `:296-371`) · 파이프라인 순서는 **decode→confidence filter→NMS→inverse-letterbox**(NMS는 모델-입력 좌표계에서, `postprocess`, `:466-518`) · `MAX_NMS_CANDIDATES=1000`(`:103`)가 안전판, 신뢰도 사전필터는 하지 않음(UI 슬라이더가 담당) · 클래스명은 config 오버라이드→ONNX 메타데이터 `names`→형제 `classes.txt`→`class_N` 순(`_resolve_class_names`, `:789-816`; config 오버라이드는 생성자, `:677`) |

## `libs/assist/` (포크, ML-assist) — 모델↔캔버스 seam

| 파일 | 심볼 | 역할 |
|---|---|---|
| `libs/assist/__init__.py` (20줄) | (문서만) | `libs.inference.Detection` → (`suggestion.py`) → `libs.shape.Shape(provisional)` 흐름 설명 |
| `libs/assist/suggestion.py` (77줄) | `detection_to_shape`(`:42`) / `detections_to_shapes`(`:61`) / `style_as_committed`(`:65`) / `PROVISIONAL_LINE_COLOR`/`PROVISIONAL_FILL_COLOR`(호박색, `:38-39`) | `Detection`↔`Shape` 순수 어댑터(MainWindow·캔버스·설정 몰라도 됨). 좌표는 그대로 캔버스에 옮겨진다 — 스케일 계산이 이 파일에 등장하면 버그(`:9-14`) |
| `libs/assist/controller.py` (1113줄) | `AssistController`(QObject, `:112`) | MainWindow가 AI를 위임하는 유일한 객체. 액션: **Auto-label Image=Ctrl+I**(`:256-259`)·**Accept All=Ctrl+Return**(`:260-263`)·**Reject All=Ctrl+Backspace**(`:264-267`)+신뢰도 슬라이더(`_create_threshold_action`, `:296-328`). `on_prediction_ready`(`:468`)가 `_is_current`(`:498`)로 **stale 결과를 드롭**(이미지 전환 후 늦게 도착한 예측이 엉뚱한 이미지에 박히는 것 방지) · `provisional_shapes()`(`:540`)는 `_shapes` 캐시가 아니라 **캔버스를 직접 읽음**(Ctrl+D 복제나 수동 삭제로 캔버스와 내부 상태가 어긋날 수 있다는 주석, `:540-560`) · `accept_all`/`reject_all`(`:618`/`:639`). **설정 읽기 시 레거시 `'stub'` 무시**: `__init__`(`:114-133`)이 `SETTING_MODEL_BACKEND`가 `'stub'`이면(구 `DEFAULT_BACKEND`가 `'stub'`이던 시절 저장된 값) 미설정으로 취급한다(`_LEGACY_IMPLICIT_DEFAULT_BACKEND`, `:94-109`) → [settings.md](settings.md). **능동학습 (Phase 4, 신규)**: `score_folder`/`cancel_batch_scoring`(`:719-782`)가 `m_img_list`를 한 번에 한 이미지씩 배치 채점하고, `_is_batch_result`(`:509-538`)가 그 결과를 대화형 stale-drop과 분리해서 처리한다 · `sort_by_uncertainty`/`restore_original_order`/`_reorder`(`:869-981`)가 `m_img_list`를 재정렬하고 현재 선택을 보존한다 · `refresh_file_list`(`:1025-1047`)가 파일 목록에 순위/점수를 표시한다 |

관련 → [shortcuts.md](shortcuts.md) · [settings.md](settings.md).

## 설정·i18n·유틸

| 파일 | 심볼 | 역할 |
|---|---|---|
| `libs/settings.py` (45줄) | `Settings` (`:5`) | dict → `~/.labelImgSettings.pkl` pickle 영속화. `save`/`load`/`get`/`reset` |
| `libs/stringBundle.py` (78줄) | `StringBundle` (`:23`) | 로케일별 i18n 문자열. `get_bundle()` 팩토리, `get_string(id)` |
| `libs/ustr.py` (17줄) | `ustr(x)` (`:4`) | py2/py3 유니코드 강제(py3는 no-op) |
| `libs/utils.py` (117줄) | 헬퍼 함수군 | `new_action`/`new_button`/`new_icon`/`add_actions`/`label_validator`/`generate_color_by_text`/`natural_sort`/`distance`/`format_shortcut`/`Struct`/`trimmed`(QT4/5 strip 호환, `labelDialog`에서 사용)/`have_qstring`/`util_qt_strlistclass` |
| `libs/constants.py` (27줄) | 상수 | `SETTING_*` 키(분류 카테고리 `SETTING_CLASSIFY_TARGETS`, ML-assist `SETTING_MODEL_BACKEND`/`SETTING_MODEL_PATH`/`SETTING_CONF_THRESHOLD` `:24-26` 포함), `FORMAT_PASCALVOC/YOLO/CREATEML/COCO`(`FORMAT_COCO` 포크 추가 `:18`), `DEFAULT_ENCODING='utf-8'` → [settings.md](settings.md) |

## 소형 위젯

| 파일 | 클래스 | 역할 |
|---|---|---|
| `libs/labelDialog.py` (95줄) | `LabelDialog` (`:14`) | 라벨 입력 모달(validator+completer+기존라벨 리스트) |
| `libs/combobox.py` (33줄) | `ComboBox` (`:15`) | `QComboBox` 래퍼(라벨 필터). `update_items` |
| `libs/default_label_combobox.py` (27줄) | `DefaultLabelComboBox` (`:15`) | "use default label" 모드의 기본 라벨 선택(single-class 모드와 별개) |
| `libs/colorDialog.py` (37줄) | `ColorDialog` (`:12`) | 선/채움 색 선택(알파+Restore Defaults) |
| `libs/toolBar.py` (39줄) | `ToolBar`/`ToolButton` (`:10`/`:30`) | 텍스트-아이콘 하단 좌측 툴바 |
| `libs/zoomWidget.py` (26줄) | `ZoomWidget` (`:10`) | 줌 % QSpinBox(1~500) |
| `libs/lightWidget.py` (33줄) | `LightWidget` (`:10`) | 밝기 % QSpinBox(0~100), `color()` 오버레이 |
| `libs/hashableQListWidgetItem.py` (28줄) | `HashableQListWidgetItem` (`:22`) | dict 키용 해시 가능 리스트 아이템 |

## 도구·빌드

| 파일 | 내용 |
|---|---|
| `tools/label_to_csv.py` (216줄) | `txt2csv`(`:18`)/`xml2csv`(`:70`)/CLI(`:132`) — VOC/YOLO → AutoML CSV. → [../how-to/export-to-csv.md](../how-to/export-to-csv.md) |
| `Makefile` | `qt5py3`(pyrcc5 컴파일)·`testpy3`(unittest)·`clean`·`pip_upload` |
| `setup.py` | PyPI 패키징. `REQUIRES_PYTHON='>=3.7'`(포크에서 상향 — `libs.inference`/`libs.assist`가 `from __future__ import annotations`·dataclasses를 쓰므로 3.6 이하는 임포트 자체가 실패, `setup.py:12-17`). 필수 deps `pyqt5`,`lxml`(그대로); 선택 익스트라 `EXTRA_DEP={'ai': ['onnxruntime>=1.15', 'numpy']}` → 이 이름이 `pip install labelImg[ai]`로 매핑되는 것은 PyPI에 배포된 패키지 기준이며, 이 포크는 그 이름으로 배포되어 있지 **않으므로** 로컬 체크아웃에서 `pip install -e ".[ai]"`로 설치해야 한다(`setup.py:19-28`). 진입점 `labelImg.labelImg:main` |
| `resources.qrc` | 아이콘 + `strings*.properties`(en/zh-TW/zh-CN/ja-JP) 매니페스트 |
| `libs/__init__.py` | 버전 `1.8.6` |
| `data/models/README.md` (포크) | AI 백엔드용 `.onnx` 안내. **모델 가중치는 번들되지 않음**(Ultralytics YOLOv5/v8 가중치는 AGPL-3.0이라 MIT 앱과 라이선스 충돌) — 사용자가 직접 `.onnx`를 공급. YOLOX(Apache-2.0) 등 permissive 대안 소개 |
| `tests/` (8→13개 파일, 30→249개 테스트) | `test_io`·`test_settings`·`test_stringBundle`·`test_utils`·`test_qt`·`test_classify`(분류 원자성/롤백/undo/N분류)·`test_yolo_reader`(YOLO 견고성)·`test_create_ml_reader`(verified 매칭·utf-8)·`test_coco_io`(포크)·`test_inference_core`(포크, 의존성 0 코어)·`test_assist`(포크, `AssistController`)·`test_yolo_onnx`(포크, onnxruntime 필요 — 없으면 skip)·`test_doc_citations`(포크, docs/**/*.md·README.rst·FORK_CHANGES.md의 `file.py:NNN` 인용을 실제 소스와 대조하는 회귀 방지 린트 — 의존성 0). 기본 설치(pyqt5+lxml만)에서도 전부 green — onnxruntime/numpy가 필요한 테스트는 **skip이지 error가 아님**. CI는 코어 매트릭스(pyqt5+lxml만) + 별도 `test-ai` job(`[ai]` 익스트라 설치 후 ONNX 테스트까지 실행)로 두 경로를 모두 검증(`.github/workflows/ci.yml`) |
