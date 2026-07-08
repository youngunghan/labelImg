# 레퍼런스: 모듈 인벤토리

파일별 핵심 클래스/함수. 라인 번호는 현재 저장소(`D:\labelImg`) 기준이다.

## `labelImg.py` (2113줄) — 앱 본체

| 심볼 | 위치 | 역할 |
|---|---|---|
| `WindowMixin` | `:54` | `QMainWindow`에 `menu()`/`toolbar()` 생성 헬퍼 믹스인 |
| `MainWindow` | `:73` | 앱 컨트롤러(God-object). `QMainWindow`+`WindowMixin` 다중상속 |
| `inverted(color)` | `:2059` | RGB 반전 `QColor` |
| `read(filename, default)` | `:2063` | `QImageReader`(autoTransform=EXIF 회전)로 `QImage` 읽기 |
| `get_main_app(argv)` | `:2072` | `QApplication`(기존 인스턴스 재사용)+argparse(image_dir/class_file/save_dir)+`MainWindow` |
| `main()` | `:2107` | 엔트리포인트(`get_main_app` → `exec_`) |

`MainWindow` 주요 메서드:

| 메서드 | 위치 | 역할 |
|---|---|---|
| `__init__` | `:76` | Settings 로드, 위젯/액션/메뉴/툴바/도크 전체 구성, Canvas 시그널 연결 |
| `set_format` / `change_format` | `:589` / `:608` | 포맷 표시 갱신 / VOC→YOLO→CreateML 순환 |
| `set_dirty` / `set_clean` / `toggle_actions` | `:656` / `:660` / `:665` | 저장 필요 상태·이미지 의존 액션 활성화 |
| `reset_state` | `:680` | 도형/라벨리스트/캔버스 초기화 |
| `add_label` / `remove_label` / `load_labels` | `:852` / `:865` / `:875` | Shape↔리스트 항목 동기화 |
| `save_labels` | `:916` | 포맷별 `LabelFile.save_*_format` 호출(입출력 디스패처) |
| `new_shape` | `:998` | `newShape` 슬롯; LabelDialog 팝업 후 라벨 부여 |
| `shape_selection_changed` / `label_selection_changed` / `label_item_changed` | `:837` / `:978` / `:987` | 선택/체크 상태 상호 동기화 |
| `zoom_request` / `light_request` / `scroll_request` / `paint_canvas` | `:1053` / `:1105` / `:1037` / `:1256` | 줌·밝기·스크롤·스케일·페인트 |
| `load_file` | `:1133` | 이미지/라벨파일 판별·로드, `canvas.load_pixmap`, 어노테이션 자동 로드 |
| `show_bounding_box_from_annotation_file` | `:1220` | xml > txt > json 우선순위로 어노테이션 자동 탐색 |
| `open_dir_dialog` | `:1383` | 디렉터리 열기 진입점(시작 시 `__init__` `:577`에서 `dir_path`+`silent=True`로 호출). (로컬 수정 2026-07-03) `dir_path` 인자가 명시되면 설정 pkl의 `lastOpenDir`보다 우선(`:1390-1395`) · 다이얼로그 취소 시 상태 불변 조기 반환(`:1402-1404`) · `default_save_dir`를 임포트 전에 확정 — 시작 시 명령줄 `save_dir`가 있으면 그것을 유지, 없으면 연 폴더(`:1410-1413`) |
| `import_dir_images` / `scan_all_images` / `open_next_image` / `open_prev_image` | `:1416` / `:1323` / `:1474` / `:1449` | 디렉터리 스캔·이미지 탐색 |
| `save_file` / `save_file_as` | `:1519` / `:1534` | 저장 경로 결정 → `save_labels` |
| `classify_current_image` / `undo_classify` / `create_classify_actions` / `edit_classify_categories` / `rebuild_classify_actions` | `:1585` / `:1726` / `:1765` / `:1778` / `:1837` | (로컬 확장) 사용자 정의 카테고리(기본 g→good·b→bad, `SETTING_CLASSIFY_TARGETS`) 단축키로 현재 이미지+라벨(.xml/.txt/.json)을 형제 폴더 `<폴더>_<이름>`으로 이동 후 다음 이미지 표시, Ctrl+Z로 직전 분류 원복. File > Edit Classify Categories로 카테고리를 라이브 편집(메뉴/단축키 재구성, 재시작 불필요). 라벨은 `default_save_dir` 우선, 없으면 이미지 옆에서 탐색(라벨 없이 이미지만 이동되면 상태바 경고). 내부 헬퍼 `_cleanup_dest_dir` `:1617` / `_taken` `:1634` / `_rollback` `:1647` |
| `get_persistent_classes_file` / `edit_default_classes` / `reload_predefined_classes` | `:1939` / `:1959` / `:1978` | (로컬 확장) `predefined_classes.txt` 영속 경로 결정(frozen exe면 exe 옆 파일, 없으면 번들 기본으로 부트스트랩) · Edit Default Classes 다이얼로그(Ctrl+Shift+E, 저장 후 리로드) · 클래스 목록 리로드(`label_hist`·`default_label`·`LabelDialog`·기본라벨 콤보 갱신) |
| `load_pascal_xml_by_filename` / `load_yolo_txt_by_filename` / `load_create_ml_json_by_filename` | `:1993` / `:2006` / `:2029` | 각 Reader 인스턴스화 → `load_labels`. YOLO는 로드 실패(`YoloParseError` 등)를 에러 대화상자로 처리하고 불량 라인 수를 상태바에 표시(`:2012-2024`) |
| `closeEvent` | `:1285` | 모든 `SETTING_*` 키 기록 후 `Settings.save()` |

주의: 소스가 수정되면 (특히 후반부) 라인 번호가 밀릴 수 있다 — 정확한 위치는 심볼명 검색으로 재확인한다.

## `libs/canvas.py` (748줄) — `Canvas(QWidget)`

드로잉/편집 캔버스. 시그널 7종: `zoomRequest`,`lightRequest`,`scrollRequest`,`newShape`,`selectionChanged`,`shapeMoved`,`drawingPolygon`(`:24-31`). 모드 `CREATE,EDIT`(`:33`), `epsilon=24.0`(정점 근접).

주요 메서드: `mouseMoveEvent`(`:111`)·`mousePressEvent`(`:258`)·`mouseReleaseEvent`(`:278`)·`handle_drawing`(`:322`)·`finalise`(`:574`)·`bounded_move_vertex`(`:400`)·`bounded_move_shape`(`:436`)·`select_shape`/`select_shape_point`(`:355`/`:363`)·`paintEvent`(`:495`)·`transform_pos`(`:557`)·`wheelEvent`(`:605`)·`keyPressEvent`(`:629`)·`load_pixmap`(`:708`)·`load_shapes`(`:713`).

## `libs/shape.py` (209줄) — `Shape`

박스 1개. 상수 `P_SQUARE/P_ROUND`(`:24`), `MOVE_VERTEX/NEAR_VERTEX`(`:26`). 클래스 변수 색상·`point_size=16`·`scale`·`label_font_size=8`(`:28-39`).

메서드: `add_point`(`:72`)·`pop_point`(`:76`)·`close`(`:64`)·`is_closed`(`:81`)·`set_open`(`:84`)·`reach_max_points`(`:67`, `len>=4`)·`paint`(`:87`)·`draw_vertex`(`:137`)·`nearest_vertex`(`:155`)·`contains_point`(`:164`)·`make_path`(`:167`)·`bounding_rect`(`:173`)·`move_by`(`:176`)·`move_vertex_by`(`:179`)·`highlight_vertex`/`highlight_clear`(`:182`)·`copy`(`:189`)·`__len__/__getitem__/__setitem__`(`:202`).

## `libs/labelFile.py` (174줄)

| 심볼 | 위치 | 역할 |
|---|---|---|
| `LabelFileFormat(Enum)` | `:18` | `PASCAL_VOC=1`, `YOLO=2`, `CREATE_ML=3` |
| `LabelFileError(Exception)` | `:24` | 라벨 파일 오류 |
| `LabelFile`(`libs/labelFile.py` 174줄) | `:28` | 포맷별 writer 위임 파사드. `suffix`(클래스변수, 기본 `.xml`) |
| `LabelFile.save_pascal_voc_format` | `:54` | `PascalVocWriter` 위임 |
| `LabelFile.save_yolo_format` | `:84` | `YOLOWriter` 위임(+`class_list`) |
| `LabelFile.save_create_ml_format` | `:39` | `CreateMLWriter` 위임(+`class_list`) |
| `LabelFile.toggle_verify` | `:114` | `verified` 반전 |
| `LabelFile.is_label_file` | `:147` | 확장자 == `suffix` 판별(staticmethod) |
| `LabelFile.convert_points_to_bnd_box` | `:152` | points → `(xmin,ymin,xmax,ymax)`, min좌표 1 클램프(staticmethod) |

## I/O 모듈

| 파일 | 클래스 | 위치 |
|---|---|---|
| `libs/pascal_voc_io.py` (171줄) | `PascalVocWriter` / `PascalVocReader` | `:15` / `:127` |
| `libs/yolo_io.py` (171줄) | `YoloParseError` / `YOLOWriter` / `YoloReader` | `:12` / `:16` / `:86` |
| `libs/create_ml_io.py` (136줄) | `CreateMLWriter` / `CreateMLReader` | `:13` / `:96` |

상세 포맷 → [formats.md](formats.md).

## 설정·i18n·유틸

| 파일 | 심볼 | 역할 |
|---|---|---|
| `libs/settings.py` (45줄) | `Settings` (`:5`) | dict → `~/.labelImgSettings.pkl` pickle 영속화. `save`/`load`/`get`/`reset` |
| `libs/stringBundle.py` (78줄) | `StringBundle` (`:23`) | 로케일별 i18n 문자열. `get_bundle()` 팩토리, `get_string(id)` |
| `libs/ustr.py` (17줄) | `ustr(x)` (`:4`) | py2/py3 유니코드 강제(py3는 no-op) |
| `libs/utils.py` (117줄) | 헬퍼 함수군 | `new_action`/`new_button`/`new_icon`/`add_actions`/`label_validator`/`generate_color_by_text`/`natural_sort`/`distance`/`format_shortcut`/`Struct`/`trimmed`(QT4/5 strip 호환, `labelDialog`에서 사용)/`have_qstring`/`util_qt_strlistclass` |
| `libs/constants.py` (21줄) | 상수 | `SETTING_*` 키(분류 카테고리 `SETTING_CLASSIFY_TARGETS` 포함), `FORMAT_PASCALVOC/YOLO/CREATEML`, `DEFAULT_ENCODING='utf-8'` |

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
| `tools/label_to_csv.py` (215줄) | `txt2csv`(`:18`)/`xml2csv`(`:70`)/CLI(`:132`) — VOC/YOLO → AutoML CSV. → [../how-to/export-to-csv.md](../how-to/export-to-csv.md) |
| `Makefile` | `qt5py3`(pyrcc5 컴파일)·`testpy3`(unittest)·`clean`·`pip_upload` |
| `setup.py` | PyPI 패키징. deps `pyqt5`,`lxml`, 진입점 `labelImg.labelImg:main` |
| `resources.qrc` | 아이콘 + `strings*.properties`(en/zh-TW/zh-CN/ja-JP) 매니페스트 |
| `libs/__init__.py` | 버전 `1.8.6` |
| `tests/` | `test_io`·`test_settings`·`test_stringBundle`·`test_utils`·`test_qt`·`test_classify`(분류 원자성/롤백/undo/N분류)·`test_yolo_reader`(YOLO 견고성)·`test_create_ml_reader`(verified 매칭·utf-8) |
