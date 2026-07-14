# How-to: 추가 기능 (분류·복사·클래스 편집·보기 모드)

라벨링 핵심 외에 이 저장소에 있는 부가 기능들. 일부(이미지 분류·영속 클래스 편집)는 **이 포크에서 추가된 커스텀 기능**으로 업스트림 README에는 없다.

## ⚠️ 이미지 분류 Good/Bad (`g` / `b` / `Ctrl+Z`) — 파일을 이동함

> **주의: 파괴적 동작.** `g`/`b`는 수정자 없는 단일 키라 실수로 눌리기 쉽다. 누르는 즉시 **현재 이미지와 그 라벨 파일이 디스크에서 다른 폴더로 이동**한다.

데이터셋을 빠르게 선별(검수)할 때, 현재 이미지를 좋은/나쁜 더미로 분류한다(`classify_current_image`, `labelImg.py:1752`):

- `g` (Classify Good) → 현재 이미지 + 라벨을 `<연 폴더>_good/` 으로 이동
- `b` (Classify Bad) → `<연 폴더>_bad/` 으로 이동
- `Ctrl+Z` (Undo Classify) → 가장 최근 이동을 원위치로 되돌림(`undo_classify`, `labelImg.py:1909`)

분류 동작들은 File 메뉴(카테고리별 `Classify <이름> (<키> → _<이름>)` / Undo Classify / Edit Classify Categories, `labelImg.py:500-503`)에도 있으며, `g`/`b`는 이미지가 로드되기 전에는 비활성이다(`onLoadActive`, `labelImg.py:468-471`).

동작 상세:
- **`Open Dir`로 폴더를 먼저 열어야** 한다(아니면 상태바에 안내만 뜨고 아무 일도 안 함).
- **현재 이미지가 열린 디렉터리 목록(`m_img_list`)에 속할 때만 동작**한다(`labelImg.py:1764-1766`). `Open File`로 목록 밖 단일 파일을 연 상태에서는 차단된다 — 예전 `last_open_dir` 값 기준의 엉뚱한 `_good`/`_bad` 폴더로 이동하는 사고를 막기 위한 가드.
- 대상 폴더는 연 폴더의 **형제** 폴더 `os.path.dirname(연폴더) + "/<연폴더이름>_good"`(또는 `_bad`)에 자동 생성된다.
- 이미지(`shutil.move`)와 **대응 라벨(.xml/.txt/.json)** 을 같은 stem으로 함께 옮긴다. 라벨은 기본 저장 폴더(`default_save_dir`)에서 먼저 찾고, 없으면 이미지 옆 폴더에서도 찾는다. 라벨을 못 찾아 이미지만 옮겨지면 상태바에 "no label file found to move" 경고가 표시된다.
  - ✅ **원자적**: 라벨 이동이 하나라도 실패하면(권한/잠금 등) 이미 옮긴 이미지·라벨을 모두 원위치로 되돌리고(`_rollback`) "Classify failed" 경고를 띄운다(`labelImg.py:1865-1884`). 따라서 이미지만 옮겨지고 라벨이 원래 폴더에 남는 어긋남은 생기지 않는다. 되돌리기까지 실패하면(드묾) "일부 파일이 남았을 수 있으니 확인하라"고 명시적으로 알리고, 이번에 새로 만든 빈 `_good`/`_bad` 폴더는 정리한다.
  - ⚠️ **YOLO 주의**: 분류는 `<stem>.txt`만 옮기고 공용 `classes.txt`는 옮기지 않는다 — `_good`/`_bad` 폴더의 YOLO 라벨을 단독으로 쓰려면 `classes.txt`를 따로 복사해야 한다.
- 대상 폴더에 같은 이름이 있으면 `_1`, `_2` … 로 충돌을 피한다(dedup).
- 미저장 변경이 있으면 저장/폐기/취소를 먼저 묻는다.
- 이동 후 디렉터리를 다시 스캔하고 다음 이미지로 넘어간다.
- `classify_history`에 이동 기록이 쌓여 `Ctrl+Z`로 순서대로 되돌릴 수 있다.
- `Ctrl+Z` 성공 시 디렉터리를 다시 스캔하고 복원된 이미지를 자동으로 다시 연다(`labelImg.py:1939-1946`).
- 되돌리기 도중 일부 파일 복원이 실패하면 "Undo failed" 경고가 뜨고 이동 기록은 보존되므로, 원인(잠금/권한)을 해소한 뒤 `Ctrl+Z`를 다시 누르면 남은 파일부터 이어서 복원된다(`undo_classify`, `labelImg.py:1921-1932`).

판정은 **자동이 아니라** 사용자가 누른 키로만 정해진다.

### 카테고리는 사용자 정의 (2026-07-08)

g/good·b/bad는 **기본값**일 뿐이다. **File > Edit Classify Categories**에서 한 줄에
`<단축키> <폴더이름>` 형식으로 원하는 N개 카테고리(예: `k keep` / `x trash` / `r review`)를
정의하면 메뉴·단축키가 **재시작 없이** 재구성되고(`rebuild_classify_actions`,
`labelImg.py:2020`), 설정 pkl(`classifyTargets`)에 영구 저장된다. 중복 단축키는
거부된다. 이동 대상은 항상 `<현재폴더>_<이름>` 형제 폴더이고, 원자성·`Ctrl+Z` undo는
카테고리 수와 무관하게 동일하게 동작한다.

## 이전 이미지의 박스 복사 (`Ctrl+V`)

`Copy Previous Bounding Boxes`(`copy_previous_bounding_boxes`, `labelImg.py:2316`)는 **바로 이전 이미지의 어노테이션을 현재 이미지에 그대로 올리고 즉시 저장**한다. 연속 프레임/영상 캡처처럼 객체 위치가 거의 같은 이미지를 빠르게 라벨링할 때 유용하다. **File 메뉴**에 등록돼 있다(`labelImg.py:501`). **`Open Dir`로 연 이미지 목록 안에서만 동작**한다(목록의 이전 이미지를 참조하므로) — `Open File`로 단독으로 연 파일에서는 조용히 무시된다(현재 파일이 `m_img_list`에 없으면 그대로 반환하는 가드, `labelImg.py:2317-2318`).

## 사전 정의 클래스 앱에서 편집 (`Ctrl+Shift+E`)

`File → Edit Default Classes`(`Ctrl+Shift+E`, `edit_default_classes`, `labelImg.py:2142`)는 여러 줄 입력창을 띄워 클래스 목록을 받고 **`predefined_classes.txt`를 영구적으로 덮어쓴다**. 저장 후 라벨 히스토리·`w` 라벨 입력창·use-default-label 콤보박스를 즉시 갱신한다. 파일을 직접 편집하지 않고 앱 안에서 클래스를 관리할 수 있다. (메뉴 위치는 File, `labelImg.py:503`.)

> **frozen(exe)에서의 클래스 파일**: 빌드된 실행파일에서는 항상 **실행파일 옆의 쓰기 가능한 `predefined_classes.txt`** 를 우선 사용한다(`get_persistent_classes_file`, `labelImg.py:2122-2140`). 이 영속 파일이 한 번 생기면 **CLI 두 번째 인자 `[PRE-DEFINED CLASS FILE]` 는 이후 무시**되고(최초 1회 시드로만 쓰임), 클래스 변경은 Edit Default Classes 또는 그 파일 직접 편집으로 한다. 소스(비 frozen)로 실행할 때는 CLI 인자가 그대로 적용된다.

## 고급 모드 (`Ctrl+Shift+A`)

`View → Advanced Mode`(`toggle_advanced_mode`, `labelImg.py:692`, 체크형)는 초보/고급 UI를 전환한다. 고급 모드는 **Create RectBox / Edit 모드를 분리된 액션으로 노출**하고 도크 패널을 더 자유롭게 다룰 수 있게 한다. [explanation/canvas-interaction-model.md](../explanation/canvas-interaction-model.md)에서 말하는 CREATE/EDIT 모드가 UI에 드러나는 통로다.

## 밝기 조절 (LightWidget, `Ctrl+Shift++` / `-` / `=`)

툴바의 **밝기 스핀박스(0~100%)** 와 `Ctrl+Shift++`(밝게)/`Ctrl+Shift+-`(어둡게)/`Ctrl+Shift+=`(리셋), `Ctrl+Shift+휠`로 어두운 이미지를 밝게 보며 검수할 수 있다. **비파괴적**이다 — 화면 오버레이(`canvas.overlay_color`)만 바꾸고 저장 픽셀은 건드리지 않는다(`LightWidget.color()` → `paint_canvas`, `labelImg.py:1388`). 기본값 50%는 "변화 없음". 밝기 스핀박스는 기본(초보) 모드 툴바에만 표시된다 — 고급 모드에서는 툴바에서 사라지므로 단축키나 View 메뉴로 조절한다(`labelImg.py:525-533`).

## 박스 색 변경 (`Ctrl+L`)

`Edit → Box Line Color`(`Ctrl+L`, `choose_color1`)는 색상 다이얼로그(알파 채널·Restore Defaults)로 박스 선 색을 바꾸고 설정에 영속화한다(`SETTING_LINE_COLOR`). 선택한 박스의 선/채움 색은 고급 모드의 Shape Line/Fill Color 액션으로 따로 바꿀 수 있다.

## AI 자동 라벨링 (`&AI` 메뉴)

모델이 이미지를 보고 박스를 **제안**하고, 사람이 받아들이거나 버리는 기능이다(이 포크에서 추가). 새
**AI 메뉴**에 `Auto-label Image`(`Ctrl+I`)·`Accept All Suggestions`(`Ctrl+Return`)·
`Reject All Suggestions`(`Ctrl+Backspace`)와 신뢰도 임계값 슬라이더가 있다(`AssistController`,
`libs/assist/controller.py`). 제안은 점선/반투명으로 표시되며 **받아들이기 전까지는 저장되지 않는다**.
기본 설치는 백엔드가 전혀 설정되지 않은 상태라 **메뉴 자체가 비활성화**돼 있다(설치/설정 안내
툴팁만 뜬다) — 실제 ONNX 모델로 검출하려면 이 저장소 루트에서 `pip install -e ".[ai]"`(이 포크는
PyPI 미배포이므로 `pip install labelImg[ai]`는 무관한 업스트림 패키지를 받는다)와 모델 백엔드/경로
설정이 모두 필요하다. 전체 절차 → [auto-label.md](auto-label.md).

## 보기/탐색

- **Fit Window** `Ctrl+F` · **Fit Width** `Ctrl+Shift+F` · **원본 크기(100%)** `Ctrl+=` (`labelImg.py:371-378`).
- **모든 박스 숨기기/보이기** `Ctrl+H` / `Ctrl+A` (`toggle_polygons`).
- **최근 파일** `File → Open Recent` — 최근 연 파일 최대 7개(세션 간 영속, `SETTING_RECENT_FILES`).

전체 단축키 → [../reference/shortcuts.md](../reference/shortcuts.md). 검증/플래그/모드 → [verify-and-difficult.md](verify-and-difficult.md).
