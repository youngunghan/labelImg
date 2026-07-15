# 레퍼런스: 설정

labelImg는 UI 상태를 **사용자 홈 디렉터리의 `~/.labelImgSettings.pkl`** 에 pickle로 저장한다. 구현은 `libs/settings.py`의 `Settings` 클래스.

## `Settings` 동작

| 메서드 | 위치 | 동작 |
|---|---|---|
| `__init__` | `settings.py:6-10` | `data={}`, `path = ~/.labelImgSettings.pkl` |
| `settings[key] = v` / `settings[key]` | `:12-16` | dict 유사 set/get |
| `get(key, default)` | `:18-21` | 키 없으면 `default` |
| `save()` | `:23-28` | `pickle.HIGHEST_PROTOCOL`로 `data`를 `path`에 dump |
| `load()` | `:30-38` | `path` 있으면 복원. **모든 예외를 삼키고** 메시지만 출력 |
| `reset()` | `:40-45` | pkl 파일 삭제, `data={}`, `path=None` |

- 시작 시 `MainWindow.__init__`이 `Settings().load()`로 메모리에 올린다.
- 종료 시 `closeEvent`(`labelImg.py:1431`)가 아래 키를 모두 기록한 뒤 `save()`한다.

> ⚠️ `load()`가 손상된 pkl을 조용히 무시하므로(빈 설정으로 동작), 설정이 이상하면 파일을 지우는 게 가장 확실하다 → [../how-to/reset-and-troubleshoot.md](../how-to/reset-and-troubleshoot.md).
> ⚠️ `reset()` 이후 `path=None`이라 그 세션의 이후 `save()`는 저장되지 않는다(`settings.py:24,45`).

## 저장되는 키 (`libs/constants.py`)

| 상수 | 키 문자열 | 의미 |
|---|---|---|
| `SETTING_FILENAME` | `filename` | 마지막 파일명 |
| `SETTING_RECENT_FILES` | `recentFiles` | 최근 파일 목록 |
| `SETTING_WIN_SIZE` | `window/size` | 창 크기 |
| `SETTING_WIN_POSE` | `window/position` | 창 위치 |
| `SETTING_WIN_STATE` | `window/state` | 도크/툴바 상태 |
| `SETTING_LINE_COLOR` | `line/color` | 박스 선 색 |
| `SETTING_FILL_COLOR` | `fill/color` | 박스 채움 색 |
| `SETTING_ADVANCE_MODE` | `advanced` | 고급 모드 on/off |
| `SETTING_SAVE_DIR` | `savedir` | 기본 저장 디렉터리 |
| `SETTING_LAST_OPEN_DIR` | `lastOpenDir` | 마지막 연 디렉터리 |
| `SETTING_PAINT_LABEL` | `paintlabel` | 박스 위 라벨 표시 |
| `SETTING_AUTO_SAVE` | `autosave` | 자동 저장 on/off |
| `SETTING_SINGLE_CLASS` | `singleclass` | single-class(기본 라벨) 모드 |
| `SETTING_DRAW_SQUARE` | `draw/square` | 정사각형 그리기 |
| `SETTING_LABEL_FILE_FORMAT` | `labelFileFormat` | 마지막 사용 포맷(VOC/YOLO/CreateML/COCO) |
| `SETTING_CLASSIFY_TARGETS` | `classifyTargets` | (포크, 2026-07-08) 분류 카테고리 `(단축키, 폴더이름)` 쌍 목록 — 기본 `[('g','good'),('b','bad')]`. Edit Classify Categories 저장 시 즉시 `save()`되며, closeEvent가 개별 기록하지 않아도 로드된 dict에 남아 종료 시에도 보존된다 |
| `SETTING_MODEL_BACKEND` | `model/backend` | (포크, ML-assist) 사용할 추론 백엔드 이름(`libs/inference/registry.py`의 등록 키, 예: `'stub'`/`'yolo_onnx'`). `DEFAULT_BACKEND`는 `'stub'`가 아니라 **`None`**이다(`registry.py:39`) — 설정 파일에 이 키가 없으면 **어떤 백엔드도 자동 선택되지 않는다.** 기본 설치는 백엔드 미설정 상태이고 AI 액션은 비활성으로 남는다(`AssistController.__init__`, `libs/assist/controller.py:119-133`). ⚠️ **레거시 `'stub'`은 미설정으로 취급된다**: `DEFAULT_BACKEND`가 `'stub'`이던 이전 빌드로 한 번이라도 종료한 적이 있으면 이 키에 `'stub'`이 그대로 박혀 있을 수 있는데(당시 `closeEvent`가 조건 없이 기록했다), `AssistController.__init__`이 이 값을 읽을 때마다 미설정으로 취급하고 로그를 남긴다(`_LEGACY_IMPLICIT_DEFAULT_BACKEND`, `libs/assist/controller.py:94-109`) — 그렇지 않으면 실제 모델이 돈 적 없는데도 `StubBackend`의 가짜(이미지 크기에서 유도한) 검출 결과가 되살아난다. `'stub'`을 명시적으로 고르려면 인-프로세스로 (`AssistController.set_backend` 등) 직접 선택해야 한다 — pkl에 `'stub'`을 손으로 써넣는 것은 지원되지 않는다 |
| `SETTING_MODEL_PATH` | `model/path` | (포크, ML-assist) `yolo_onnx` 백엔드가 로드할 `.onnx` 파일 경로. 기본값 `None`(모델 미설정 → AI 액션 비활성) — `controller.py:131` |
| `SETTING_CONF_THRESHOLD` | `model/confThreshold` | (포크, ML-assist) AI 메뉴 슬라이더의 신뢰도 임계값(0.0~1.0로 클램프, `AssistController._sanitize_threshold`). 기본값 `DEFAULT_CONF_THRESHOLD=0.5`(`controller.py:46, 120-121`) |

세 키 모두 `constants.py:24-26`에 정의되어 있다. `closeEvent`가 `model_path`/`threshold`는 그대로 기록하지만(`labelImg.py:1471-1472`), `SETTING_MODEL_BACKEND`만은 **조건부**다: `self.assist.backend_name`이 실제로(사용자가 고른 값으로) 설정돼 있을 때만 쓰고, 아무것도 설정된 적이 없으면 그 키를 아예 쓰지 않을 뿐 아니라 이미 pkl에 남아 있는 레거시 값도 지운다(`labelImg.py:1449-1472`). 예전에는 `self.assist.backend_name`을 무조건 기록했는데, `DEFAULT_BACKEND`가 아직 `'stub'`이던 시절엔 이 무조건 기록이 바로 그 값을 사용자 pkl에 영구히 박아 넣은 원인이었다 — 위 `SETTING_MODEL_BACKEND` 행의 ⚠️ 참고.

> ⚠️ `SETTING_WIN_GEOMETRY`(`window/geometry`)는 `constants.py:5`에 **정의만** 되어 있고 `closeEvent`에서는 저장되지 않는다 — 실제로 기록되는 창 키는 size/position/state뿐이다(`labelImg.py:1427-1429`).

> ℹ️ **기록 조건**: `filename`은 단일 파일 모드에서만 실제 경로가 저장되고 폴더를 연 상태로 종료하면 `''`가 저장된다(`labelImg.py:1422-1425`). `savedir`·`lastOpenDir`도 종료 시점에 해당 경로가 존재하지 않으면 `''`로 기록된다(`labelImg.py:1434-1442`).

> ℹ️ **시작 시 우선순위 (로컬 수정, 2026-07-03)**: `labelImg.py 이미지폴더 [클래스파일] [저장폴더]` 형태로 명령줄 인자를 주고 시작하면, `__init__`(`labelImg.py:614`)이 `open_dir_dialog(dir_path=이미지폴더, silent=True)`를 호출하고 `open_dir_dialog`(`labelImg.py:1541-1553`)는 **명시된 `dir_path`를 pkl의 `lastOpenDir`보다 우선**한다. 다이얼로그 취소 시 상태 불변 조기 반환(`labelImg.py:1560-1562`). 이때 `default_save_dir`도 설정되는데, 명령줄 저장폴더 인자가 있으면 그것을 유지하고 없으면 연 폴더가 된다(`labelImg.py:1568-1571`). 따라서 pkl의 `savedir`·`lastOpenDir`는 그 세션에서 무시되고, 종료 시 새 값으로 갱신된다. pkl의 `savedir`는 명령줄 저장폴더 인자가 없을 때만 시작 시 적용된다(`labelImg.py:569-572`). (수정 전에는 pkl의 `lastOpenDir`가 명령줄 폴더보다 우선되어 라벨 XML을 찾지 못하는 문제가 있었다.)

## 기타 상수

- `FORMAT_PASCALVOC='PascalVOC'`, `FORMAT_YOLO='YOLO'`, `FORMAT_CREATEML='CreateML'`, `FORMAT_COCO='COCO'`(포크 추가, `constants.py:18`) — 포맷 토큰.
- `DEFAULT_ENCODING='utf-8'` — `ustr`과 I/O 모듈의 기본 인코딩.

COCO의 데이터셋 파일 자체(`annotations.json` 등)는 이 pkl 설정과 무관하다 — 세션 중 고른 COCO 데이터셋 경로(`MainWindow.coco_dataset_path`)는 종료 시 저장되지 않으며, 다음 세션은 다시 기본 타깃(`<save dir>/annotations.json`)에서 시작한다 → [formats.md](formats.md).
