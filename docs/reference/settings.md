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
- 종료 시 `closeEvent`(`labelImg.py:1285`)가 아래 키를 모두 기록한 뒤 `save()`한다.

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
| `SETTING_LABEL_FILE_FORMAT` | `labelFileFormat` | 마지막 사용 포맷(VOC/YOLO/CreateML) |
| `SETTING_CLASSIFY_TARGETS` | `classifyTargets` | (포크, 2026-07-08) 분류 카테고리 `(단축키, 폴더이름)` 쌍 목록 — 기본 `[('g','good'),('b','bad')]`. Edit Classify Categories 저장 시 즉시 `save()`되며, closeEvent가 개별 기록하지 않아도 로드된 dict에 남아 종료 시에도 보존된다 |

> ⚠️ `SETTING_WIN_GEOMETRY`(`window/geometry`)는 `constants.py:5`에 **정의만** 되어 있고 `closeEvent`에서는 저장되지 않는다 — 실제로 기록되는 창 키는 size/position/state뿐이다(`labelImg.py:1295-1297`).

> ℹ️ **기록 조건**: `filename`은 단일 파일 모드에서만 실제 경로가 저장되고 폴더를 연 상태로 종료하면 `''`가 저장된다(`labelImg.py:1290-1293`). `savedir`·`lastOpenDir`도 종료 시점에 해당 경로가 존재하지 않으면 `''`로 기록된다(`labelImg.py:1302-1310`).

> ℹ️ **시작 시 우선순위 (로컬 수정, 2026-07-03)**: `labelImg.py 이미지폴더 [클래스파일] [저장폴더]` 형태로 명령줄 인자를 주고 시작하면, `__init__`(`labelImg.py:577`)이 `open_dir_dialog(dir_path=이미지폴더, silent=True)`를 호출하고 `open_dir_dialog`(`labelImg.py:1390-1393`)는 **명시된 `dir_path`를 pkl의 `lastOpenDir`보다 우선**한다. 이때 `default_save_dir`도 설정되는데, 명령줄 저장폴더 인자가 있으면 그것을 유지하고 없으면 연 폴더가 된다(`labelImg.py:1410-1413`). 따라서 pkl의 `savedir`·`lastOpenDir`는 그 세션에서 무시되고, 종료 시 새 값으로 갱신된다. pkl의 `savedir`는 명령줄 저장폴더 인자가 없을 때만 시작 시 적용된다(`labelImg.py:534-535`). (수정 전에는 pkl의 `lastOpenDir`가 명령줄 폴더보다 우선되어 라벨 XML을 찾지 못하는 문제가 있었다.)

## 기타 상수

- `FORMAT_PASCALVOC='PascalVOC'`, `FORMAT_YOLO='YOLO'`, `FORMAT_CREATEML='CreateML'` — 포맷 토큰.
- `DEFAULT_ENCODING='utf-8'` — `ustr`과 I/O 모듈의 기본 인코딩.
