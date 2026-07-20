# How-to: verify · difficult · 작업 모드

라벨링 속도를 높이는 플래그·모드 모음.

## Verify(검증) 플래그

`Space`를 누르면 현재 이미지의 **verified 상태가 토글**된다 — 미검증이면 verified로 표시되고, 이미 verified면 해제된다(`toggle_verify`, `labelFile.py:145-146`). verified일 때 캔버스 배경이 **연녹색**으로 바뀐다(`Canvas.paintEvent`, `canvas.py:546-553`). 토글 직후 라벨 파일이 **즉시 저장**되며, 라벨 파일이 없으면 먼저 생성된다(`verify_image`, `labelImg.py:1621-1644`).

용도: 자동 생성된 데이터셋을 사람이 빠르게 검수할 때, 박스를 새로 그리는 대신 "이 이미지는 확인했다"고 플래그만 찍고 넘어가는 워크플로다.

포맷별 저장:
- **PASCAL VOC**: 루트에 `verified="yes"` 속성.
- **CreateML**: 이미지 항목의 `"verified": true`.
  - (2026-07-08 수정) CreateML은 다시 열 때 verified를 **현재 이미지와 매칭되는 항목**에서 읽는다(`create_ml_io.py:121`) — 이전에는 JSON 첫 항목에서 읽어 다중 이미지 파일에서 다른 이미지의 verified가 표시되는 버그가 있었다.
- **YOLO**: verify 개념을 저장하지 않는다.

## Difficult 플래그

박스를 선택하고 라벨 리스트 옆의 **difficult 체크박스**를 켜면, 그 박스의 `difficult`가 1이 된다. "명확히 보이지만 맥락 없이는 인식이 어려운" 객체를 표시하는 PASCAL VOC 관례다. 학습 시 difficult 객체를 포함/제외할 수 있다.

포맷별:
- **PASCAL VOC**: `<difficult>1</difficult>`로 보존.
- **YOLO**: **저장되지 않는다**(읽으면 항상 False).
- **CreateML**: **저장 시 difficult가 기록되지 않으며**, 읽을 때는 모든 박스가 True로 들어온다(코드 특성, `create_ml_io.py:133`).

## Single Class Mode 와 "use default label" — 서로 다른 두 기능

매번 라벨 입력창을 띄우지 않고 새 박스에 라벨을 자동 부여하는 길은 **두 가지**이며, 코드상 별개의 상태로 동작한다(`new_shape`, `labelImg.py:1134-1171`):

- **Single Class Mode** (`View → Single Class Mode`, 단축키 `Ctrl+Shift+C`, `SETTING_SINGLE_CLASS`로 영속화): 켜면 새 박스가 **직전에 입력/사용한 라벨**(`self.lastLabel`)을 재사용한다(`labelImg.py:1145-1146`). 한 종류만 연속으로 찍을 때 편하다. — 콤보박스 값이 아니라 "마지막 라벨"이라는 점에 주의. (단축키는 업스트림의 `Ctrl+Shift+S`가 Save As와 충돌해 이 포크에서 `Ctrl+Shift+C`로 분리됨.)
- **Use default label**(라벨 도크의 "use default label" 체크박스 + `DefaultLabelComboBox`): 체크하면 새 박스가 **콤보박스에서 고른 기본 라벨**(`self.default_label`)로 붙는다(`labelImg.py:1151`).

둘 다 꺼져 있으면 박스를 완성할 때마다 `LabelDialog`가 뜬다. 둘 다 켜져 있으면 **use default label이 우선**한다 — `new_shape`가 체크박스를 먼저 검사하므로(`labelImg.py:1139`) Single Class Mode는 무시된다.

> YOLO 저장에는 이 default class(콤보박스 기본 라벨) 기능이 참조되지 않는다.

## Auto Save(자동 저장)

`View → Auto Save mode`. 켜면 다음/이전 이미지로 넘어갈 때(`d`/`a`) dirty 상태면 자동 저장한다(`open_prev_image`/`open_next_image`, `labelImg.py:1646-1700`). 단, 기본 저장 폴더가 없으면 폴더 지정을 먼저 요구한다.

## 정사각형 그리기(draw squares)

`Edit → Draw Squares`(`Ctrl+Shift+R`, 체크형) 또는 그리는 중 `Ctrl`을 누르고 있으면 박스가 정사각형으로 제약된다(`SETTING_DRAW_SQUARE`). 구현은 그리기 분기(`canvas.py:148-155`)와 `Canvas.bounded_move_vertex`(`canvas.py:409-417`); `Ctrl` hold 토글은 `MainWindow.keyPressEvent`/`keyReleaseEvent`(`labelImg.py:622-629`).

## 라벨 표시(display labels)

`View → Display Labels`로 박스 위 라벨 텍스트 표시를 토글한다(`SETTING_PAINT_LABEL` → `Shape.paint_label`).

분류·복사·클래스 편집·고급 모드·밝기 등 부가 기능 → [more-features.md](more-features.md). 단축키 전체 → [../reference/shortcuts.md](../reference/shortcuts.md). 설정 키 → [../reference/settings.md](../reference/settings.md).
