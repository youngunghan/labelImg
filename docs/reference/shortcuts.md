# 레퍼런스: 단축키(핫키)

코드의 액션 정의(`labelImg.py` `__init__`) 기준 전체 단축키. 루트 [../../README.rst](../../README.rst)의 핫키 표는 이 중 일부만 싣는다.

## 파일·디렉터리

| 키 | 동작 |
|---|---|
| `Ctrl+O` | 파일 열기(Open) |
| `Ctrl+U` | 디렉터리 열기(Open Dir) — 모든 이미지 로드 |
| `Ctrl+R` | 기본 어노테이션 저장 디렉터리 변경 |
| `Ctrl+Shift+O` | 어노테이션 파일 열기 |
| (메뉴 전용) | **File > Import COCO...** — COCO 데이터셋 json을 골라 현재 이미지의 박스를 불러옴(`labelImg.py:257-258, 2264`) |
| (메뉴 전용) | **File > Export COCO...** — COCO 데이터셋 json을 골라(또는 새로 만들어) 현재 이미지를 병합 저장(`labelImg.py:259-260, 2293`) |
| `Ctrl+S` | 저장 |
| `Ctrl+Shift+S` | 다른 이름으로 저장(Save As) |
| `Ctrl+Y` | 저장 포맷 전환(PascalVOC→YOLO→CreateML→COCO 순환, `labelImg.py:651-661`) |
| `Ctrl+W` | 현재 파일 닫기 |
| `Ctrl+Shift+D` | 현재 이미지 삭제 |
| `d` / `a` | 다음 / 이전 이미지 |
| `Ctrl+Q` | 종료(Quit) |

## 박스 그리기·편집

| 키 | 동작 |
|---|---|
| `w` | 사각형 박스 생성(Create RectBox) |
| `Ctrl+E` | 선택 박스의 라벨 편집 |
| `Ctrl+D` | 현재 라벨·박스 복사 |
| `Ctrl+V` | **이전 이미지의 박스 복사**(현재 이미지에 올리고 저장) → [../how-to/more-features.md](../how-to/more-features.md) |
| `Delete` | 선택한 박스 삭제 |
| `↑ → ↓ ←` | 선택한 박스를 1픽셀 이동 |
| `Esc` | 그리던 박스 취소 |
| `Return` | 그리던 박스 닫기/확정(메인 Enter 키만 해당, 숫자패드 Enter 미지원) |
| `Ctrl`(누른 채 그리기) | 정사각형으로 제약 |
| `Ctrl+Shift+R` | Draw Squares 모드 토글 |
| `Ctrl+H` / `Ctrl+A` | 모든 박스 숨기기 / 보이기 |

## 보기 (줌·맞춤·밝기)

| 키 | 동작 |
|---|---|
| `Ctrl++` / `Ctrl+-` | 줌 인 / 아웃 |
| `Ctrl+=` | 원본 크기(100%) |
| `Ctrl+F` | 창에 맞춤(Fit Window) |
| `Ctrl+Shift+F` | 너비에 맞춤(Fit Width) |
| `Ctrl+휠` | 줌 인/아웃 |
| `Ctrl+Shift++` / `Ctrl+Shift+-` | 밝게 / 어둡게 |
| `Ctrl+Shift+=` | 밝기 리셋(50%) |
| `Ctrl+Shift+휠` | 밝기 조절 |
| 휠 / 좌클릭 드래그(빈 영역) | 스크롤·패닝 |

## 검증·모드·색

| 키 | 동작 |
|---|---|
| `Space` | 현재 이미지의 **verified** 상태 토글(연녹색 배경; 다시 누르면 해제). 라벨 파일이 없으면 먼저 저장해 생성 |
| `Ctrl+Shift+C` | Single Class Mode 토글(직전 라벨 재사용) |
| `Ctrl+Shift+P` | 박스 위 라벨 표시 토글(Display Labels) |
| `Ctrl+Shift+L` | 라벨 패널(Box Labels 도크) 표시/숨김 |
| `Ctrl+Shift+A` | 고급 모드(Advanced) 토글 |
| `Ctrl+L` | 박스 선 색 변경(Box Line Color) |
| `Ctrl+Shift+E` | 사전 정의 클래스 편집(영구 저장) |
| `Ctrl+J` | Edit 모드(고급 모드의 분리 액션) |

## AI (모델 보조 라벨링, `&AI` 메뉴)

구현: `libs/assist/controller.py`(`AssistController`). 이미지가 열려 있고 사용 가능한 모델 백엔드가 있을 때만 활성화된다(`refresh_actions`, `controller.py:344-408`) — 백엔드가 없으면(base install 등) 툴팁에 설치 안내가 뜬다.

| 키 | 동작 |
|---|---|
| `Ctrl+I` | **Auto-label Image** — 현재 이미지에 모델을 실행해 결과를 점선·반투명 박스(제안)로 표시(`SHORTCUT_AUTO_LABEL`, `controller.py:57, 177-180`) |
| `Ctrl+Return` | **Accept All Suggestions** — 이 이미지의 모든 제안을 실제 박스로 확정(`SHORTCUT_ACCEPT_ALL`, `controller.py:58, 181-184`) |
| `Ctrl+Backspace` | **Reject All Suggestions** — 이 이미지의 모든 제안을 폐기(`SHORTCUT_REJECT_ALL`, `controller.py:59, 185-188`) |
| (메뉴 전용, 슬라이더) | **Confidence Threshold** — 이 값 미만의 제안은 화면에서 숨김(재추론 없이 필터만 재적용, `controller.py:200-232, 299-316`) |
| `Ctrl+Shift+U` | **Sort by Uncertainty** (Phase 4, 신규) — 폴더 전체 배치 채점 결과로 `m_img_list`를 불확실성 내림차순 재정렬(`SHORTCUT_SORT_BY_UNCERTAINTY`, `controller.py:71`; `sort_by_uncertainty`, `:869-893`). 현재 열린 이미지는 그대로 선택 유지 |
| (메뉴 전용) | **Score Folder for Active Learning** — 폴더의 모든 이미지를 배치 추론해 이미지별 불확실성 점수를 매김. 두 번째 트리거는 **취소**(`score_folder`/`cancel_batch_scoring`, `controller.py:719-782`) |
| (메뉴 전용) | **Restore Filesystem Order** — Sort by Uncertainty를 되돌려 원래 스캔 순서로 복원(`restore_original_order`, `controller.py:905-928`) |

> 제안(provisional) 박스는 사용자가 Accept하기 전까지 저장 파일에 절대 기록되지 않는다(`MainWindow.save_labels`의 단일 필터, `labelImg.py:1033-1040`) → [formats.md](formats.md) · [modules.md](modules.md).
>
> **Score/Sort/Restore Order에 단축키를 하나만 준 이유**: 이 포크는 이미 단축키 이중바인딩 버그를 한 번 겪었다(Single Class Mode가 Ctrl+Shift+S에서 옮겨진 이력, `controller.py:48-56` 주석 참조). `Ctrl+Shift+U`("Uncertainty")는 그 주석이 열거한 기존 바인딩 전부와 겹치지 않는 것을 확인하고 골랐다. Score Folder/Restore Order는 자주 쓰는 액션이 아니라 메뉴 전용으로 남겨, 사용자 정의 분류 카테고리(File > Edit Classify Categories)가 임의의 단일 키를 계속 자유롭게 쓸 수 있는 여지를 줄이지 않는다.

## ⚠️ 이미지 분류 (파일 이동 — 주의)

| 키 | 동작 |
|---|---|
| `g`(기본값, 변경 가능) | Classify Good — 현재 이미지+라벨을 `<폴더>_good/`로 **이동** 후 다음 |
| `b`(기본값, 변경 가능) | Classify Bad — `<폴더>_bad/`로 **이동** 후 다음 |
| (메뉴 전용) | **File > Edit Classify Categories** — 분류 카테고리 `(단축키, 폴더이름)` 목록 편집(재시작 불필요, 설정에 영속) |
| `Ctrl+Z` | 마지막 분류 이동 되돌리기(Undo Classify) |

> `g`/`b`는 수정자 없는 단일 키이고 디스크에서 파일을 옮긴다. 상세·주의는 [../how-to/more-features.md](../how-to/more-features.md).

verify·difficult·single-class 등 워크플로는 [../how-to/verify-and-difficult.md](../how-to/verify-and-difficult.md) 참고.
