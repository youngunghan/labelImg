# How-to: AI 자동 라벨링 (Auto-label)

모델이 이미지를 보고 박스를 **제안**하면, 사람이 확인하고 받아들이거나 버리는 기능이다. 라벨링 자체를
대신하지는 않는다 — 제안은 받아들이기 전까지 절대 저장되지 않는다.

## 설치

기본 설치(`pyqt5`+`lxml`)만으로는 **AI 메뉴가 비활성화되어 있다** — 백엔드가 아무것도 설정되지
않았기 때문이다. 아래 [기본 상태: 미설정 시 AI 메뉴는 꺼져 있다](#기본-상태-미설정-시-ai-메뉴는-꺼져-있다)를
먼저 읽을 것. **실제 모델**로 추론하려면 ONNX 런타임이 필요하다. 이 저장소는 PyPI에 배포되어 있지
않으므로, 이 저장소 루트에서 로컬 소스로 설치한다:

```shell
pip install -e ".[ai]"
```

(`pip install labelImg[ai]`는 이 포크와 무관한 업스트림 패키지를 PyPI에서 받아오므로 쓰지 말 것 —
그 패키지에는 이 문서가 설명하는 AI 코드가 전혀 없다.)

이 extra는 `onnxruntime>=1.15`와 `numpy`를 설치한다(`setup.py:26-28`). 기본 설치에는 포함되지 않는다 —
`labelImg`는 이 extra 없이도 순수 라벨링 도구로 완전히 동작한다.

## 모델은 앱에 포함돼 있지 않다 (그리고 왜)

**이 저장소는 모델 가중치를 배포하지 않는다.** 가장 흔한 기본값일 Ultralytics YOLOv5/v8 사전학습
가중치는 **AGPL-3.0**이고, 이 앱은 **MIT**다. AGPL 가중치를 MIT 앱에 번들하거나 그것에 의존하게 만들면
MIT 사용자가 기대하지 않는 라이선스 의무가 딸려온다. 그래서 labelImg는 **백엔드(코드, MIT)만** 제공하고
**가중치는 사용자가 직접 준비**한다. 자세한 라이선스 설명·permissive 대안(Apache-2.0 YOLOX 등)·Ultralytics
내보내기 방법은 [`data/models/README.md`](../../data/models/README.md)를 참고할 것.

## 내 `.onnx` 모델을 앱에 연결하기

아직 모델 경로를 고르는 UI(파일 선택 대화상자)는 없다 — 백엔드/경로는 **설정 파일로만** 지정한다
(`labelImg.py:1449-1450`의 주석: "there is no picker UI yet, so it is config-file driven"). 앱을 닫은
상태에서 아래처럼 한 번 써 주면 된다(`~/.labelImgSettings.pkl`에 저장됨, `libs/settings.py`):

```python
from libs.settings import Settings
from libs.constants import SETTING_MODEL_BACKEND, SETTING_MODEL_PATH

settings = Settings()
settings.load()
settings[SETTING_MODEL_BACKEND] = 'yolo_onnx'
settings[SETTING_MODEL_PATH] = '/path/to/your/model.onnx'
settings.save()
```

키는 `model/backend` / `model/path` / `model/confThreshold`(`libs/constants.py:24-26`)다. 이후 앱을
실행하면 `AssistController`가 시작할 때 이 설정을 읽어 백엔드를 구성한다
(`AssistController.__init__`, `libs/assist/controller.py:85-129`). 신뢰도 임계값은 메뉴의 슬라이더로도
바로 조절되고, 앱 종료 시 그 값이 같은 설정에 다시 저장된다(`labelImg.py:1451-1453`).

### 지원하는 모델 출력 형식

`yolo_onnx` 백엔드(`libs/inference/yolo_onnx.py`)는 Ultralytics의 두 내보내기 레이아웃을 **자동 판별**한다:

| 레이아웃 | 출력 shape | 내용 |
|---|---|---|
| YOLOv8 | `(1, 4+nc, N)` | `cx, cy, w, h` + 클래스별 점수(objectness 없음) |
| YOLOv5 | `(1, N, 5+nc)` | `cx, cy, w, h, objectness` + 클래스별 점수 |

두 레이아웃 중 어느 쪽으로도 애매하게 해석될 수 있는 출력 shape(예: 정사각형)를 만나면, 백엔드는
**추측하지 않고 에러를 던진다** — 잘못된 추측은 그럴듯해 보이는 틀린 박스를 조용히 만들어내기 때문이다.
필요하면 config의 `layout` 키(`v8`/`v5`)로 강제할 수 있다. 클래스 이름은 ① config 오버라이드 →
② ONNX 메타데이터의 `names` → ③ 모델 파일 옆 `classes.txt` → ④ `class_0`, `class_1`, ... 순으로
해석된다(`data/models/README.md`, `libs/inference/yolo_onnx.py:674-677`, `:789-816`).

`[ai]` extra(`pip install -e ".[ai]"`)가 설치돼 있지 않으면(또는 `model_path`가 없거나 파일이
깨졌으면) `yolo_onnx` 백엔드는 그냥 구성에 실패하고 **AI 액션이 비활성화**된다 — 크래시하지 않는다(아래 참고).

## 기본 상태: 미설정 시 AI 메뉴는 꺼져 있다

여기가 놓치기 쉬운 부분이다(과거에는 반대로 놓치기 쉬웠다 — 아래 참고). 백엔드를 한 번도 설정하지 않은
**기본 상태**에서 `AssistController`는 **어떤 백엔드도 자동으로 띄우지 않는다**:
`libs/inference/registry.py`의 `DEFAULT_BACKEND`는 `'stub'`가 아니라 **`None`**이고
(`registry.py:39`), 설정 파일에 `model/backend`가 없으면 `build_backend()`는 그냥 `None`을 반환한다
(`registry.py:140-155`). 즉:

- 기본 설치 상태(`pyqt5`+`lxml`만, `[ai]` extras도 설정도 없음)에서는 `Ctrl+I`를 눌러도 아무 일도
  일어나지 않는다 — AI 액션 자체가 처음부터 비활성화되어 있고, 메뉴 툴팁에 "No model backend
  configured" 안내가 뜬다(`NO_BACKEND_CONFIGURED_HINT`, `libs/assist/controller.py:85-88`,
  `_unavailable_hint`/`refresh_actions`, `libs/assist/controller.py:234-245, 344-408`).
- **AI 메뉴를 켜려면** [위 절차](#내-onnx-모델을-앱에-연결하기)대로 `model/backend`를 `yolo_onnx`로,
  `model/path`를 실제 `.onnx` 파일 경로로 설정해야 한다 — `pip install -e ".[ai]"`(onnxruntime+numpy,
  이 저장소 루트에서 실행)도 함께 필요하다. 둘 중 하나라도 빠지면(익스트라 미설치, 경로 미설정, 파일 손상 등) `build_backend()`가
  예외 대신 `None`을 돌려주고, AI 액션은 계속 비활성 상태로 남으며 툴팁에 다른 안내
  (`BACKEND_UNAVAILABLE_HINT`, `libs/assist/controller.py:89-92`)가 뜬다.
- `libs/inference/stub.py`의 `StubBackend`(numpy/onnxruntime 없이 동작하는 결정론적 가짜 검출기,
  `predict`, `libs/inference/stub.py:105-139`)는 여전히 코드베이스에 있고 테스트 스위트를 구동하지만,
  더 이상 아무것도 설정하지 않았을 때 자동으로 선택되지 않는다 — 쓰려면 `model/backend`를 명시적으로
  `'stub'`로 설정해야 한다(레지스트리에는 계속 등록되어 있다, `registry.py:85-88`).

## 사용 흐름

1. **AI 메뉴**(`&AI`, `labelImg.py:478`·`504`)에서 **Confidence Threshold** 슬라이더로 원하는 신뢰도
   임계값을 미리 맞춰도 되고, 나중에 결과를 보면서 조절해도 된다(0~100%, 기본 50%).
2. **`Ctrl+I`** (`Auto-label Image`) — 현재 이미지에 모델을 돌려 박스를 **제안**으로 올린다
   (`SHORTCUT_AUTO_LABEL`, `libs/assist/controller.py:57`). 이미지가 열려 있고 백엔드가 사용 가능할 때만
   활성화된다. 다시 누르면 이전 라운드의 제안을 지우고 새로 돌린다(`auto_label_image`,
   `libs/assist/controller.py:459-481`).
3. 제안 박스는 **점선 + 반투명**으로 그려져 실제 박스와 한눈에 구별된다(`Shape.provisional`,
   `libs/shape.py:62`,`110`,`132`,`159-162`). 각 제안에는 모델이 매긴 신뢰도(`Shape.confidence`)가 함께
   붙는다.
4. **Confidence Threshold** 슬라이더를 움직이면 이미 받은 검출 결과 중 임계값 이상인 것만 다시
   그려진다 — **모델을 다시 돌리지 않는다**(`AssistController.set_threshold`/`_sync_suggestions`,
   `libs/assist/controller.py:438-458`,`619-655`). 슬라이더는 탐색용이라 실시간으로 켜고 끌 수 있다.
5. **`Ctrl+Return`** (`Accept All Suggestions`) — 화면의 모든 제안을 한 번에 **진짜 박스로 승격**한다.
   승격된 박스는 점선/반투명이 풀리고 일반 색으로 바뀌며, 라벨이 클래스 목록에 없었다면 자동으로
   추가된다(`accept_all`, `libs/assist/controller.py:656-676`).
6. **`Ctrl+Backspace`** (`Reject All Suggestions`) — 화면의 모든 제안을 한 번에 버린다(`reject_all`,
   `libs/assist/controller.py:677-689`).
7. 물론 제안 하나하나를 캔버스에서 골라 `Delete`로 개별적으로 지울 수도 있다 — 지운 제안은 임계값을
   다시 움직여도 되살아나지 않는다(`discard_shape`, `libs/assist/controller.py:600-618`).

## 제안은 받아들이기 전까지 저장되지 않는다

`Ctrl+S`를 포함한 **모든** 저장 경로(수동 저장, 자동저장, Export COCO..., Verify)는 결국
`save_labels` 하나로 수렴하고, 그 안에 있는 **단 한 줄의 필터**가 `provisional`(아직 받아들이지 않은
제안) 상태인 도형을 어노테이션 파일에서 걸러낸다:

```python
shapes = [format_shape(shape) for shape in self.canvas.shapes if not shape.provisional]
```
(`labelImg.py:1054`)

그래서 저장 직전에 `Ctrl+I`로 제안만 띄워 놓고 파일을 닫아도, 받아들이지 않은 박스는 디스크에 한 번도
쓰이지 않는다. `Ctrl+Return`으로 승격해야 비로소 다른 박스들과 똑같이 저장 대상이 된다. 이 규칙은
포맷(VOC/YOLO/CreateML/COCO)과 무관하게 동일하다.

## 참고

- 예측은 UI 스레드를 막지 않는 별도 워커에서 돈다(`InferenceService`, 단일 워커 QThreadPool). 추론 중에
  다른 이미지로 넘어가면, 이미 떠난 이미지에 대한 느린 결과가 나중에 도착해도 **현재 이미지와 경로가
  다르면 조용히 버려진다**(`AssistController._is_current`, `libs/assist/controller.py:565-577`) — 새
  이미지에 엉뚱한 박스가 얹히는 사고를 막는다.
- `Ctrl+D`(Duplicate)로 제안을 복제하면 복제본도 `provisional`을 물려받아 그대로 점선/반투명으로
  남는다(`Shape.copy`, `libs/shape.py:220`,`234-236`) — 저장하려면 마찬가지로 받아들여야 한다.

관련: [../explanation/ml-assist-architecture.md](../explanation/ml-assist-architecture.md) ·
[../reference/shortcuts.md](../reference/shortcuts.md) · [annotate-coco.md](annotate-coco.md)
