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

**GitHub Release의 Windows exe를 쓰는 경우**는 이 단계가 필요 없다 — `labelImg.spec`이 빌드 시점에
onnxruntime + numpy를 이미 exe 안에 번들해 두었다(자세한 내용은
[`docs/how-to/install-and-build.md`](install-and-build.md#onnxruntime-onnx-런타임-번들)). 다만 그 exe에도
모델 가중치는 들어있지 않으므로, 아래 "모델은 앱에 포함돼 있지 않다" 절과 "내 `.onnx` 모델을 앱에
연결하기" 절은 exe 사용자에게도 그대로 적용된다.

## 모델은 앱에 포함돼 있지 않다 (그리고 왜)

**이 저장소는 모델 가중치를 배포하지 않는다.** 가장 흔한 기본값일 Ultralytics YOLOv5/v8 사전학습
가중치는 **AGPL-3.0**이고, 이 앱은 **MIT**다. AGPL 가중치를 MIT 앱에 번들하거나 그것에 의존하게 만들면
MIT 사용자가 기대하지 않는 라이선스 의무가 딸려온다. 그래서 labelImg는 **백엔드(코드, MIT)만** 제공하고
**가중치는 사용자가 직접 준비**한다. 자세한 라이선스 설명·permissive 대안(Apache-2.0 YOLOX 등)·Ultralytics
내보내기 방법은 [`data/models/README.md`](../../data/models/README.md)를 참고할 것.

## 내 `.onnx` 모델을 앱에 연결하기

**AI 메뉴 > Model Settings...** 에서 앱을 껐다 켤 필요 없이 바로 설정한다. 다이얼로그는
(`libs/assist/settings_dialog.py`의 `ModelSettingsDialog`, `AssistController.open_model_settings_dialog`가
연다) 딱 두 가지만 물어본다:

- **Backend**: `사용 안 함`(AI 끄기) 또는 `YOLO (ONNX)`. `stub`(가짜 검출기, 테스트 전용)은 **의도적으로
  선택지에 없다** — 골라도 다음 실행에서 미설정으로 취급되는 값이라 UI로 노출하면 사용자가 고른 게
  조용히 무효화되는 모순이 생긴다(`_LEGACY_IMPLICIT_DEFAULT_BACKEND`, 아래
  [§기본 상태](#기본-상태-미설정-시-ai-메뉴는-꺼져-있다)의 `StubBackend` 설명 참고).
- **Model path (.onnx)**: 직접 입력하거나 **Browse...** 버튼(`QFileDialog.getOpenFileName`, `*.onnx`로
  필터링)으로 고른다. 이 앱에는 모델 가중치가 들어있지 않다는 안내와
  [`data/models/README.md`](../../data/models/README.md) 링크가 다이얼로그 안에 함께 있다.

**OK**를 누르면 즉시 검증하고 — 경로가 비었거나 없거나 `.onnx`가 아니거나, onnxruntime이 이 설치에
없거나(다른 안내), 파일은 있지만 로드에 실패하거나(손상/미지원 형식) — 문제가 있으면 그 원인에 맞는
메시지를 보여주고 아무것도 적용/저장하지 않는다. 성공하면:

1. **그 자리에서 설정을 저장**한다(`AssistController.apply_model_settings`가 `settings.save()`를
   즉시 호출 — `closeEvent`가 나중에 저장해 줄 거라고 미루지 않는다. 즉 앱이 중간에 죽어도 방금 고른
   설정은 이미 디스크에 있다).
2. **백엔드를 그 자리에서 재구성**하고 `refresh_actions()`를 호출한다 — **재시작 없이** AI 액션(Auto-label
   Image 등)이 바로 활성화된다. `edit_classify_categories`가 분류 액션을 재빌드하는 것과 같은 방식이다.
3. 상태 표시줄에 결과를 알린다.

**사용 안 함**을 고르면 저장된 `model/backend` 키만 지우고 AI 액션을 다시 비활성화한다 — **모델
경로(`model/path`, 메모리상 `self.model_path`도 함께)는 그대로 남긴다.** 백엔드 이름 없이는 경로
혼자 아무것도 하지 못하므로(`AssistController.__init__`이 백엔드가 없으면 항상 `is_available()`을
False로 두고, `_build_backend`도 백엔드 이름이 없으면 아무것도 만들지 않는다) 남겨두는 것 자체는
무해하며, 대신 다음에 다시 켤 때 `.onnx` 파일을 또 찾아 지정할 필요 없이 백엔드만 다시 고르면 된다.
**Score Folder로 배치 채점이 도는 중에 사용 안 함을 골라도 안전하다** — `apply_model_settings`가
백엔드를 떨구기 전에 실행 중인 배치를 먼저 취소하고(`AssistController.apply_model_settings`,
`libs/assist/controller.py:349-467`), 설령 그 취소가 다른 이유로 스킵되더라도 Score Folder
액션(배치를 멈출 유일한 컨트롤)은 백엔드 가용성과 무관하게 배치가 도는 동안 계속 눌린다
(`refresh_actions`, `:599-679`) — 그래서 스캔 도중 AI를 꺼도 취소 버튼이 회색으로 죽어버리는 일이
없다.

**모델을 교체(다른 `.onnx`로 다시 적용)해도 이전 모델의 흔적이 남지 않는다**: 화면에 떠 있던 제안
박스와 그 아래 추적 상태(`_dismissed` 등)는 교체 즉시 전부 지워지고(`AssistController.set_backend`,
`:324-347`), 교체 전에 이미 날아간 대화형(Ctrl+I) 요청이 교체 후에야 응답하면 조용히 버려진다 —
같은 이미지를 보고 있어도 이전 모델의 결과가 새 모델의 결과인 것처럼 Accept All로 커밋되는 일은
없다.

이 다이얼로그가 하는 일은 전부 `AssistController.apply_model_settings(backend_name, model_path)`라는
평범한 메서드 하나에 있다 — 다이얼로그 자체는 입력만 모아 그 메서드를 호출하는 얇은 껍데기다. 이전
버전(설정 파일을 손으로 편집)도 여전히 그대로 동작한다 — `~/.labelImgSettings.pkl`(`libs/settings.py`)에
직접 쓰고 싶다면:

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
(`AssistController.__init__`, `libs/assist/controller.py:86-306`). 신뢰도 임계값은 메뉴의 슬라이더로도
바로 조절되고, 앱 종료 시 그 값이 같은 설정에 다시 저장된다(`labelImg.py:1486`).

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
  configured" 안내가 뜬다(`NO_BACKEND_CONFIGURED_HINT`, `libs/assist/controller.py:86-89`,
  `_unavailable_hint`/`refresh_actions`, `libs/assist/controller.py:487-498, 611-691`).
- **AI 메뉴를 켜려면** [위 절차](#내-onnx-모델을-앱에-연결하기)대로 **AI 메뉴 > Model Settings...**에서
  `YOLO (ONNX)`를 고르고 `.onnx` 파일 경로를 지정한다 — `pip install -e ".[ai]"`(onnxruntime+numpy,
  이 저장소 루트에서 실행)도 함께 필요하다. 둘 중 하나라도 빠지면(익스트라 미설치, 경로 미설정, 파일 손상 등)
  다이얼로그가 그 자리에서 원인이 다른 에러 메시지를 보여주고 아무것도 적용하지 않으며, 이미 AI가 켜져
  있었다면 그 상태 그대로 남는다 — `AssistController.apply_model_settings`가 실패를 반영하기 전에는 아무것도
  바꾸지 않기 때문이다. 다이얼로그를 거치지 않고 설정 파일을 직접 건드린 경우에는 `build_backend()`가
  예외 대신 `None`을 돌려주고, AI 액션은 계속 비활성 상태로 남으며 툴팁에 다른 안내
  (`BACKEND_UNAVAILABLE_HINT`, `libs/assist/controller.py:90-93`)가 뜬다.
- `libs/inference/stub.py`의 `StubBackend`(numpy/onnxruntime 없이 동작하는 결정론적 가짜 검출기,
  `predict`, `libs/inference/stub.py:105-139`)는 여전히 코드베이스에 있고 테스트 스위트를 구동하지만,
  더 이상 아무것도 설정하지 않았을 때 자동으로 선택되지 않는다 — 쓰려면 `model/backend`를 명시적으로
  `'stub'`로 설정해야 한다(레지스트리에는 계속 등록되어 있다, `registry.py:85-88`).

## 사용 흐름

1. **AI 메뉴**(`&AI`, `labelImg.py:478`·`504`)에서 **Confidence Threshold** 슬라이더로 원하는 신뢰도
   임계값을 미리 맞춰도 되고, 나중에 결과를 보면서 조절해도 된다(0~100%, 기본 50%).
2. **`Ctrl+I`** (`Auto-label Image`) — 현재 이미지에 모델을 돌려 박스를 **제안**으로 올린다
   (`SHORTCUT_AUTO_LABEL`, `libs/assist/controller.py:60`). 이미지가 열려 있고 백엔드가 사용 가능할 때만
   활성화된다. 다시 누르면 이전 라운드의 제안을 지우고 새로 돌린다(`auto_label_image`,
   `libs/assist/controller.py:732-759`).
3. 제안 박스는 **점선 + 반투명**으로 그려져 실제 박스와 한눈에 구별된다(`Shape.provisional`,
   `libs/shape.py:62`,`110`,`132`,`159-162`). 각 제안에는 모델이 매긴 신뢰도(`Shape.confidence`)가 함께
   붙는다.
4. **Confidence Threshold** 슬라이더를 움직이면 이미 받은 검출 결과 중 임계값 이상인 것만 다시
   그려진다 — **모델을 다시 돌리지 않는다**(`AssistController.set_threshold`/`_sync_suggestions`,
   `libs/assist/controller.py:711-728`,`693-728`). 슬라이더는 탐색용이라 실시간으로 켜고 끌 수 있다.
5. **`Ctrl+Return`** (`Accept All Suggestions`) — 화면의 모든 제안을 한 번에 **진짜 박스로 승격**한다.
   승격된 박스는 점선/반투명이 풀리고 일반 색으로 바뀌며, 라벨이 클래스 목록에 없었다면 자동으로
   추가된다(`accept_all`, `libs/assist/controller.py:976-995`).
6. **`Ctrl+Backspace`** (`Reject All Suggestions`) — 화면의 모든 제안을 한 번에 버린다(`reject_all`,
   `libs/assist/controller.py:997-1008`).
7. 물론 제안 하나하나를 캔버스에서 골라 `Delete`로 개별적으로 지울 수도 있다 — 지운 제안은 임계값을
   다시 움직여도 되살아나지 않는다(`discard_shape`, `libs/assist/controller.py:920-937`).

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
  다르면 조용히 버려진다**(`AssistController._is_current`, `libs/assist/controller.py:885-894`) — 새
  이미지에 엉뚱한 박스가 얹히는 사고를 막는다.
- `Ctrl+D`(Duplicate)로 제안을 복제하면 복제본도 `provisional`을 물려받아 그대로 점선/반투명으로
  남는다(`Shape.copy`, `libs/shape.py:220`,`234-236`) — 저장하려면 마찬가지로 받아들여야 한다.

관련: [../explanation/ml-assist-architecture.md](../explanation/ml-assist-architecture.md) ·
[../reference/shortcuts.md](../reference/shortcuts.md) · [annotate-coco.md](annotate-coco.md)
