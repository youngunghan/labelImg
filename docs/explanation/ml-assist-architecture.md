# ML 어시스트 아키텍처 (설계 + 구현 현황)

> ⚠️ **Phase 1~2·4 구현 완료 (Phase 1 머지 `a32acd3`) · Phase 3·5~6 미완료.** 이 페이지가 서술하는 `libs/coco_io.py`·`libs/inference/`(types·backend·stub·registry·service·**yolo_onnx**)·`libs/assist/`(controller·suggestion, 능동학습 포함)·`Shape`의 provisional 필드는 **저장소에 실제로 존재한다.** 아직 끝나지 않은 것은 수락/거부 다듬기, 폴리곤/키포인트, SAM이다. 단계별 현황은 [§단계별 순서](#단계별-순서)를 참조.
>
> 이 문서는 여전히 **explanation**이다 — "무엇을 왜 그렇게 지었는가"(설계 근거·트레이드오프)가 본체이고, 구현이 끝난 지금도 그 근거는 그대로다. 인용 규약: **구현된 코드만 `file.py:line`으로 인용**하고(모든 줄 번호는 실제 파일에서 확인함), **아직 없는 Phase 3~6 모듈은 이름으로만 지칭**한다(줄 번호 없음).

현재 저장소는 [architecture.md](architecture.md)가 서술한 대로 **MainWindow(God-object) ↔ Canvas ↔ Shape ↔ Reader/Writer** 구조의 수동 박스 도구다. 이 설계의 목표는 그 포크를 **AI 보조 주석 도구**로 진화시키되, **`MainWindow`의 God-object 문제를 더 악화시키지 않는 것**이다.

## 왜 새 클래스인가 — 분류 기능이 남긴 교훈

포크가 이미 붙인 이미지 분류 기능은 **`MainWindow`에 직접 용접된 약 285줄**이다(`classify_current_image` `labelImg.py:1805` ~ `rebuild_classify_actions` 끝 `labelImg.py:2089`). 동작은 하지만, 이것이 **반복하면 안 되는 성장 패턴**이다. `MainWindow` 클래스는 이미 `labelImg.py:76`부터 파일 끝(2,442줄) 직전까지 약 2,300줄이다. 추론 서비스·어시스트 컨트롤러·능동학습·SAM을 같은 방식으로 얹으면 600~1,000줄이 더 붙는다.

> **이 예측은 Phase 1에서 검증됐다.** 추론 서비스 + 어시스트 컨트롤러 + provisional 도형 + COCO 레인이 다 들어왔는데도 `MainWindow`가 얻은 것은 **배선뿐**이다: import 3줄(`labelImg.py:48-50`), 생성 3줄(`labelImg.py:444-446`), 액션 등록(`labelImg.py:468-471`·`labelImg.py:504`), `toggle_actions`의 후처리 1줄(`labelImg.py:746`), 저장 초크포인트의 **필터 1줄**(`labelImg.py:1040`). AI 행위 자체는 `MainWindow` 안에 **한 줄도 없다.**

동시에, 분류 기능은 **따라 해야 할 좋은 패턴**도 남겼다: 액션을 **동적으로 만들어 등록**하는 방식이다.

- `create_classify_actions()`(`labelImg.py:2003`)가 설정에서 (단축키, 이름) 쌍을 읽어 `QAction` 리스트를 만든다.
- 만들어진 액션은 File 메뉴 튜플에 `+ tuple(self.classify_actions)`로 합쳐지고(`labelImg.py:500-503`), 이미지 의존 액션 집합 `onLoadActive`에도 합쳐진다(`labelImg.py:468-471`).
- `toggle_actions()`(`labelImg.py:735-746`)가 `onLoadActive`를 일괄 enable/disable 한다 — 이미지가 열려야 켜지는 액션은 **여기에 등록만 하면 된다.**

`AssistController.create_actions()`(`libs/assist/controller.py:274-319`)와 `load_active_actions()`(`libs/assist/controller.py:355-367`)가 정확히 이 패턴을 그대로 쓴다.

즉 **`MainWindow`는 "만들어진 것을 소유하고 배선하는" 껍데기 역할을 이미 하고 있다.** ML 어시스트는 이 껍데기 역할만 쓰고, 로직은 밖으로 뺀다.

## 핵심 결정 — 형제 패키지 + 얇은 컨트롤러 2개

**ML 어시스트 + 포맷 확장 스파인**을 기존 `libs/` 옆의 **새 형제 패키지**로 만든다: `libs/inference/`, `libs/assist/`, `libs/coco_io.py`. 그리고 `MainWindow`가 **소유하고 배선만** 하는 얇은 컨트롤러 객체 **두 개**를 둔다.

| 객체 | 책임 | MainWindow와의 관계 | 현황 |
|---|---|---|---|
| `InferenceService` (QObject)<br/>`libs/inference/service.py:186` | 모델 백엔드를 들고 스레드풀에서 추론을 돌리고 결과를 시그널로 낸다 | 생성·소유·시그널 연결만 (`labelImg.py:444`) | Phase 1c 완료 |
| `AssistController` (QObject)<br/>`libs/assist/controller.py:112` | AI 액션, provisional 도형 수명주기, 임계값, 능동학습 재정렬(Phase 4) | 생성·소유, 액션을 메뉴/`onLoadActive`에 등록만 (`labelImg.py:445-446`, `labelImg.py:468-471`) | Phase 1c + 4 완료 |

`MainWindow`는 **Qt 껍데기로 남는다.** 새 기능의 코드는 `MainWindow` 메서드 안이 아니라 컨트롤러 안에 있다.

부수 효과가 하나 더 있는데, 이게 사실상 결정적이다: **컨트롤러로 뽑아내면 각 관심사가 단위 테스트 가능해진다.** `StubBackend`(아래) 하나만 물리면 **Qt 이벤트 루프도, 실제 모델도 없이** 어시스트 로직을 검증할 수 있다. `MainWindow` 안에 박아 넣은 로직은 그럴 수 없다. Phase 1 완료 시점에 이 예측도 그대로 맞았다: 테스트가 30개에서 123개로 늘었고, 어시스트 로직은 `SynchronousExecutor`(`libs/inference/service.py:157-172`)를 주입해 **이벤트 루프 없이 결정론적으로** 검증한다. (이후 스펙 확장으로 더 늘었다 — 현재 수치는 [§패키징과 테스트](#패키징과-테스트) 참고.)

## AI 심(seam) — Detection/Mask 경계

이 설계에서 가장 중요한 한 줄:

> **AI 경계는 `Detection`/`Mask` 데이터클래스로만 말한다. `Shape`도 Qt도 절대 노출하지 않는다.**

`libs/inference/`의 코어(`types`·`backend`·`stub`·`registry`)는 PyQt5를 import하지 않는다 — 패키지 `__init__`의 지연 export 테이블 `_LAZY_EXPORTS`(`libs/inference/__init__.py:50-62`)에는 애초에 `service` 항목이 없다: Qt를 아는 유일한 모듈(`service.py`)은 그 테이블에도, `__getattr__`(`:65-74`)이 처리하는 이름에도 없어서 `import libs.inference`는 그 모듈을 아예 import하지 않는다 — "지연 re-export"가 아니라 **"목록에서 처음부터 빠져 있음"**이다. `InferenceService`가 필요한 호출자는 `from libs.inference.service import InferenceService`(`labelImg.py:49`)처럼 **직접** import한다. 모델·백엔드·스레드는 순수 데이터(dataclass, numpy 배열 또는 `RawImage`)만 주고받는다. **UI 경계에서만** 얇은 어댑터(`detection_to_shape` — `libs/assist/suggestion.py:42-58`)가 `Detection → Shape`로 변환한다.

이 심이 있어야 나중에 SAM·폴리곤이 **Phase 1 코드를 다시 쓰지 않고** 접붙는다. 심이 없으면 백엔드가 `Shape`를 알게 되고, `Shape`가 사각형에 묶여 있으므로(→ [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크)) 백엔드까지 사각형에 묶인다.

## 컴포넌트 구성

```mermaid
graph TD
    MW["MainWindow (labelImg.py)<br/>Qt 껍데기 — 소유 + 배선만"]

    MW -->|소유·배선| AC["AssistController (QObject)<br/>libs/assist/controller.py:112"]
    MW -->|소유·배선| IS["InferenceService (QObject)<br/>libs/inference/service.py:186"]

    AC -->|Detection→Shape 어댑터| SUG["libs/assist/suggestion.py:42"]
    AC -->|재사용| CV["Canvas.load_shapes<br/>libs/canvas.py:713"]
    AC -->|재사용| SL["save_labels — 단일 저장 초크포인트<br/>labelImg.py:1033"]
    AC -->|predict 요청| IS
    IS -.->|결과 시그널 (queued)| AC

    IS -->|보유| BE["ModelBackend (ABC)<br/>libs/inference/backend.py:36"]
    IS -->|실행| TP["QThreadPool(maxThreadCount=1)"]
    REG["registry.build_backend(config)<br/>설정 기반, 의존성 없으면 None"] -->|생성| BE

    BE --> STUB["StubBackend — 구현됨<br/>결정론적·의존성 0"]
    BE --> YOLO["YoloOnnxBackend.predict() — 구현됨 (Phase 2)<br/>letterbox → decode/필터 → NMS → 역-letterbox"]
    BE -.-> SAM["MobileSamBackend.segment()/embed() — Phase 6"]

    LF["LabelFile (libs/labelFile.py:30)"] --> RW["PascalVoc / YOLO / CreateML<br/>Reader·Writer (기존)"]
    LF --> COCO["COCOReader / COCOWriter<br/>libs/coco_io.py:120,259"]
```

(점선 = 아직 없는 백엔드.)

읽는 법:

- **`MainWindow` → 컨트롤러**: 생성·소유·시그널 연결. 새 AI 분기를 `MainWindow` 메서드 안에 넣지 않는다.
- **`AssistController` → 기존 코드**: `Canvas.load_shapes`(`libs/canvas.py:713`)와 **단일 저장 초크포인트** `save_labels`(`labelImg.py:1033`)를 **그대로 재사용**한다. 새 캔버스도, 새 저장 경로도 만들지 않았다 — 실제로 `_sync_suggestions`가 `canvas.load_shapes(canvas.shapes + new_shapes)`로 배치 1회 리페인트를 한다(`libs/assist/controller.py:654-689`).
- **`InferenceService` → 백엔드**: 설정 기반 레지스트리가 만들어 준 `ModelBackend` 하나를 들고 `QThreadPool`에서 돌린다.
- **`LabelFile`**: `COCOReader`/`COCOWriter`가 기존 Reader/Writer 옆에 형제로 붙었다.

## 모듈 — 계획 대비 현황

| 모듈 | 책임 (한 줄) | 현황 |
|---|---|---|
| `libs/coco_io.py` | `COCOWriter`(`:120`)/`COCOReader`(`:259`). CreateML의 **데이터셋 병합 패턴**(기존 파일을 읽어 현재 이미지 항목만 갈아끼움)을 따르되, 키는 **basename이 아니라 데이터셋 상대경로**다(`dataset_relative_name` `libs/coco_io.py:34-61` — 아래 설계 검증 참조) | **완료** (1a) |
| `libs/inference/types.py` | `Detection`(`:55`)·`Mask`(`:76`)·`SegPrompt`(`:90`)·`Prediction`(`:103`) 데이터클래스 — AI 심의 어휘. 능동학습 채점 함수 `least_confidence`(`:118`)도 여기 있다(순수 함수, 아직 호출부 없음) | **완료** (1b) |
| `libs/inference/backend.py` | `ModelBackend` ABC(`:36`) — `predict`(추상, `:57`)/`segment`(`:66`)/`embed`(`:79`) + capability 플래그. 선택 의존성 부재는 `MissingDependency`(`:26`)로 신호 | **완료** (1b) |
| `libs/inference/stub.py` | `StubBackend`(`:61`) — 결정론적·의존성 0. **테스트를 구동하는 주체**이며, Phase 1 시점의 유일한 백엔드 | **완료** (1b) |
| `libs/inference/registry.py` | `build_backend(config)` → 백엔드, 의존성이 없으면 예외가 아니라 **`None`**. `DEFAULT_BACKEND`는 `'stub'`가 아니라 **`None`**이다 — 설정에 `model/backend`가 없으면 아무 백엔드도 자동 선택하지 않는다(기본 설치는 백엔드 미설정 상태이고 AI 메뉴는 비활성이다). `'stub'`는 레지스트리에 계속 등록돼 있지만 명시적으로 골라야 쓰인다. ⚠️ 이 파일은 Phase 2에서 `yolo_onnx` 팩토리가 추가되며 이미 한 번 바뀌었고 Phase 6의 `mobile_sam` 팩토리 추가로 다시 바뀔 것이라 **줄 번호를 달지 않는다** | **완료** (1b) · Phase 2에서 `yolo_onnx` 팩토리 추가 완료 |
| `libs/inference/service.py` | `InferenceService`(`:186`): 단일 워커 `QThreadPool`(`:142-154`), 비동기 `predict_async`(`:246`), 결과 시그널(`:195-197`), 이미지별 임베딩 캐시(`:235-242`) | **완료** (1c) |
| `libs/assist/controller.py` | `AssistController`(`:112`): AI 액션(`:260-305`), provisional 수명주기, 신뢰도 임계값(`:431-463`) | **완료** (1c). 능동학습 재정렬은 **완료** (Phase 4) |
| `libs/assist/suggestion.py` | `Detection → Shape` 어댑터(`:42-58`) (UI 경계의 **유일한** 변환 지점) + 수락 시 스타일 전환(`:65-77`) | **완료** (1c) |
| `libs/inference/yolo_onnx.py` | ONNX YOLO 백엔드. 파이프라인은 letterbox → 추론 → **decode(`decode_output` `:374`) → 신뢰도 필터 → NMS(`nms` `:235`) → 역-letterbox(`inverse_letterbox` `:182`, 원본 픽셀 복원)** 순서로 `postprocess`(`:466`)가 묶는다 — **NMS가 역변환보다 먼저** 도는 것은 의도적이다: 모델 자체의 IoU 통계가 모델-입력 좌표 공간에 있고, 스케일이 축마다 하나뿐이라 IoU가 어느 좌표계에서 재도 불변이기 때문이다(근거는 `postprocess` docstring `:477-481`). letterbox 자체의 단일 진실원은 `_letterbox_geometry`(`:129`)이고 `letterbox_params`(`:167`)가 그걸 감싼다. 레이아웃(v8/v5) 자동판별은 `detect_layout`(`:296`), 백엔드 클래스는 `YoloOnnxBackend`(`:606`, `predict` `:947`). `onnxruntime`은 **메서드 안에서 지연 import** | **완료** (Phase 2) |
| `libs/inference/mobile_sam.py` · `mask_utils.py` | MobileSAM 백엔드와 마스크→폴리곤 유틸 | **Phase 6 — 계획** |

## 바뀐 기존 파일 (Phase 1)

| 파일 | 무엇이 바뀌었나 |
|---|---|
| `libs/shape.py` | `provisional`·`confidence`·`shape_type` 필드 추가(`libs/shape.py:57-64`, 기하 종류 상수는 `:28-34`). **`copy()`(`libs/shape.py:218-237`)가 이 필드들을 함께 복사한다** — `copy()`는 필드를 하나씩 나열해 옮기는 화이트리스트라 새 필드는 **말없이 누락될 수 있었다.** `paint()`(`libs/shape.py:104`)는 provisional일 때 점선(`:110-113`)+반투명 채움(`:162-164`)으로 그린다 |
| `libs/labelFile.py` | `LabelFileFormat.COCO` 추가(`libs/labelFile.py:19-24`) + `save_coco_format`(`:56-83`) |
| `libs/constants.py` | `FORMAT_COCO`(`libs/constants.py:15-18`) + `SETTING_MODEL_BACKEND`/`SETTING_MODEL_PATH`/`SETTING_CONF_THRESHOLD`(`:22-26`) |
| `labelImg.py` | 컨트롤러 배선(`:444-446`), `&AI` 메뉴(`:478`·`:504`), **저장 초크포인트의 provisional 필터 1줄**(`:1040`), `get_format_meta` COCO 분기 + **방어적 기본값**(`:287-302`), COCO 데이터셋 레인(`coco_dataset_target` `:670-687`, Import/Export COCO 액션 `:255-260`), `.json` 내용 스니핑 디스패치(`load_json_by_filename` `:2267-2277`) |
| `setup.py` | Python 지원 하한을 `>=3.7`로 올렸다(dataclass·f-string 요구, Phase 1). **`extras_require`의 `'ai'`(`onnxruntime>=1.15`·`numpy`)는 Phase 2에서 추가됐다**(`setup.py:27`) |
| `.github/workflows/ci.yml` | 코어 잡은 여전히 AI 의존성 없이 green. **선택적 `[ai]` 잡(`test-ai`)은 Phase 2에서 추가됐다**(`.github/workflows/ci.yml:44-68`) |

## 핵심 인터페이스

아래는 **구현된 것**이다(`libs/inference/types.py:55-115`). 설계 스케치와 달라진 세 곳은 주석으로 표시했다 — 셋 다 구현 중에 더 단순한 쪽으로 정리된 것이다.

```python
# libs/inference/types.py:55
@dataclass(frozen=True)
class Detection:
    label: str                  # 클래스 '이름'. id→이름 매핑은 백엔드의 책임
    box: Box                    # (x1, y1, x2, y2) — 원본 이미지 픽셀
    score: float                # 0..1
    class_id: Optional[int] = None   # provenance용. 없는 백엔드도 있다

# libs/inference/types.py:76
@dataclass(frozen=True)
class Mask:
    polygon: List[Point]        # 외곽 컨투어, 원본 이미지 픽셀
    score: float = 1.0
    # 설계에는 label이 있었으나 뺐다: 대화형 분할(SAM)은 사용자가 가리킨
    # '그 객체'를 돌려줄 뿐 클래스를 주장하지 않는다. 라벨은 UI가 붙인다.

# libs/inference/types.py:90
@dataclass                      # frozen 아님 — 프롬프트는 클릭마다 제자리에서 자란다
class SegPrompt:
    points: List[PromptPoint] = field(default_factory=list)  # (x, y, 1=fg/0=bg)
    box: Optional[Box] = None
    # 설계의 points/labels 두 리스트를 한 리스트의 3-튜플로 합쳤다:
    # 길이가 어긋날 수 없는 표현이 어긋날 수 있는 표현보다 낫다.

# libs/inference/types.py:103
@dataclass                      # frozen 아님 — 능동학습이 uncertainty만 나중에 채운다
class Prediction:
    image_path: str
    detections: List[Detection] = field(default_factory=list)
    uncertainty: Optional[float] = None   # None = 아직 채점 안 함
    # 설계의 masks 필드는 넣지 않았다 — 검출과 분할은 호출 경로가 다르고,
    # 쓰이지 않는 필드를 미리 파 두면 계약이 흐려진다. Phase 6에서 필요하면 그때 붙인다.
```

```python
# libs/inference/backend.py:36
class ModelBackend(ABC):
    name: str = 'base'
    supports_detection: bool = False
    supports_segmentation: bool = False
    class_names: List[str] = []     # 클래스 id 순서

    @abstractmethod
    def predict(self, image: Any) -> List[Detection]: ...          # :57
    def segment(self, image, prompts: SegPrompt, embedding=None) -> Mask: ...  # :66
    def embed(self, image) -> Any: ...                             # :79
    def close(self) -> None: ...                                   # :87
```

`conf`는 `predict()`의 인자가 아니라 **백엔드 생성 시 설정**이다(`registry.build_backend`의 config). 이유가 있다: 컨트롤러는 백엔드를 `conf_threshold=0.0`으로 만들어(`libs/assist/controller.py:237-248`) **모델이 찾은 것을 전부 받아 두고**, 화면에 무엇을 보일지는 UI 임계값이 정한다. 백엔드가 미리 걸러 버리면 사용자가 슬라이더를 낮출 때마다 **추론을 다시 돌려야** 한다.

설계에 있던 `supports_embedding` 플래그는 빼고 `embed()`의 기본 구현이 `NotImplementedError`를 던지게 뒀다 — 플래그 하나를 더 두는 대신 임베딩은 `supports_segmentation`을 따라간다.

### 좌표 규약 (이 설계의 두 번째 불변식)

> **`Detection.box`·`Mask.polygon`은 항상 원본 이미지 픽셀 좌표다** — 기존 Reader들이 내놓는 것과 **완전히 동일한 좌표계**다.

기존 Reader는 `(label, points, None, None, difficult)` 5-튜플을 원본 픽셀로 돌려주고(예: `libs/create_ml_io.py:125-133`, `COCOReader.add_shape` `libs/coco_io.py:330-336`), `MainWindow.load_labels`(`labelImg.py:977`)가 그걸 `Shape`로 복원한다. 예측 결과가 **같은 좌표계**를 쓰면 UI는 예측을 **Reader 출력과 구별할 필요가 없다.** 실제로 `detection_to_shape`(`libs/assist/suggestion.py:42-58`)에는 스케일 연산이 **한 줄도 없다** — 네 모서리를 그대로 `QPointF`로 올린다.

letterbox(패딩·스케일) 역변환은 **백엔드 안에서** 끝난다 — `libs/inference/yolo_onnx.py`(Phase 2)가 모델 입력 텐서 좌표를 원본 픽셀로 되돌린 뒤에야 `Detection`을 만든다. **UI는 letterbox를 모른다.** 이 역변환을 UI로 새어 나가게 두면, 백엔드가 바뀔 때마다 UI가 같이 바뀐다. `StubBackend`(`libs/inference/stub.py:61-139`)가 이 계약을 **테스트에서 강제**한다: 박스가 이미지 크기의 순수 함수라서, 중간 계층이 몰래 스케일을 먹이면 숫자가 즉시 틀어진다.

## provisional Shape 수명주기

예측은 `provisional=True` + `confidence`가 붙은 `Shape`(`libs/shape.py:57-64`)가 되어 캔버스에 올라간다. 세 가지가 걸려 있다.

**1. 캔버스에만 올리면 안 된다.** `AssistController`는 각 provisional 도형을 **반드시 `add_label`(`labelImg.py:946`)로도 등록**한다(`libs/assist/controller.py:684-688`). 선택 처리 경로가 `shapes_to_items`(`labelImg.py:939`에서 채워짐)를 조회하고, `remove_label`(`labelImg.py:945-961`)은 `self.shapes_to_items[shape]`를 **가드 없이 인덱싱**한다(`labelImg.py:949`) — 캔버스에만 있는 도형을 선택/삭제하면 `KeyError`다.

반대 방향의 배선도 필요했다: `remove_label`은 **자기가 지운 모든 도형을 컨트롤러에 되돌려 보고**한다(`labelImg.py:960-961` → `discard_shape` `libs/assist/controller.py:635-652`). 사용자가 제안 박스 하나를 평범한 "Delete RectBox"로 지웠을 때 컨트롤러가 그 사실을 모르면, 죽은 `Shape` 참조를 들고 있다가 다음 임계값 변경에서 그것을 두 번째로 지우려 든다(`list.remove(x)` → `ValueError`).

**2. 디스크에는 절대 안 나간다 — 필터는 단 하나.** 모든 저장 경로(`Ctrl+S` → `save_file` `labelImg.py:1716`, 다음/이전 이미지 자동저장 → `open_next_image`의 `auto_saving` 분기 `labelImg.py:1673-1676`와 `open_prev_image`의 같은 분기 `labelImg.py:1648-1651`, verify → `verify_image` `labelImg.py:1621`, Export COCO)는 전부 `_save_file`(`labelImg.py:1770`)을 거쳐 **`save_labels`(`labelImg.py:1033`) 하나로 수렴**한다. 직렬화 대상은 그 안의 한 줄이다:

```python
# labelImg.py:1040 — THE provisional filter
shapes = [format_shape(shape) for shape in self.canvas.shapes if not shape.provisional]
```

**이 필터 하나가 모든 포맷·모든 저장 경로를 동시에 막는다.** 저장 경로마다 필터를 흩뿌렸다면 하나를 빠뜨렸을 때 확정하지 않은 AI 추측이 디스크에 새어 나갔을 것이다.

**3. 수락/거부.** 단축키는 앱이 실제로 바인딩한 전체 단축키를 덤프해 충돌을 피해서 골랐다(`libs/assist/controller.py:50-61` — 이 포크는 이미 이중 바인딩 버그를 한 번 겪었고, Qt는 모호한 단축키를 경고 없이 **양쪽 다 비활성화**한다).

| 동작 | 단축키 | 결과 |
|---|---|---|
| **Auto-label Image** | `Ctrl+I` | 현재 이미지로 추론 요청(`auto_label_image` `libs/assist/controller.py:481-502`). 재실행은 이전 라운드를 **교체**한다 |
| **수락 (전체)** | `Ctrl+Return` | `provisional=False`로 클리어 + 라벨 색 환원 + `set_dirty()`(`labelImg.py:732`) — 이제 평범한 도형이므로 다음 저장에 포함된다 (`accept_all` `libs/assist/controller.py:691-710`) |
| **거부 (전체)** | `Ctrl+Backspace` | 도형 제거 (`reject_all` `libs/assist/controller.py:712-723` → `delete_selected_shape`) |
| **개별 거부** | `Delete` (기존 "Delete RectBox") | 별도 액션 없음 — 위 `discard_shape` 배선이 그 삭제를 관측한다 |
| **신뢰도 임계값** | 메뉴 슬라이더 (`QWidgetAction`) | **모델을 다시 돌리지 않고** 화면의 제안만 재필터링한다(`set_threshold` `libs/assist/controller.py:460-477` → `_sync_suggestions` `:654-689`) |

임계값이 슬라이더인 이유는 그것이 **입력하는 값이 아니라 탐색하는 값**이기 때문이다: 사용자는 끌면서 제안이 나타났다 사라지는 것을 본다. 이게 성립하려면 백엔드가 `conf=0.0`으로 전부 찾아 둬야 한다(위 §핵심 인터페이스).

**디스크 포맷 계약은 바뀌지 않았다.** provisional·confidence는 **메모리 전용 상태**다. Reader/Writer의 5-튜플 계약(`libs/labelFile.py`, 각 `*_io.py`)은 그대로다 — 즉 AI 어시스트를 켜도 산출물은 기존 VOC/YOLO/CreateML/COCO 파일과 **바이트 단위로 같은 종류**다.

## 스레딩

`InferenceService`는 **`QThreadPool(maxThreadCount=1)`** 을 소유한다(`ThreadPoolExecutor` `libs/inference/service.py:142-154`). 워커를 1개로 고정하는 이유는 두 가지다: ONNX Runtime의 `Session.Run`은 **한 세션에 대해 동시 호출이 안정적으로 안전하지 않고**, CPU 추론은 워커를 늘려도 코어를 나눠 쓸 뿐 이득이 없다.

규칙 세 가지 — 셋 다 구현돼 있다:

1. **워커는 순수 데이터만 받는다.** UI 스레드에서 `to_model_image`(`libs/inference/service.py:113-139`)가 `QImage`를 **소유권 있는 스냅샷**으로 바꾼다: numpy가 있으면 HxWx3 uint8 배열, 없으면 `RawImage`(`:56-85`)다. 후자가 있는 이유는 기본 설치에 numpy가 없어도 스텁 백엔드가 돌아야 하기 때문이다. Qt가 행마다 넣는 4바이트 패딩은 여기서 벗겨진다(`_rgb888_bytes` `:88-110`) — 워커는 어떤 Qt 객체도 만지지 않는다.
2. **결과는 queued 시그널로 돌아온다.** `predictionReady`/`predictionFailed`(`libs/inference/service.py:195-197`)는 워커 스레드에서 emit되지만 `InferenceService`가 UI 스레드에 살기 때문에 Qt의 `AutoConnection`이 **queued로 배달**한다. 슬롯이 UI 스레드에서 실행되므로 거기서 `Shape`를 만들어도 안전하다.
3. **stale 결과 폐기.** 모든 요청은 자기 `image_path`를 들고 다니고(`predict_async` `libs/inference/service.py:246-275`), 컨트롤러의 `_is_current`(`libs/assist/controller.py:600-609`)가 **현재 열린 파일과 다르면 버린다.** 추론 중에 사용자가 다음 이미지로 넘어가도 **엉뚱한 이미지에 박스가 꽂히는 일이 없다.** 이 가드가 없으면 사용자가 빠르게 넘길 때 조용히 오염되고, 그 오염은 다음 저장에서 **정답 파일에 기록**된다.

**이미지별 임베딩 캐시는 Phase 1에 만들어 뒀다**(`libs/inference/service.py:207`, 접근자 `:235-242`) — 채우는 건 SAM뿐이지만, 나중에 캐시를 끼워 넣으려면 서비스의 요청/결과 배관을 다시 건드려야 한다. 백엔드를 교체하면 캐시는 무효화된다(`set_backend` `:214-220`).

## 패키징과 테스트

- **지연 import.** AI 관련 import는 모듈 최상단이 아니라 함수 안에 둔다(`registry`의 백엔드 팩토리 `_build_stub`, `to_model_image`의 numpy `libs/inference/service.py:131-134`). `build_backend()`는 의존성이 없으면 예외가 아니라 **`None`을 돌려주고**(`libs/inference/registry.py`) 컨트롤러가 "`pip install -e ".[ai]"`" 힌트와 함께 AI 액션을 꺼 둔다(`NO_BACKEND_CONFIGURED_HINT`/`BACKEND_UNAVAILABLE_HINT` `libs/assist/controller.py:85-92`, `refresh_actions` `libs/assist/controller.py:369-440`) — AI 없이 실행해도 앱은 정상 동작한다.
- **`StubBackend`가 테스트를 구동한다.** 모델 파일도, `onnxruntime`도 없이 어시스트 로직(수락/거부, provisional 필터, stale 폐기)을 결정론적으로 검증한다 — `tests/test_inference_core.py`, `tests/test_assist.py`. 테스트는 30개에서 **280개**(테스트 파일 14개)로 늘었고 코어 매트릭스는 AI 의존성 없이 green이다.
- **COCO 라운드트립 테스트**(`tests/test_coco_io.py`)는 `tests/test_io.py`의 기존 패턴을 따른다.
- **Python 하한은 `>=3.7`** 로 올라갔다(`setup.py`) — 코어가 dataclass를 쓰기 때문이다.
- **선택적 `[ai]` extra**(`onnxruntime`·`numpy`, `setup.py:27`)와 CI의 선택적 `[ai]` 잡(`test-ai`, `.github/workflows/ci.yml:44-68`)은 **Phase 2에서 들어왔다.** 현재 기본 설치는 여전히 `pyqt5`+`lxml`뿐이다.

## 확장성 증명 — 심이 진짜인지

설계의 심(seam)이 진짜인지는 "**나중 기능이 이 심 위에서 몇 줄로 되는가**"로만 증명된다. 아래 중 능동학습(Phase 4)은 **구현 완료**됐다 — 아래 서술은 그 as-built 요약이다. SAM(Phase 6)은 **아직 구현되지 않았다** — 심의 설계 근거로 남긴다.

### 능동학습 (uncertainty 우선 정렬) — Phase 4 (구현 완료)

1. 폴더 전체를 배치 추론한다 — `AssistController.score_folder`(`libs/assist/controller.py:792-842`)가 `self.app.m_img_list`의 이미지를 **한 번에 하나씩** 돌린다: UI 스레드에서 `QImageReader`로 로드(`_load_model_image`, `:962-981`) → `predict_async` → 결과가 오면 다음 이미지로(`_batch_step`/`_advance_batch`, `:868-942`). 단일 워커 풀이라 자연히 직렬화되고, 매 스텝 이벤트 루프로 돌아가므로 폴더가 커도 UI가 멈추지 않는다. 두 번째 트리거(같은 액션)로 **취소** 가능(`cancel_batch_scoring`, `:844-866`). **로드 실패가 연속될 때의 재귀 방지**: `_batch_step`이 이미지를 못 읽으면(파일이 손상됐거나, scan 이후 다른 프로세스가 지웠거나) 최대 불확실성 1.0으로 기록하고 다음 이미지로 넘어가는데, 이 "다음 스텝" 호출을 `_advance_batch(..., synchronous=False)`로 `QTimer.singleShot(0, ...)`에 넘겨 **이벤트 루프의 새 프레임에서** 실행한다(`:919-942`) — 그 자리에서 바로 다음 `_batch_step()`을 호출(동기 재귀)했다면, 읽을 수 없는 파일이 수백 개 연속되는 폴더에서 콜 스택이 끝없이 쌓여 `RecursionError`로 배치(그리고 잠재적으로 앱 전체)가 죽었을 것이다(교차엔진 리뷰로 발견 — 회귀 테스트: `tests/test_active_learning.py`의 `TestBatchLoadFailureDoesNotRecurse`).
2. `uncertainty = 1 - mean(top-k scores)`를 이미지별로 기록한다. **채점 함수는 이미 있던 것을 그대로 쓴다**: `least_confidence`(`libs/inference/types.py:118-149`) — 순수 함수라 모델 없이 테스트되며, 검출이 0개인 이미지는 **최대 불확실성 1.0**을 돌려줘 리뷰 큐의 앞으로 온다(모델이 눈이 먼 케이스야말로 사람이 봐야 한다). 호출부는 `on_prediction_ready`/`on_prediction_failed`가 각 `predict_async` 요청을 **디스패치 순서**로 태깅해 배치 결과와 대화형 결과를 구분한다(`_dispatch_request`/`_pop_request_kind`, `libs/assist/controller.py:504-555`) — 경로(path) 일치만으로 구분하던 이전 방식은, Score Folder 시작 시점에 아직 안 끝난 대화형(Ctrl+I) 요청이 있고 그 경로가 배치의 첫 대상과 같으면 둘을 뒤바꿔 처리하는 교차 레이스가 있었다(교차엔진 리뷰로 발견, 아래 §알려진 결함이 아니라 이미 고쳐짐 — `tests/test_active_learning.py`의 `test_interactive_request_outstanding_when_a_batch_starts_is_not_misattributed`). **취소가 항상 닿는다**: Score Folder 액션은 배치가 도는 동안 `has_folder or batch_running`으로 활성 상태를 유지한다(`refresh_actions`, `:412-413`) — 배치 도중 남은 이미지가 전부 분류돼 `m_img_list`가 비어도, 실행 중인 배치를 멈출 유일한 컨트롤이 회색으로 죽지 않는다(회귀 테스트: `TestScoreFolderStaysEnabledWhenFolderEmptiesMidBatch`).
3. **기존 `m_img_list`를 재정렬**하고(`sort_by_uncertainty`/`_reorder`, `libs/assist/controller.py:994-1017`, `:1090-1115`), 파일 리스트를 `import_dir_images`가 이미 쓰던 그 루프를 흡수한 `refresh_file_list`(`libs/assist/controller.py:1161-1208`)로 다시 채운다. 위젯은 항상 통째로 비우고 다시 채우므로(행 i가 `m_img_list[i]`를 계속 가리키게 하려면 그래야 한다), 다시 채운 **직후** 현재 열린 이미지의 행을 재선택한다(`:1203-1208`) — 그러지 않으면 진짜 Open Directory 직후 `load_file`이 아직 빈 위젯에 하이라이트를 시도했다가 그대로 사라진다(회귀 테스트: `TestFileListSelectionAfterOpenDirectory`). 순위/"채점 N" 총계(`_ranks()`, `:1242-1269`)도 `_uncertainty`와 `m_img_list`의 **교집합**만 센다 — 분류로 폴더를 떠난 이미지의 점수는 undo를 위해 `_uncertainty`엔 남지만(지우지 않음), 더 이상 폴더에 없는 동안은 순위/총계에 끼지 않는다(회귀 테스트: `TestRankAndTotalExcludeAbsentImages`).

탐색 코드는 `m_img_list`를 **인덱스로만** 읽으므로(`self.m_img_list[self.cur_img_idx]` — `open_prev_image` `labelImg.py:1646`, `open_next_image` `labelImg.py:1671`) **변경이 0줄**이다. 포크의 기존 `g`/`b` 분류 트리아지(`classify_current_image` `labelImg.py:1805`)도 **그대로** 재사용되며, 이제 불확실성 순으로 정렬된 스트림 위에서 돈다. 유일한 함정은 `classify_current_image`/`delete_image`/`undo_classify`가 같은 폴더를 다시 스캔할 때(`import_dir_images`, `labelImg.py:1588`)마다 `scan_all_images`가 파일시스템 순서로 `m_img_list`를 **무조건 재생성**한다는 것 — 정렬이 걸려 있으면 매번 g/b 한 번에 정렬이 풀렸을 것이다. `reapply_sort_if_active`(`libs/assist/controller.py:1054-1088`)가 같은 정렬 키로 다시 정렬해 이 재스캔을 흡수한다(`reset_active_learning=False` 경로에서만; 진짜 새 폴더를 열 때는 `on_directory_scanned`(`:1118-1144`)가 정상적으로 초기화한다).

### SAM (프롬프트 기반 분할) — Phase 6

- `MobileSamBackend`가 **같은 `ModelBackend` ABC** 위에서 `segment()`/`embed()`를 구현한다 — 둘 다 ABC에 **이미 선언돼 있다**(`libs/inference/backend.py:66`·`:79`).
- 캔버스 클릭이 `SegPrompt`(`libs/inference/types.py:90`)가 된다.
- **캐시된 임베딩**(`libs/inference/service.py:235-242` — 슬롯은 이미 있고 비어 있다) 덕분에 클릭마다 가벼운 디코더만 돈다(무거운 인코더는 이미지당 1회).
- 결과 마스크는 폴리곤 `Shape`가 되어 **동일한 수락/거부 수명주기**를 탄다.

**새 컨트롤러 없음, 새 스레딩 없음, 새 저장 경로 없음.** 이게 심이 진짜라는 뜻이다. (단, 폴리곤 자체는 공짜가 아니다 — [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크).)

## 단계별 순서

| 단계 | 내용 | 규모 | 상태 |
|---|---|---|---|
| **1a** | COCO I/O (`libs/coco_io.py`) | M | ✅ **완료** (`a32acd3`) |
| **1b** | 추론 코어: `types`/`backend`/`stub`/`registry` — **순수 파이썬, Qt 없음** | S | ✅ **완료** (`a32acd3`) |
| **1c** | 얇은 수직 슬라이스: `InferenceService` + `AssistController` + `Shape` 필드 + provisional 점선 렌더 + AI 액션 | L | ✅ **완료** (`a32acd3`) |
| **2** | `YoloOnnxBackend` + `[ai]` extra + `test-ai` CI 잡 | M | ✅ **완료** |
| **3** | 수락/거부/편집 다듬기 + Auto-label Folder | M | 계획 (신뢰도 슬라이더는 1c에서 앞당겨 들어옴) |
| **4** | 능동학습 (uncertainty 정렬) | S/M | ✅ **완료** |
| **5** | 폴리곤/키포인트 (**사각형 결합 해체**) | L | 계획 |
| **6** | MobileSAM | L | 계획 |

1b가 **Qt 없이** 끝나는 게 중요했다 — 심이 실제로 Qt와 분리돼 있는지를 이 단계에서 강제로 증명한다. 실제로 그렇게 끝났다: `libs/inference/`의 코어 4모듈은 PyQt5를 import하지 않고, Qt를 아는 `service.py`는 패키지 `__init__`의 지연 export 밖에 있다(`libs/inference/__init__.py:47-62`).

## 리스크와 트레이드오프

각 항목에 **현재 상태**를 붙였다. ✅ = Phase 1에서 닫힘, ⬜ = 여전히 열려 있음.

### 1. God-object 성장

> **⬜ 열림** — 영구적 규율 리스크.

컨트롤러 2개로 완화하지만, **진짜 리스크는 규율이다.** "여기 한 줄만"이라며 AI 분기를 `MainWindow` 메서드 안에 넣기 시작하면 분류 기능(`labelImg.py:1752-2035`)이 그랬듯 몇 백 줄이 다시 쌓인다. `MainWindow`가 만져도 되는 것: 컨트롤러 **생성·소유·배선**, 액션을 메뉴와 `onLoadActive`(`labelImg.py:468-471`)에 등록, 저장 초크포인트의 **필터 1줄**. 그 이상은 컨트롤러 안이다.

Phase 1은 이 선을 지켰다(위 §왜 새 클래스인가의 배선 목록). Phase 2~6에서 다시 지켜야 한다.

### 2. 좌표계 매핑

> **✅ 닫힘** (Phase 2) — 이 리스크는 **실제로 터졌다.** 아래는 실전에서 무엇이 어떻게 틀렸고 어떻게 잡았는지의 기록이다.

예측대로 역-letterbox는 조용히 틀렸다: **forward 방향(letterbox 페인트)은 pad를 정수 픽셀로 반올림해 캔버스에 붙였는데, inverse 방향(박스 역변환)은 반올림하지 않은 float pad를 그대로 빼고 있었다.** 원본 크기가 모델 입력 크기로 딱 나눠떨어지지 않으면(예: 1000×995 → 640×640은 위아래로 3행이 남아 이상적인 pad는 1.5) 두 방향이 같은 숫자를 두 번 따로 계산하다가 어긋난 것이다 — 실제 붙인 자리(정수로 내림된 행)와 되돌릴 때 뺀 자리(1.5)가 반 픽셀만큼 달라, **홀수 padding이 나오는 이미지마다 모든 박스가 `0.5 / scale` 원본 픽셀만큼 일정하게 밀렸다.** 예외 없이 조용히 틀렸다는 점에서 위 예측 그대로였다.

**고친 방법은 이중 계산 자체를 없애는 것**: `_letterbox_geometry()`(`libs/inference/yolo_onnx.py:129-164`) 하나가 `(scale, new_w, new_h, pad_x, pad_y)`를 계산하는 **유일한 소스**가 됐고, pad는 짝을 정수로 내림해(`libs/inference/yolo_onnx.py:162-163`) 홀수 나머지를 하단/우측에 준다. forward 쪽 `_letterbox()`는 이 값을 그대로 받아 캔버스에 붙이고(`libs/inference/yolo_onnx.py:910`, 실제 페인트는 `:939-941`) 별도 산술을 하지 않으며, inverse 쪽 `letterbox_params()`/`inverse_letterbox()`(`libs/inference/yolo_onnx.py:167-179`, `:182-216`)도 같은 함수가 반환한 같은 pad를 받는다. 한 값을 두 번 다시 계산하지 않으므로 두 방향이 서로 어긋날 수 없다 — 이 계약은 모듈 docstring에도 못박혀 있다(`libs/inference/yolo_onnx.py:48-57`).

**테스트는 계산된 값이 아니라 실제로 붙인 텐서를 읽어서 검증한다** — 이게 핵심이다. 순수-파이썬 라운드트립 테스트(`letterbox_params`를 `inverse_letterbox`로 되돌리는 식)는 자기 자신하고만 대조하므로 애초에 이 버그를 잡지 못했을 것이다. 대신 `content_rect()`(`tests/test_yolo_onnx.py:797-813`)는 흰 이미지를 회색 패딩 위에 붙인 뒤 **텐서에서 흰 픽셀이 시작하는 좌표를 직접 읽어** "진짜로 붙은 자리"를 구한다. `test_the_paste_offset_is_exactly_the_pad_the_inverse_subtracts`(`tests/test_yolo_onnx.py:815-828`)는 홀수/짝수 padding이 섞인 다섯 원본 크기(`1000x995`, `995x1000`, `1000x993`, `1280x720`, `65x33`)에서 이 실측 좌표가 `letterbox_params`가 돌려주는 pad와 **정확히 일치**함을 규정한다. `test_an_odd_pad_maps_the_pasted_image_back_onto_the_whole_image`(`tests/test_yolo_onnx.py:830-858`)는 한 걸음 더 나가 end-to-end로 확인한다: 실제 페인트된 사각형 좌표를 읽어 그 자리에 정확히 박스 하나를 예측하게 만든 뒤, `predict()`가 돌려준 박스가 **원본 이미지 전체**(`(0, 0, orig_w, orig_h)`)로 정확히 복원되는지 본다 — 고치기 전에는 이 케이스에서 위쪽 경계가 `y=0` 대신 `y=0.78`로 돌아왔다.

닫힌 근거를 요약하면: 계약(`libs/inference/types.py:10-28`, `detection_to_shape`에 스케일 연산 0줄)은 애초부터 옳았고, **깨진 것은 그 계약을 만족시키는 산술의 구현**이었다. `StubBackend`는 스케일이 아예 없으므로 이 버그를 절대 잡을 수 없는 종류였고, `YoloOnnxBackend`가 들어오고 나서야(위 예측대로) 실전 이미지 기반 테스트로 발견·수정됐다.

### 3. 스레딩 함정

> **✅ 닫힘** (Phase 1c).

단일 워커 풀, 순수 데이터 워커, queued 시그널, **현재 파일 경로 기준 stale 폐기** — 네 가지가 모두 구현됐다(위 §스레딩). 특히 stale 폐기가 없으면 "빨리 넘기는 사용자"에게만 재현되는, 디버깅하기 최악인 오염 버그가 된다.

### 4. 모델 라이선스 (법적 리스크)

> **✅ 닫힘** (Phase 2) — 결정: 가중치를 아예 싣지 않는다.

Ultralytics YOLOv5/v8 **가중치는 AGPL-3.0**이고, 이 앱의 MIT 라이선스와 **충돌한다.** 검토했던 두 대응 중:

- 관대한 라이선스 모델을 싣는 방안(예: YOLOX / Apache-2.0)은 채택하지 않았고,
- **가중치를 아예 싣지 않는 쪽으로 결정됐다** — 사용자가 설정에서 자기 `.onnx` 경로를 가리킨다.

`data/models/README.md`가 이 결정과 전체 근거(AGPL-3.0 vs MIT 라이선스 충돌 표, permissive 대안
목록인 YOLOX/YOLOv6/RT-DETR)를 문서화한다. 가중치는 기본 설치에 들어가지 않으며, 설정 키
`SETTING_MODEL_PATH`(`libs/constants.py:25`)는 이 결정을 전제로 이미 뚫려 있다.

### 5. 기본 설치 무게

> **✅ 닫힘** (Phase 2) — 3요소 모두 완료.

`[ai]` extra + 지연 import + **StubBackend만 쓰는 코어 테스트** 세 가지가 함께 있어야 `pip install labelImg`가 가벼운 채로 남는다. 하나라도 빠지면 `onnxruntime`(수백 MB)이 기본 의존성으로 새어 들어온다.

셋 다 됐다: **지연 import와 스텁 기반 코어 테스트는 Phase 1**부터(기본 의존성은 여전히 `pyqt5`+`lxml`뿐이고, 전체 280개 테스트 중 코어 252개는 numpy조차 없이 돈다 — 나머지 28개는 onnxruntime/numpy가 필요해 **skip**되지 error는 아니다), **`[ai]` extra(`onnxruntime`·`numpy`)는 Phase 2**에서 추가됐다(`setup.py:27`).

### 6. 사각형 결합의 벽 — 가장 비싼 미래 리워크

> **⬜ 열림 — 이 로드맵에서 가장 비싼 리워크이고, Phase 1은 이 벽을 전혀 건드리지 않았다.**
>
> ⚠️ **"`shape_type` 필드만 추가하면 마스크→폴리곤이 된다"는 말은 거짓이다.** 폴리곤·키포인트·마스크는 필드 하나로 해금되지 않는다.

Phase 1이 `Shape.shape_type`(`libs/shape.py:64`)과 기하 종류 상수(`libs/shape.py:28-34`)를 넣었지만, **그건 이름표일 뿐 아무것도 해금하지 않았다.** 사각형 가정은 여전히 코드 전반에 **퍼져 있다**:

| 위치 | 사각형 가정 |
|---|---|
| `Shape.reach_max_points` / `add_point` (`libs/shape.py:84-91`) | **4점 상한** — 5번째 점은 조용히 버려진다 |
| `Canvas.handle_drawing` (`libs/canvas.py:322-333`) | 클릭 2번으로 **4점 사각형을 합성** |
| `Canvas.bounded_move_vertex` (`libs/canvas.py:400-434`) | `(index + 2) % 4` — **모서리 산술** |
| `Canvas.move_one_pixel` (`libs/canvas.py:647-672`) | `points[0]`~`points[3]`을 **직접 4개 다** 이동 |
| `Canvas.move_out_of_bound` (`libs/canvas.py:676-678`) | `zip(points, [step] * 4)` — **4점 하드코딩** |
| 상태바 크기 표시 (`libs/canvas.py:186-187`, `244-245`) | `h_shape[1]` / `h_shape[3]`을 **인덱스로** 집어 폭·높이 계산 |
| 모든 박스 writer | 점들을 bbox로 **환원** — `LabelFile.convert_points_to_bnd_box`(`libs/labelFile.py:182-205`), CreateML은 `points[0][0]`/`points[1][0]`/`points[2][1]`을 직접 인덱싱(`libs/create_ml_io.py:42-45`) |
| COCO writer (신규) | 같은 환원을 거친다 — `save_coco_format`(`libs/labelFile.py:79`)도 `convert_points_to_bnd_box`를 호출한다 |

`detection_to_shape`(`libs/assist/suggestion.py:42-58`)도 네 모서리를 만들어 `Shape.RECT`로 닫는다. 따라서 **Phase 1~4는 사각형 전용으로 남는다.** 이 일반화는 **Phase 5/6의 몫**이며, 그 자체로 큰 작업이다. 이걸 "필드 하나"로 착각하고 일정을 잡는 것이 이 로드맵에서 가장 비싼 실수다 — 그리고 Phase 1이 끝난 지금도 **그 벽은 한 뼘도 낮아지지 않았다.**

### 7. COCO 저장 경로가 다르다 — 데이터셋 레인

> **✅ 닫힘** (Phase 1a) — 다만 설계 항목 하나가 구현 중에 **뒤집혔다**(아래 자동저장).

VOC/YOLO/CreateML은 **이미지별 사이드카 경로를 유도해서 현재 이미지 하나만 직렬화**한다(`save_file` `labelImg.py:1716-1740`가 이미지 stem으로 경로를 만들고, `save_labels` `labelImg.py:1033-1094`이 확장자를 붙인다). **이 경로로는 데이터셋 하나짜리 `instances.json`을 만들 수 없다.**

따라서 **COCO는 데이터셋 수준의 Import/Export 레인**으로 뒀다:

- **명시적 데이터셋 대상**을 갖는다 — `coco_dataset_target()`(`labelImg.py:670-687`), 기본값은 저장 폴더의 `annotations.json`(`COCO_DEFAULT_DATASET_NAME` `libs/coco_io.py:17`). Import/Export COCO... 액션(`labelImg.py:255-260`)이 이를 덮어쓴다.
- 저장 분기는 현재 이미지 항목을 **그 고정된 파일에 병합**한다 — `save_labels`의 COCO 분기(`labelImg.py:1058-1069`)가 호출자가 만든 사이드카 경로를 **버리고** 데이터셋 경로로 갈아끼우며, `COCOWriter.save`(`libs/coco_io.py:216-256`)가 read-modify-write로 이 이미지의 항목만 교체한다.
- **"COCO는 이미지별 자동저장에 연결하지 않는다"는 설계 항목은 폐기됐다.** 대신 자동저장도 **같은 데이터셋 파일로 병합**된다(`save_file`의 COCO 선분기 `labelImg.py:1684-1694`). 이유: 자동저장만 예외로 두면 사용자는 "저장했는데 없다"를 겪는다. `<stem>.json` 사이드카는 여전히 **한 개도 만들어지지 않는다** — 원래 금지의 실제 목적은 그것이었다.
- 부작용 하나는 공개해 둔다: COCO 스키마에 `verified`·`difficult` 슬롯이 없어 두 플래그는 **저장되지 않는다.** `verify_image`가 그 사실을 상태바로 알린다(`labelImg.py:1606-1612`).

### 8. COCO / CreateML의 `.json` 충돌

> **✅ 닫힘** (Phase 1a).

둘 다 `.json`을 쓰는데, 로드 디스패치는 원래 **확장자로만** 분기했다(`show_bounding_box_from_annotation_file` `labelImg.py:1342-1379`).

해결(구현됨): **모든 `.json` 디스패치를 `load_json_by_filename`(`labelImg.py:2267-2277`) 하나로 모으고, `CreateMLReader`보다 먼저 내용을 스니핑**한다(`is_coco_json` `libs/coco_io.py:94-102`).

- 최상위가 `images`/`annotations`/`categories`를 가진 **dict → COCO** (`is_coco_dict` `libs/coco_io.py:82-91`)
- 최상위가 **list → CreateML**

이게 필수였던 이유: `CreateMLReader`는 **`ValueError`만 잡는다**(`libs/create_ml_io.py:102-105`). COCO dict를 물리면 `parse_json`의 `for image in output_list`(`libs/create_ml_io.py:119-120`)가 **dict의 키(문자열)를 순회**하다 `image["image"]`에서 `TypeError`를 낸다 — `ValueError`가 아니므로 **잡히지 않고 앱까지 올라간다.** 즉 `CreateMLReader`는 COCO 파일을 **우아하게 거부하지 못한다.**

거꾸로도 막았다: 데이터셋 json이 이 이미지를 **모르면** COCO 로더는 포맷을 바꾸지 않고 `False`를 돌려준다(`load_coco_json_by_filename` `labelImg.py:2256-2281`) — 저장 폴더에 우연히 놓인 남의 `annotations.json`이 앱을 COCO로 끌고 가지 못한다.

### 9. pickle된 enum + 기본값 없는 `get_format_meta` — 다음 실행 때 죽는다

> **✅ 닫힘** (Phase 1a) — COCO 분기 + 방어적 기본값 + unpickle 타입 검사.

- `label_file_format`은 **설정에 pickle로 저장**되고(`closeEvent` `labelImg.py:1448`), 시작할 때 그대로 읽힌다(`labelImg.py:99`).
- `get_format_meta(...)`(`labelImg.py:287-302`)는 **액션 생성 중에** 호출되어 곧바로 `[0]`/`[1]`로 인덱싱된다(`labelImg.py:304-307`).
- 원래 이 함수는 VOC/YOLO/CreateML **세 분기뿐이고 else가 없었다** → 알 수 없는 포맷이면 **`None`을 반환**했다. COCO 분기 없이 COCO 포맷이 한 번이라도 저장되면, 다음 실행에서 `None[0]`으로 `__init__` 도중 죽는다 — 설정 파일을 지우기 전까지 앱이 아예 안 뜬다.

해결(구현됨): COCO 분기(`labelImg.py:297-298`) + **방어적 기본값**(`labelImg.py:299-302`, else 없이 VOC로 떨어짐). 여기에 더해 `__init__`이 unpickle된 값의 **타입까지 검사**한다(`labelImg.py:100-103`) — 다른 빌드에서 온 enum이나 손상된 값도 포맷 디스패처에 닿지 못한다.

(참고로 `save_labels`는 미지원 포맷에 대해 `LabelFileError`를 던지고(`labelImg.py:1070-1072`) 그걸 잡아 다이얼로그로 보여준다 — **저장 쪽은 안전하게 실패한다.** 위험했던 건 **시작 경로**였다.)

### 10. `_dismissed`의 스코프 — 제안 라운드 경계

> **⬜ 열림 — 신규·비차단.** Phase 1 교차엔진 리뷰 이후 알려진 잔여 결함.

교차엔진 리뷰 이후 알려진 **잔여 결함**이다. 사용자가 손으로 지운 제안은 인덱스로 기억돼(`_dismissed` `libs/assist/controller.py:145`) 임계값을 다시 움직여도 되살아나지 않는다. 그런데 이 집합은 **예측 라운드 단위가 아니라 트리거 시점에** 비워진다(`auto_label_image` → `clear_suggestions` → `_forget` `libs/assist/controller.py:740-743`). 결과가 도착하는 `on_prediction_ready`(`libs/assist/controller.py:558-577`)는 `_detections`/`_shapes`만 갈아끼우고 `_dismissed`는 건드리지 않는다.

따라서 **Auto-label을 겹쳐 실행하면**(A 결과가 오는 중에 다시 트리거 → A 결과 도착 → 사용자가 그중 하나를 삭제 → B 결과 도착) A 라운드의 인덱스로 기록된 dismissal이 B 라운드의 **무관한 검출 하나를 가린다.**

영향은 작다: 크래시 없음, 데이터 손상 없음(가려진 것은 저장되지 않을 제안일 뿐), 다음 Auto-label에서 **자가 치유**된다. 제대로 된 수정은 dismissal을 인덱스가 아니라 **예측 라운드 id로 스코프**하는 것이다 — Phase 3(수락/거부 다듬기)에서 처리한다.

### 11. 레거시 `'stub'`이 pkl에 눌러앉는 문제 — DEFAULT_BACKEND=None 수정을 우회

> **✅ 닫힘** — Codex 교차엔진 리뷰가 발견, 같은 리뷰 사이클에서 수정.

`DEFAULT_BACKEND`를 `'stub'`에서 `None`으로 바꾼 수정(`libs/inference/registry.py:39`, 위 §핵심 인터페이스)은 **신규 설치**만 보호했다. 이 브랜치의 더 이른 커밋(`DEFAULT_BACKEND`가 아직 `'stub'`이던 시절)으로 앱을 한 번이라도 실행하고 닫은 사용자는, `closeEvent`가 그때 조건 없이 `settings[SETTING_MODEL_BACKEND] = self.assist.backend_name`을 기록했으므로(당시 `backend_name`은 `'stub'`) 이제 pkl에 **명시적** `'stub'`을 갖고 있다. `AssistController.__init__`이 `settings.get(SETTING_MODEL_BACKEND, DEFAULT_BACKEND)`로 그 값을 그대로 읽으면(`libs/assist/controller.py:107-118`), `DEFAULT_BACKEND=None` 수정을 완전히 우회하고 `StubBackend`(이미지 크기에서 유도한 가짜 검출)를 다시 세운다 — 사용자는 실제 모델이 돈 적이 없는데도 `Ctrl+I` 결과를 신뢰할 수 있다.

**해결(구현됨), 두 갈래**:

1. **읽기 시점 무해화** (`libs/assist/controller.py:82-97`, `:107-118`) — `SETTING_MODEL_BACKEND`가 정확히 `'stub'`이면 미설정으로 취급하고 로그를 남긴다(`_LEGACY_IMPLICIT_DEFAULT_BACKEND`). 설정 피커 UI가 아직 없고 `'stub'`을 손으로 쓰라는 문서도 없으므로, pkl에 있는 `'stub'`은 **예전 암묵적 기본값이 새어나온 것 말고는 있을 수 없다** — 인-프로세스에서 명시적으로 고른 `'stub'`(테스트 등)은 이 경로를 거치지 않으므로 계속 그대로 동작한다.
2. **쓰기 시점 절제** (`labelImg.py:1449-1472`) — `closeEvent`는 이제 `self.assist.backend_name`이 참일 때만(즉 실제로 설정된 값일 때만) 쓴다. 아무것도 설정되지 않았으면 그 키를 아예 쓰지 않고, pkl에 남아 있던 레거시 값도 지운다.

두 갈래 다 있어야 하는 이유: (1)만 있으면 pkl의 `'stub'`은 매 실행 무해화될 뿐 계속 남고, (2)만 있으면 이미 오염된 pkl은 못 고친다. 함께 있으면 **반복되는 save/load 사이클에서도** 서로를 되돌리지 않는다 — (1)이 항상 읽기를 방어하므로 (2)가 아직 pkl을 청소하지 못한 한 번의 세션에서도 안전하고, (2)가 결국 pkl 자체를 청소하므로 다음 세션부터는 (1)이 나설 일도 사라진다. → [settings.md](../reference/settings.md), [modules.md](../reference/modules.md).

## 설계 검증

### 설계 단계 (구현 전)

이 아키텍처는 **두 번째 엔진(Codex/GPT-5.5)으로 교차검증**했다. 그쪽은 독립적으로 다음에 수렴했다: **컨트롤러 추출**, **`ModelBackend` ABC**, **데이터클래스 심**, **provisional 도형**, **워커 스레드 + 현재 이미지 가드**, 그리고 위 상위 리스크 목록.

추가로 **다섯 가지 구체적인 주장을 깨뜨렸고**, 그 결과는 위 본문에 모두 반영돼 있다:

| 깨진 주장 | 반영 위치 |
|---|---|
| `Shape.copy()`가 새 필드를 **말없이 누락**한다 (필드 화이트리스트) | [§바뀐 기존 파일](#바뀐-기존-파일-phase-1) — `libs/shape.py:218-237`이 세 필드를 복사 |
| **COCO는 이미지별 저장 경로에 맞지 않는다** | [§리스크 7](#7-coco-저장-경로가-다르다--데이터셋-레인) |
| `CreateMLReader` **앞에서 내용 스니핑이 필요하다** (`TypeError`는 안 잡힌다) | [§리스크 8](#8-coco--createml의-json-충돌) |
| **`get_format_meta`가 다음 실행 때 크래시**시킨다 (pickle된 enum) | [§리스크 9](#9-pickle된-enum--기본값-없는-get_format_meta--다음-실행-때-죽는다) |
| "`shape_type`만 있으면 폴리곤이 열린다"는 **과장** | [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크) |

### 구현 단계 (Phase 1 교차엔진 리뷰)

Phase 1 산출물도 병합 전에 같은 방식으로 교차엔진 리뷰를 받았다. 리뷰가 **찾아내서 닫은** 것 — 넷 다 설계가 예측하지 못한, 구현에서만 나올 수 있는 결함이다:

| 결함 | 왜 위험했나 | 지금 |
|---|---|---|
| **제안 추적이 프로세스를 죽인다** | 컨트롤러가 `_shapes` dict를 "화면에 뭐가 있나"의 진실로 믿었다. 사용자가 제안을 평범한 Delete로 지우면 죽은 참조가 남고, 다음 임계값 변경이 그것을 두 번째로 지우려다 `ValueError` — **슬라이더를 끄는 도중에** 터진다 | `provisional_shapes()`가 **캔버스를 직접 읽는다**(`libs/assist/controller.py:613-633`), `remove_label`은 모든 삭제를 컨트롤러에 되돌려 보고한다(`labelImg.py:960-961`) |
| **COCO basename 충돌이 박스를 조용히 오염시킨다** | 데이터셋 항목을 파일명(basename)으로만 키잉했다. labelImg는 폴더를 **재귀 스캔**하므로 `train/0001.jpg`와 `val/0001.jpg`가 서로의 항목을 덮어쓰고 주석을 합집합으로 읽었다 — 예외 없이 **박스가 섞인다** | 키가 **데이터셋 상대경로**다(`dataset_relative_name` `libs/coco_io.py:34-61`). 남의 도구가 쓴 bare basename 항목은 **후보가 유일할 때만** 채택·마이그레이션한다(`libs/coco_io.py:174-214`, `:306-313`) |
| **Import COCO가 라벨 리스트를 어긋나게 한다** | `load_labels`가 캔버스는 통째로 갈아치우면서 라벨 리스트에는 **덧붙였다.** `load_file`은 먼저 리셋하지만 Import COCO·이전 박스 복사는 아니라서, 캔버스에 없는 도형의 리스트 항목이 남고 그걸 선택/삭제하면 `ValueError`·저장 누락 | `load_labels`가 매핑 3개를 **먼저 비운다**(`labelImg.py:963-977`) |
| **Python 지원 하한 회귀** | 코어가 dataclass(3.7+)를 쓰는데 패키지 메타데이터는 더 낮은 파이썬을 계속 허용했다 — 설치는 되고 **import에서 죽는** 조합 | `setup.py`의 `REQUIRES_PYTHON`이 `>=3.7`, classifier도 함께 정리 |

리뷰 이후에도 남은 것이 하나 있다 — [§리스크 10](#10-_dismissed의-스코프--제안-라운드-경계). 비차단이라 Phase 3으로 넘겼다.

## 참고

- 현재 구조: [architecture.md](architecture.md)
- 포맷 설계와 Reader/Writer 대칭: [annotation-formats.md](annotation-formats.md) · [../reference/formats.md](../reference/formats.md)
- 캔버스의 사각형 가정: [canvas-interaction-model.md](canvas-interaction-model.md)
