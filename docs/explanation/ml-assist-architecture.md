# ML 어시스트 아키텍처 (설계)

> ⚠️ **설계 문서 — 미구현.** 이 페이지가 서술하는 `libs/inference/`·`libs/assist/`·`libs/coco_io.py`·`InferenceService`·`AssistController`는 **현재 저장소에 존재하지 않는다.** 구현 단계는 [§단계별 순서](#단계별-순서)를 참조. 이 문서는 "무엇을 왜 그렇게 지을 것인가"(설계 근거·트레이드오프)를 남기는 explanation이며, 코드 인용 중 **`file.py:line` 형태로 줄 번호가 붙은 것은 모두 현재 존재하는 코드**이고, **신규 모듈은 이름으로만 지칭**한다(줄 번호 없음).

현재 저장소는 [architecture.md](architecture.md)가 서술한 대로 **MainWindow(God-object) ↔ Canvas ↔ Shape ↔ Reader/Writer** 구조의 수동 박스 도구다. 이 설계의 목표는 그 포크를 **AI 보조 주석 도구**로 진화시키되, **`MainWindow`의 God-object 문제를 더 악화시키지 않는 것**이다.

## 왜 새 클래스인가 — 분류 기능이 남긴 교훈

포크가 이미 붙인 이미지 분류 기능은 **`MainWindow`에 직접 용접된 약 270줄**이다(`classify_current_image` `labelImg.py:1585` ~ `rebuild_classify_actions` 끝 `labelImg.py:1851`). 동작은 하지만, 이것이 **반복하면 안 되는 성장 패턴**이다. `MainWindow` 클래스는 이미 `labelImg.py:73`부터 파일 끝(2,113줄) 직전까지 약 2,000줄이다. 추론 서비스·어시스트 컨트롤러·능동학습·SAM을 같은 방식으로 얹으면 600~1,000줄이 더 붙는다.

동시에, 분류 기능은 **따라 해야 할 좋은 패턴**도 남겼다: 액션을 **동적으로 만들어 등록**하는 방식이다.

- `create_classify_actions()`(`labelImg.py:1765`)가 설정에서 (단축키, 이름) 쌍을 읽어 `QAction` 리스트를 만든다.
- 만들어진 액션은 File 메뉴 튜플에 `+ tuple(self.classify_actions)`로 합쳐지고(`labelImg.py:464-467`), 이미지 의존 액션 집합 `onLoadActive`에도 합쳐진다(`labelImg.py:435-436`).
- `toggle_actions()`(`labelImg.py:665-672`)가 `onLoadActive`를 일괄 enable/disable 한다 — 이미지가 열려야 켜지는 액션은 **여기에 등록만 하면 된다.**

즉 **`MainWindow`는 "만들어진 것을 소유하고 배선하는" 껍데기 역할을 이미 하고 있다.** ML 어시스트는 이 껍데기 역할만 쓰고, 로직은 밖으로 뺀다.

## 핵심 결정 — 형제 패키지 + 얇은 컨트롤러 2개

**ML 어시스트 + 포맷 확장 스파인**을 기존 `libs/` 옆의 **새 형제 패키지**로 만든다: `libs/inference/`, `libs/assist/`, `libs/coco_io.py`. 그리고 `MainWindow`가 **소유하고 배선만** 하는 얇은 컨트롤러 객체 **두 개**를 둔다.

| 객체 | 책임 | MainWindow와의 관계 |
|---|---|---|
| `InferenceService` (QObject) | 모델 백엔드를 들고 스레드풀에서 추론을 돌리고 결과를 시그널로 낸다 | 생성·소유·시그널 연결만 |
| `AssistController` (QObject) | AI 액션, provisional 도형 수명주기, 임계값, 능동학습 재정렬 | 생성·소유, 액션을 메뉴/`onLoadActive`에 등록만 |

`MainWindow`는 **Qt 껍데기로 남는다.** 새 기능의 코드는 `MainWindow` 메서드 안이 아니라 컨트롤러 안에 있다.

부수 효과가 하나 더 있는데, 이게 사실상 결정적이다: **컨트롤러로 뽑아내면 각 관심사가 단위 테스트 가능해진다.** `StubBackend`(아래) 하나만 물리면 **Qt 이벤트 루프도, 실제 모델도 없이** 어시스트 로직을 검증할 수 있다. `MainWindow` 안에 박아 넣은 로직은 그럴 수 없다.

## AI 심(seam) — Detection/Mask 경계

이 설계에서 가장 중요한 한 줄:

> **AI 경계는 `Detection`/`Mask` 데이터클래스로만 말한다. `Shape`도 Qt도 절대 노출하지 않는다.**

`libs/inference/*`는 PyQt5를 import하지 않는다. 모델·백엔드·스레드는 순수 데이터(dataclass, numpy 배열)만 주고받는다. **UI 경계에서만** 얇은 어댑터(`libs/assist/suggestion.py`)가 `Detection → Shape`로 변환한다.

이 심이 있어야 나중에 SAM·폴리곤이 **Phase 1 코드를 다시 쓰지 않고** 접붙는다. 심이 없으면 백엔드가 `Shape`를 알게 되고, `Shape`가 사각형에 묶여 있으므로(→ [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크)) 백엔드까지 사각형에 묶인다.

## 컴포넌트 구성

```mermaid
graph TD
    MW["MainWindow (labelImg.py)<br/>Qt 껍데기 — 소유 + 배선만"]

    MW -->|소유·배선| AC["AssistController (QObject)<br/>libs/assist/controller.py"]
    MW -->|소유·배선| IS["InferenceService (QObject)<br/>libs/inference/service.py"]

    AC -->|Detection→Shape 어댑터| SUG["libs/assist/suggestion.py"]
    AC -->|재사용| CV["Canvas.load_shapes<br/>libs/canvas.py:713"]
    AC -->|재사용| SL["save_labels — 단일 저장 초크포인트<br/>labelImg.py:916"]
    AC -->|predict/segment 요청| IS
    IS -.->|결과 시그널 (queued)| AC

    IS -->|보유| BE["ModelBackend (ABC)<br/>libs/inference/backend.py"]
    IS -->|실행| TP["QThreadPool(maxThreadCount=1)"]
    REG["registry.build_backend(config)<br/>설정 기반, 의존성 없으면 None"] -->|생성| BE

    BE --> STUB["StubBackend<br/>결정론적·의존성 0"]
    BE --> YOLO["YoloOnnxBackend.predict()<br/>letterbox + 역-letterbox + NMS"]
    BE --> SAM["MobileSamBackend.segment()/embed()"]

    LF["LabelFile (libs/labelFile.py:28)"] --> RW["PascalVoc / YOLO / CreateML<br/>Reader·Writer (기존)"]
    LF --> COCO["COCOReader / COCOWriter<br/>libs/coco_io.py (신규)"]
```

읽는 법:

- **`MainWindow` → 컨트롤러**: 생성·소유·시그널 연결. 새 AI 분기를 `MainWindow` 메서드 안에 넣지 않는다.
- **`AssistController` → 기존 코드**: `Canvas.load_shapes`(`libs/canvas.py:713`)와 **단일 저장 초크포인트** `save_labels`(`labelImg.py:916`)를 **그대로 재사용**한다. 새 캔버스도, 새 저장 경로도 만들지 않는다.
- **`InferenceService` → 백엔드**: 설정 기반 레지스트리가 만들어 준 `ModelBackend` 하나를 들고 `QThreadPool`에서 돌린다.
- **`LabelFile`**: `COCOReader`/`COCOWriter`가 기존 Reader/Writer 옆에 형제로 붙는다.

## 신규 모듈

| 모듈 (신규) | 책임 (한 줄) |
|---|---|
| `libs/coco_io.py` | `COCOReader`/`COCOWriter`. CreateML의 **데이터셋 병합 패턴**(기존 파일을 읽어 현재 이미지 항목만 갈아끼움)을 그대로 따른다 |
| `libs/inference/types.py` | `Detection`·`Mask`·`Prediction`·`SegPrompt` 데이터클래스 — AI 심의 어휘 |
| `libs/inference/backend.py` | `ModelBackend` ABC — `predict`/`segment`/`embed` + capability 플래그 |
| `libs/inference/stub.py` | `StubBackend` — 결정론적·의존성 0. **테스트를 구동하는 주체** |
| `libs/inference/yolo_onnx.py` | ONNX YOLO 백엔드: letterbox → 추론 → **역-letterbox(원본 픽셀 복원)** → NMS. `onnxruntime`은 **메서드 안에서 지연 import** |
| `libs/inference/registry.py` | `build_backend(config)` → 백엔드, 의존성이 없으면 **`None`** |
| `libs/inference/service.py` | `InferenceService`: `QThreadPool`, 비동기 predict/segment, 결과 시그널, 이미지별 임베딩 캐시, **stale 결과 폐기** |
| `libs/assist/controller.py` | `AssistController`: AI 액션, provisional 수명주기, 신뢰도 임계값, 능동학습 재정렬 |
| `libs/assist/suggestion.py` | `Detection → Shape` 어댑터 (UI 경계의 **유일한** 변환 지점) |
| `libs/inference/mobile_sam.py` · `mask_utils.py` | MobileSAM 백엔드와 마스크→폴리곤 유틸 (Phase 6) |

## 바뀌는 기존 파일

| 파일 | 무엇이 바뀌나 |
|---|---|
| `libs/shape.py` | `provisional`·`confidence`·`shape_type` 필드 추가. **`copy()`(`libs/shape.py:189-200`)가 이 필드들을 반드시 함께 복사해야 한다** — 현재 `copy()`는 필드를 하나씩 나열해 옮기는 화이트리스트라, 새 필드는 **말없이 누락된다.** `paint()`(`libs/shape.py:87`)는 provisional일 때 점선으로 그린다 |
| `libs/labelFile.py` | `LabelFileFormat.COCO`(현재 `libs/labelFile.py:18-21`에 3개뿐) + `save_coco_format` |
| `libs/constants.py` | `FORMAT_COCO`(현재 `FORMAT_*`는 `libs/constants.py:15-17`의 3개) + `SETTING_MODEL_*` 키 |
| `labelImg.py` | 컨트롤러 배선, AI 메뉴, **저장 초크포인트에 provisional 필터 1개**, `get_format_meta`(`labelImg.py:269`) COCO 분기 + **방어적 기본값** |
| `setup.py` | `extras_require`에 `'ai'` 추가 (현재 `REQUIRED_DEP = ['pyqt5', 'lxml']` `setup.py:13`, `install_requires` `setup.py:102`) |
| `.github/workflows/ci.yml` | 선택적 `[ai]` 잡 추가 (현재 코어 잡은 `pip install pyqt5 lxml` `ci.yml:29-30`) |

## 핵심 인터페이스

```python
# libs/inference/types.py  (계획)
@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple[float, float, float, float]   # (x1, y1, x2, y2) — 원본 이미지 픽셀
    score: float
    class_id: int

@dataclass(frozen=True)
class Mask:
    label: str
    polygon: list[tuple[float, float]]       # 원본 이미지 픽셀
    score: float

@dataclass(frozen=True)
class Prediction:
    image_path: str
    detections: list[Detection]
    uncertainty: float          # 능동학습 정렬 키
    masks: list[Mask] = ()

@dataclass(frozen=True)
class SegPrompt:
    points: list[tuple[float, float]]        # 원본 이미지 픽셀
    labels: list[int]                        # 1=foreground, 0=background
    box: tuple[float, float, float, float] | None = None
```

```python
# libs/inference/backend.py  (계획)
class ModelBackend(ABC):
    supports_detection: bool = False
    supports_segmentation: bool = False
    supports_embedding: bool = False

    @abstractmethod
    def predict(self, image: "np.ndarray", conf: float) -> list[Detection]: ...
    def segment(self, image: "np.ndarray", prompt: SegPrompt,
                embedding=None) -> list[Mask]: ...
    def embed(self, image: "np.ndarray"): ...
```

### 좌표 규약 (이 설계의 두 번째 불변식)

> **`Detection.box`·`Mask.polygon`은 항상 원본 이미지 픽셀 좌표다** — 기존 Reader들이 내놓는 것과 **완전히 동일한 좌표계**다.

기존 Reader는 `(label, points, None, None, difficult)` 5-튜플을 원본 픽셀로 돌려주고(예: `libs/create_ml_io.py:125-133`), `MainWindow.load_labels`(`labelImg.py:875`)가 그걸 `Shape`로 복원한다. 예측 결과가 **같은 좌표계**를 쓰면 UI는 예측을 **Reader 출력과 구별할 필요가 없다.**

letterbox(패딩·스케일) 역변환은 **백엔드 안에서** 끝난다 — `libs/inference/yolo_onnx.py`가 모델 입력 텐서 좌표를 원본 픽셀로 되돌린 뒤에야 `Detection`을 만든다. **UI는 letterbox를 모른다.** 이 역변환을 UI로 새어 나가게 두면, 백엔드가 바뀔 때마다 UI가 같이 바뀐다.

## provisional Shape 수명주기

예측은 `provisional=True` + `confidence`가 붙은 `Shape`가 되어 캔버스에 올라간다. 세 가지가 걸려 있다.

**1. 캔버스에만 올리면 안 된다.** `AssistController`는 각 provisional 도형을 **반드시 `add_label`(`labelImg.py:852`)로도 등록**해야 한다. 선택 처리 경로가 `shapes_to_items`(`labelImg.py:859`에서 채워짐)를 조회하고, `remove_label`(`labelImg.py:865-872`)은 `self.shapes_to_items[shape]`를 **가드 없이 인덱싱**한다(`labelImg.py:869`) — 캔버스에만 있는 도형을 선택/삭제하면 `KeyError`다.

**2. 디스크에는 절대 안 나간다 — 필터는 단 하나.** 모든 저장 경로(`Ctrl+S` → `save_file` `labelImg.py:1519`, 다음/이전 이미지 자동저장 → `open_next_image`의 `auto_saving` 분기 `labelImg.py:1476-1479`와 `open_prev_image`의 같은 분기 `labelImg.py:1451-1454`, verify → `verify_image` `labelImg.py:1431`)는 전부 `_save_file`(`labelImg.py:1556`)을 거쳐 **`save_labels`(`labelImg.py:916`) 하나로 수렴**한다. 직렬화 대상은 그 안의 `shapes = [format_shape(shape) for shape in self.canvas.shapes]`(`labelImg.py:930`) 한 줄이다. **여기에 provisional 필터 하나만 넣으면 모든 저장 경로가 동시에 막힌다.** 저장 경로마다 필터를 흩뿌리면 하나를 빠뜨렸을 때 확정하지 않은 AI 추측이 디스크에 새어 나간다.

**3. 수락/거부.**

| 동작 | 결과 |
|---|---|
| **수락** (Enter) | `provisional=False`로 클리어 + `set_dirty()`(`labelImg.py:656`) — 이제 평범한 도형이므로 다음 저장에 포함된다 |
| **거부** (Delete) | 도형 제거 (`Canvas.delete_selected` + `remove_label`) |

**디스크 포맷 계약은 바뀌지 않는다.** provisional·confidence는 **메모리 전용 상태**다. Reader/Writer의 5-튜플 계약(`libs/labelFile.py`, 각 `*_io.py`)은 그대로다 — 즉 AI 어시스트를 켜도 산출물은 기존 VOC/YOLO/CreateML 파일과 **바이트 단위로 같은 종류**다.

## 스레딩

`InferenceService`는 **`QThreadPool(maxThreadCount=1)`** 을 소유한다. 워커를 1개로 고정하는 이유는 두 가지다: ONNX Runtime의 `Session.Run`은 **한 세션에 대해 동시 호출이 안정적으로 안전하지 않고**, CPU 추론은 워커를 늘려도 코어를 나눠 쓸 뿐 이득이 없다.

규칙 세 가지:

1. **워커는 순수 데이터만 받는다.** UI 스레드에서 `QImage`를 numpy 배열로 변환해 **소유권 있는 스냅샷**을 만들어 넘긴다. 워커는 어떤 Qt 객체도 만지지 않는다.
2. **결과는 queued 시그널로 돌아온다.** 슬롯이 UI 스레드에서 실행되므로 거기서 `Shape`를 만들어도 안전하다. (Qt 위젯·`QPixmap`은 워커 스레드에서 손대면 안 된다.)
3. **stale 결과 폐기.** 모든 요청은 자기 `image_path`를 들고 다니고, 돌아온 결과의 경로가 **현재 열린 파일과 다르면 버린다.** 추론 중에 사용자가 다음 이미지로 넘어가도(`open_next_image`, 자동저장까지 포함) **엉뚱한 이미지에 박스가 꽂히는 일이 없다.** 이 가드가 없으면 사용자가 빠르게 넘길 때 조용히 오염된다.

**이미지별 임베딩 캐시는 Phase 1에 만들어 둔다** — 채우는 건 SAM뿐이지만, 나중에 캐시를 끼워 넣으려면 서비스의 요청/결과 배관을 다시 건드려야 한다.

## 패키징과 테스트

- **선택적 `[ai]` extra**(`onnxruntime`·`numpy`·`opencv-python-headless`). 기본 설치는 지금처럼 `pyqt5`+`lxml`로 유지한다.
- **지연 import.** AI 관련 import는 모듈 최상단이 아니라 메서드 안에 둔다. `registry.build_backend()`는 의존성이 없으면 예외가 아니라 **`None`을 돌려주고** "`pip install labelImg[ai]`" 힌트를 남긴다 — AI 없이 실행해도 앱은 정상 동작한다.
- **`StubBackend`가 테스트를 구동한다.** 모델 파일도, `onnxruntime`도 없이 어시스트 로직(수락/거부, provisional 필터, stale 폐기)을 결정론적으로 검증한다.
- **COCO 라운드트립 테스트**는 `tests/test_io.py`의 기존 패턴을 그대로 따른다.
- **CI**: 코어 매트릭스는 `onnxruntime` 없이 계속 green(`.github/workflows/ci.yml:29-30`), 여기에 선택적 `[ai]` 잡 하나를 추가한다.

## 확장성 증명 — 심이 진짜인지

설계의 심(seam)이 진짜인지는 "**나중 기능이 이 심 위에서 몇 줄로 되는가**"로만 증명된다.

### 능동학습 (uncertainty 우선 정렬)

1. 폴더 전체를 배치 추론한다.
2. `uncertainty = 1 - mean(top-k scores)`(또는 margin/entropy)를 `Prediction.uncertainty`에 넣는다.
3. **기존 `m_img_list`를 재정렬**하고, 파일 리스트를 `import_dir_images`가 이미 쓰는 그 루프(`labelImg.py:1427-1429`)로 다시 채운다.

탐색 코드는 `m_img_list`를 **인덱스로만** 읽으므로(`self.m_img_list[self.cur_img_idx]` — `open_prev_image` `labelImg.py:1470`, `open_next_image` `labelImg.py:1500`) **변경이 0줄**이다. 포크의 기존 `g`/`b` 분류 트리아지(`labelImg.py:1585`)도 **그대로** 재사용되며, 이제 불확실성 순으로 정렬된 스트림 위에서 돈다. 새로 생기는 표면은 **채점 함수 1개 + 재정렬 메서드 1개**뿐이다.

### SAM (프롬프트 기반 분할)

- `MobileSamBackend`가 **같은 `ModelBackend` ABC** 위에서 `segment()`/`embed()`를 구현한다.
- 캔버스 클릭이 `SegPrompt`가 된다.
- **캐시된 임베딩** 덕분에 클릭마다 가벼운 디코더만 돈다(무거운 인코더는 이미지당 1회).
- 결과 마스크는 폴리곤 `Shape`가 되어 **동일한 수락/거부 수명주기**를 탄다.

**새 컨트롤러 없음, 새 스레딩 없음, 새 저장 경로 없음.** 이게 심이 진짜라는 뜻이다. (단, 폴리곤 자체는 공짜가 아니다 — [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크).)

## 단계별 순서

| 단계 | 내용 | 규모 |
|---|---|---|
| **1a** | COCO I/O (`libs/coco_io.py`) | M |
| **1b** | 추론 코어: `types`/`backend`/`stub`/`registry` — **순수 파이썬, Qt 없음** | S |
| **1c** | 얇은 수직 슬라이스: `InferenceService` + `AssistController` + `Shape` 필드 + provisional 점선 렌더 + AI 액션 | L |
| **2** | `YoloOnnxBackend` + `[ai]` extra | M |
| **3** | 수락/거부/편집 다듬기 + 신뢰도 슬라이더 + Auto-label Folder | M |
| **4** | 능동학습 (uncertainty 정렬) | S/M |
| **5** | 폴리곤/키포인트 (**사각형 결합 해체**) | L |
| **6** | MobileSAM | L |

1b가 **Qt 없이** 끝나는 게 중요하다 — 심이 실제로 Qt와 분리돼 있는지를 이 단계에서 강제로 증명한다.

## 리스크와 트레이드오프

### 1. God-object 성장

컨트롤러 2개로 완화하지만, **진짜 리스크는 규율이다.** "여기 한 줄만"이라며 AI 분기를 `MainWindow` 메서드 안에 넣기 시작하면 분류 기능(`labelImg.py:1585-1851`)이 그랬듯 몇 백 줄이 다시 쌓인다. `MainWindow`가 만져도 되는 것: 컨트롤러 **생성·소유·배선**, 액션을 메뉴와 `onLoadActive`(`labelImg.py:435-436`)에 등록, 저장 초크포인트의 **필터 1줄**. 그 이상은 컨트롤러 안이다.

### 2. 좌표계 매핑

역-letterbox는 **백엔드 안**에서 일어나야 한다. 틀리면 증상이 조용하다 — 예외가 아니라 **모든 박스가 일정하게 밀리거나 스케일이 어긋난다.** 스텁이 아닌 실제 이미지로 라운드트립 테스트를 걸어야 잡힌다.

### 3. 스레딩 함정

단일 워커 풀, 순수 데이터 워커, queued 시그널, **현재 파일 경로 기준 stale 폐기** — 네 가지가 모두 있어야 한다. 특히 stale 폐기가 없으면 "빨리 넘기는 사용자"에게만 재현되는, 디버깅하기 최악인 오염 버그가 된다.

### 4. 모델 라이선스 (법적 리스크)

Ultralytics YOLOv5/v8 **가중치는 AGPL-3.0**이고, 이 앱의 MIT 라이선스와 **충돌한다.** 대응:

- 관대한 라이선스 모델을 싣거나(예: YOLOX / Apache-2.0),
- **가중치를 아예 싣지 않는다** — 사용자가 설정에서 자기 `.onnx` 경로를 가리키게 한다.

어느 쪽이든 **가중치는 기본 설치에 들어가지 않는다.**

### 5. 기본 설치 무게

`[ai]` extra + 지연 import + **StubBackend만 쓰는 코어 테스트** 세 가지가 함께 있어야 `pip install labelImg`가 가벼운 채로 남는다. 하나라도 빠지면 `onnxruntime`(수백 MB)이 기본 의존성으로 새어 들어온다.

### 6. 사각형 결합의 벽 — 가장 비싼 미래 리워크

> ⚠️ **"`shape_type` 필드만 추가하면 마스크→폴리곤이 된다"는 말은 거짓이다.** 폴리곤·키포인트·마스크는 필드 하나로 해금되지 않는다.

사각형 가정은 코드 전반에 **퍼져 있다**:

| 위치 | 사각형 가정 |
|---|---|
| `Shape.add_point` / `reach_max_points` (`libs/shape.py:67-74`) | **4점 상한** — 5번째 점은 조용히 버려진다 |
| `Canvas.handle_drawing` (`libs/canvas.py:322-333`) | 클릭 2번으로 **4점 사각형을 합성** |
| `Canvas.bounded_move_vertex` (`libs/canvas.py:400-434`) | `(index + 2) % 4` — **모서리 산술** |
| `Canvas.move_one_pixel` (`libs/canvas.py:647-672`) | `points[0]`~`points[3]`을 **직접 4개 다** 이동 |
| `Canvas.move_out_of_bound` (`libs/canvas.py:676-678`) | `zip(points, [step] * 4)` — **4점 하드코딩** |
| 상태바 크기 표시 (`libs/canvas.py:186-187`, `244-245`) | `h_shape[1]` / `h_shape[3]`을 **인덱스로** 집어 폭·높이 계산 |
| 모든 박스 writer | 점들을 bbox로 **환원** — `LabelFile.convert_points_to_bnd_box`(`libs/labelFile.py:152-174`), CreateML은 `points[0][0]`/`points[1][0]`/`points[2][1]`을 직접 인덱싱(`libs/create_ml_io.py:42-45`) |

따라서 **Phase 1~4는 사각형 전용으로 남는다.** 이 일반화는 **Phase 5/6의 몫**이며, 그 자체로 큰 작업이다. 이걸 "필드 하나"로 착각하고 일정을 잡는 것이 이 로드맵에서 가장 비싼 실수다.

### 7. COCO 저장 경로가 다르다 — 데이터셋 레인

VOC/YOLO/CreateML은 **이미지별 사이드카 경로를 유도해서 현재 이미지 하나만 직렬화**한다(`save_file` `labelImg.py:1519-1532`가 이미지 stem으로 경로를 만들고, `save_labels` `labelImg.py:933-947`이 확장자를 붙인다). **이 경로로는 데이터셋 하나짜리 `instances.json`을 만들 수 없다.**

따라서 **COCO는 데이터셋 수준의 Import/Export 레인**으로 둔다:

- **명시적 데이터셋 대상**을 갖는다 (기본값: 저장 폴더의 `annotations.json`).
- 저장 분기는 현재 이미지 항목을 **그 고정된 파일에 병합**한다 (CreateML의 병합 패턴과 같은 모양).
- **COCO는 이미지별 자동저장에 연결하지 않는다.**

### 8. COCO / CreateML의 `.json` 충돌

둘 다 `.json`을 쓰는데, 로드 디스패치는 **확장자로만** 분기한다(`show_bounding_box_from_annotation_file` `labelImg.py:1234`, `labelImg.py:1246` → `load_create_ml_json_by_filename` `labelImg.py:2029`).

해결: **모든 `.json` 디스패치 지점에서 `CreateMLReader`보다 먼저 내용을 스니핑**한다.

- 최상위가 `images`/`annotations`/`categories`를 가진 **dict → COCO**
- 최상위가 **list → CreateML**

이게 필수인 이유: `CreateMLReader`는 **`ValueError`만 잡는다**(`libs/create_ml_io.py:102-105`). COCO dict를 물리면 `parse_json`의 `for image in output_list`(`libs/create_ml_io.py:119-120`)가 **dict의 키(문자열)를 순회**하다 `image["image"]`에서 `TypeError`를 낸다 — `ValueError`가 아니므로 **잡히지 않고 앱까지 올라간다.** 즉 `CreateMLReader`는 COCO 파일을 **우아하게 거부하지 못한다.**

### 9. pickle된 enum + 기본값 없는 `get_format_meta` — 다음 실행 때 죽는다

- `label_file_format`은 **설정에 pickle로 저장**되고(`closeEvent` `labelImg.py:1316`), 시작할 때 그대로 읽힌다(`labelImg.py:96`).
- `get_format_meta(...)`(`labelImg.py:269-278`)는 **액션 생성 중에** 호출되어 곧바로 `[0]`/`[1]`로 인덱싱된다(`labelImg.py:280-283`).
- 이 함수는 VOC/YOLO/CreateML **세 분기뿐이고 else가 없다** → 알 수 없는 포맷이면 **`None`을 반환**한다.

결론: **COCO 분기 없이 COCO 포맷이 한 번이라도 저장되면, 다음 실행에서 `None[0]`으로 `__init__` 도중 죽는다.** 설정 파일을 지우기 전까지 앱이 아예 안 뜬다. 그래서 **`get_format_meta`의 COCO 분기 + 방어적 기본값(else)은 선택이 아니라 필수다.**

(참고로 `save_labels`는 이미 미지원 포맷에 대해 `LabelFileError`를 던지고(`labelImg.py:948-950`) 그걸 잡아 다이얼로그로 보여준다 — **저장 쪽은 안전하게 실패한다.** 위험한 건 **시작 경로**다.)

## 설계 검증

이 아키텍처는 **두 번째 엔진(Codex/GPT-5.5)으로 교차검증**했다. 그쪽은 독립적으로 다음에 수렴했다: **컨트롤러 추출**, **`ModelBackend` ABC**, **데이터클래스 심**, **provisional 도형**, **워커 스레드 + 현재 이미지 가드**, 그리고 위 상위 리스크 목록.

추가로 **다섯 가지 구체적인 주장을 깨뜨렸고**, 그 결과는 위 본문에 모두 반영돼 있다:

| 깨진 주장 | 반영 위치 |
|---|---|
| `Shape.copy()`가 새 필드를 **말없이 누락**한다 (필드 화이트리스트) | [§바뀌는 기존 파일](#바뀌는-기존-파일) |
| **COCO는 이미지별 저장 경로에 맞지 않는다** | [§리스크 7](#7-coco-저장-경로가-다르다--데이터셋-레인) |
| `CreateMLReader` **앞에서 내용 스니핑이 필요하다** (`TypeError`는 안 잡힌다) | [§리스크 8](#8-coco--createml의-json-충돌) |
| **`get_format_meta`가 다음 실행 때 크래시**시킨다 (pickle된 enum) | [§리스크 9](#9-pickle된-enum--기본값-없는-get_format_meta--다음-실행-때-죽는다) |
| "`shape_type`만 있으면 폴리곤이 열린다"는 **과장** | [§리스크 6](#6-사각형-결합의-벽--가장-비싼-미래-리워크) |

## 참고

- 현재 구조: [architecture.md](architecture.md)
- 포맷 설계와 Reader/Writer 대칭: [annotation-formats.md](annotation-formats.md) · [../reference/formats.md](../reference/formats.md)
- 캔버스의 사각형 가정: [canvas-interaction-model.md](canvas-interaction-model.md)
