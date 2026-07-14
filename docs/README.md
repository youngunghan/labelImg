# labelImg — 개발자 문서

**labelImg**(1.8.6)는 Python + Qt(PyQt5)로 만든 그래픽 이미지 바운딩박스 주석 도구다. 이미지 위에 사각형(bounding box)을 그려 객체에 라벨을 붙이고, 그 결과를 **PASCAL VOC XML · YOLO txt · CreateML JSON · COCO JSON** 네 포맷으로 저장한다. 원래 ImageNet 작업용으로 만들어졌고 객체 검출 학습 데이터 라벨링에 널리 쓰인다. (COCO는 이미지별 사이드카가 아니라 **데이터셋 레인**이다 — [explanation/ml-assist-architecture.md](explanation/ml-assist-architecture.md) §리스크 7.)

이 폴더는 labelImg를 **이해·확장·디버깅**하려는 개발자용 문서다. 사용자 빠른 시작과 핫키 요약은 프로젝트 루트 [../README.rst](../README.rst) 에도 있다. 에이전트(LLM)용 라우팅 지도는 [llms.txt](llms.txt).

> ⚠️ labelImg 본체는 더 이상 활발히 개발되지 않으며 Label Studio 커뮤니티로 흡수되었다(루트 README 상단 안내). 이 문서는 **현재 저장소(`D:\labelImg`)의 코드 ground-truth** 기준으로 작성됐다.

## 문서 목록 (Diátaxis)

처음이면 → **[tutorials/first-run.md](tutorials/first-run.md)**.

**튜토리얼 (학습)**

| 문서 | 설명 |
|---|---|
| [tutorials/first-run.md](tutorials/first-run.md) | 처음 실행 — 설치 → 이미지 폴더 열기 → 박스 그리기 → 저장 happy path |

**작업 (how-to)**

| 문서 | 설명 |
|---|---|
| [how-to/install-and-build.md](how-to/install-and-build.md) | 설치·리소스 컴파일(`pyrcc5`)·실행·패키징(setup.py / pyinstaller) |
| [how-to/annotate-pascal-voc.md](how-to/annotate-pascal-voc.md) | PASCAL VOC(XML) 라벨링 절차 |
| [how-to/annotate-yolo.md](how-to/annotate-yolo.md) | YOLO(txt + classes.txt) 라벨링 절차와 주의점 |
| [how-to/annotate-createml.md](how-to/annotate-createml.md) | CreateML(JSON) 라벨링 절차 |
| [how-to/verify-and-difficult.md](how-to/verify-and-difficult.md) | verify 플래그·difficult·single-class/use-default-label·auto-save·정사각형 그리기 |
| [how-to/more-features.md](how-to/more-features.md) | ⚠️이미지 분류(g/b 파일이동)·이전 박스 복사(Ctrl+V)·클래스 편집·고급 모드·밝기·박스 색 |
| [how-to/export-to-csv.md](how-to/export-to-csv.md) | `tools/label_to_csv.py`로 VOC/YOLO → AutoML CSV 변환 |
| [how-to/reset-and-troubleshoot.md](how-to/reset-and-troubleshoot.md) | 설정 리셋(`.labelImgSettings.pkl`)·자주 겪는 문제 |

**레퍼런스 (찾아보기)**

| 문서 | 설명 |
|---|---|
| [reference/modules.md](reference/modules.md) | 파일별 핵심 클래스/함수 인벤토리 |
| [reference/formats.md](reference/formats.md) | 세 출력 포맷의 정확한 구조·좌표 규약·라운드트립 |
| [reference/settings.md](reference/settings.md) | `Settings`와 `.labelImgSettings.pkl`에 저장되는 키 |
| [reference/shortcuts.md](reference/shortcuts.md) | 단축키(핫키) 전체 표 |

**설계·개념 (explanation)**

| 문서 | 설명 |
|---|---|
| [explanation/architecture.md](explanation/architecture.md) | 컴포넌트 구성, MainWindow ↔ Canvas ↔ Shape ↔ I/O, Qt 시그널/슬롯 |
| [explanation/annotation-formats.md](explanation/annotation-formats.md) | 3 포맷 설계, 포맷 전환, reader/writer 대칭, difficult/verified 의미 |
| [explanation/canvas-interaction-model.md](explanation/canvas-interaction-model.md) | CREATE/EDIT 모드, 마우스·정점·도형 이동, 정사각형 제약, 줌/패닝 |
| [explanation/ml-assist-architecture.md](explanation/ml-assist-architecture.md) | ML 어시스트 스파인의 설계 근거 + 구현 현황 — `InferenceService`·`AssistController`·`ModelBackend`, provisional 도형, 스레딩, COCO 레인, 리스크. **Phase 1(COCO I/O·추론 코어·어시스트 수직 슬라이스) 구현 완료(`a32acd3`), Phase 2~6(실제 ONNX 백엔드·능동학습·폴리곤·SAM) 미구현** |

## 코드 조감도 (한 줄)

- [`labelImg.py`](../labelImg.py) — `MainWindow`(앱 컨트롤러, God-object). UI·액션·파일 입출력·설정.
- [`libs/canvas.py`](../libs/canvas.py) — `Canvas`(드로잉/편집 QWidget).
- [`libs/shape.py`](../libs/shape.py) — `Shape`(박스 하나의 데이터+렌더링).
- [`libs/labelFile.py`](../libs/labelFile.py) — `LabelFile`(포맷별 writer 위임 파사드) + `LabelFileFormat`.
- [`libs/pascal_voc_io.py`](../libs/pascal_voc_io.py) · [`libs/yolo_io.py`](../libs/yolo_io.py) · [`libs/create_ml_io.py`](../libs/create_ml_io.py) · [`libs/coco_io.py`](../libs/coco_io.py) — 포맷별 Reader/Writer.
- [`libs/inference/`](../libs/inference/) — 모델 심(seam): `Detection`/`Mask` 데이터클래스, `ModelBackend` ABC, `StubBackend`, 레지스트리, `InferenceService`(단일 워커 스레드풀). 코어는 **Qt·numpy 없이** import된다.
- [`libs/assist/`](../libs/assist/) — `AssistController`(AI 액션·provisional 도형 수명주기·신뢰도 임계값)와 `Detection → Shape` 어댑터.
- [`libs/settings.py`](../libs/settings.py) · [`libs/stringBundle.py`](../libs/stringBundle.py) · [`libs/utils.py`](../libs/utils.py) — 설정 영속화·i18n·공용 헬퍼.
- [`tools/label_to_csv.py`](../tools/label_to_csv.py) — VOC/YOLO 라벨을 AutoML CSV로 변환하는 독립 스크립트. `tests/` — 단위 테스트(코어 매트릭스는 AI 의존성 없이 green).
