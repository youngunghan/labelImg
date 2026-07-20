# How-to: 설치 · 빌드 · 패키징

## 의존성

- **Python 3.7 이상**. `setup.py`의 `REQUIRES_PYTHON='>=3.7'`.
  - 3.7이 하한인 이유: `labelImg.py`가 모듈 최상단에서 `libs.inference`·`libs.assist`를 import하고,
    이들이 `from __future__ import annotations`(3.7+)와 `dataclasses`(3.7+)를 쓴다. 3.6 이하에서는
    AI 기능만 빠지는 게 아니라 `import labelImg` 자체가 SyntaxError/ImportError로 실패한다.
- `pyqt5`, `lxml`.
- 핀 고정(`requirements/requirements-linux-python3.txt`): `pyqt5==5.14.1`, `lxml==4.9.1`.
  - (루트 README 본문은 `pyqt5==5.15.2`를 권장한다 — 핀은 재현용, 본문은 일반 권장값.)
- **선택 extra `[ai]`**: `onnxruntime>=1.15`, `numpy`(`setup.py:26-28`, `EXTRA_DEP`). AI 자동 라벨링의
  실제 ONNX 모델 백엔드(`libs/inference/yolo_onnx.py`)에만 필요하고 기본 설치(`REQUIRED_DEP`)에는
  들어가지 않는다 — 설치 안 해도 `pyqt5`+`lxml`만으로 앱은 완전히 동작한다. 다만 기본 설치는
  **백엔드가 전혀 설정되지 않은 상태**라 AI 메뉴는 처음부터 비활성화돼 있다(설치/설정 안내 툴팁만
  뜬다) — extras를 설치하고 모델 백엔드/경로를 명시적으로 설정해야 비로소 켜진다. 자세한 사용법은
  [auto-label.md](auto-label.md).

## A. PyPI 설치(가장 간단)

```shell
pip3 install labelImg
labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]
```

> ⚠️ 이 PyPI 패키지는 업스트림 `labelImg`이며 **이 포크의 기능(AI 자동 라벨링, COCO
> 가져오기/내보내기 등)을 전혀 포함하지 않는다** — `[ai]` extra를 추가로 설치해도
> 업스트림 패키지에는 애초에 이 포크의 AI 코드가 없으므로 아무 의미가 없다. AI 자동
> 라벨링을 포함한 이 포크의 기능을 쓰려면 아래 **B. 소스에서 실행**을 따라야 한다.

## B. 소스에서 실행

소스로 실행하려면 **Qt 리소스를 먼저 컴파일**해야 한다. `resources.qrc`(아이콘 + 다국어 문자열 매니페스트)를 `pyrcc5`로 `libs/resources.py`에 임베드한다.

### Linux/Ubuntu (Python3 + Qt5)

```shell
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3                 # = pyrcc5 -o libs/resources.py resources.qrc
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]
```

### macOS

```shell
pip3 install pyqt5 lxml
make qt5py3
python3 labelImg.py
```

### Windows

```shell
pyrcc5 -o libs/resources.py resources.qrc     # PyQt5는 pyrcc5 (pyrcc4 아님)
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]
```

### AI 자동 라벨링 extra (선택)

실제 ONNX 모델로 추론하려면(`[ai]` extra: `onnxruntime`+`numpy`), 위 세 플랫폼 어디서든 이
저장소 루트에서 다음을 실행한다:

```shell
pip install -e ".[ai]"
```

이 포크는 PyPI에 `labelImg`라는 이름으로 배포되어 있지 **않다** — `pip3 install
labelImg[ai]`(A절)를 실행하면 이 저장소와 무관한 업스트림 패키지를 받게 되므로, extra는
반드시 위처럼 로컬 체크아웃에서 설치해야 한다. 자세한 사용법은 [auto-label.md](auto-label.md).

> `make qt5py3`이 하는 일은 한 줄이다(`Makefile:23-24`): `pyrcc5 -o libs/resources.py resources.qrc`. 이 `libs/resources.py`(약 654KB 자동생성)가 없으면 아이콘·문자열 로딩이 실패한다.

> 위치 인자 3개는 모두 선택적이다(`labelImg.py`의 `get_main_app`, 2417-2421행): ① `IMAGE_PATH` — 이미지 폴더 또는 단일 이미지 파일, ② `PRE-DEFINED CLASS FILE` — 클래스 목록 파일(기본 `data/predefined_classes.txt`), ③ `SAVE_DIR` — 어노테이션 저장 폴더. `IMAGE_PATH`가 폴더여도 `SAVE_DIR`는 유효하다 — 시작 시 `open_dir_dialog(silent=True)`가 명령줄 `save_dir`를 `default_save_dir`로 유지한다(`labelImg.py:1582-1583`). 단, 실행 후 메뉴의 Open Dir로 폴더를 열면 `default_save_dir`가 그 폴더로 재설정되므로, 저장 폴더를 따로 쓰려면 Open Dir **이후에** Change Save Dir를 실행해야 한다.

## C. 빌드 타겟(Makefile)

| 타겟 | 동작 |
|---|---|
| `make qt5py3` (= `qt5`) | `pyrcc5`로 리소스 컴파일 |
| `make testpy3` | `python3 -m unittest discover tests` |
| `make all` | `qt5` + `test` |
| `make clean` | `~/.labelImgSettings.pkl`, `*.pyc`, `dist`, `labelImg.egg-info`, `__pycache__`, `build` 삭제 |
| `make qt4py2` / `qt4py3` | (구) PyQt4용 `pyrcc4` — 레거시 |

## D. 단일 실행파일(pyinstaller)

```shell
pip install pyinstaller
pyinstaller --hidden-import=pyqt5 --hidden-import=lxml -F -n "labelImg" -c labelImg.py -p ./libs -p ./
```

`labelImg.spec`도 함께 제공된다. macOS는 `setup.py`의 `py2app` 경로로 `.app`을 만들 수 있다(`build-tools/build-for-macos.sh`).

> 참고(이 포크의 커스텀 동작): frozen exe는 `predefined_classes.txt`를 exe 옆 경로에 만들어 영속화한다(`labelImg.py`의 `get_persistent_classes_file`, 2177-2196행). 파일이 없으면 번들 기본값을 복사하고, 그것도 없으면 `person` 한 줄로 생성한다. Edit Default Classes(Ctrl+Shift+E)로 저장한 클래스 목록은 이 파일에 남는다. 위 원시 명령은 `data/`를 번들하지 않으므로, 번들 기본값까지 포함하려면 `pyinstaller labelImg.spec`(`datas=[(os.path.join(SPECPATH, 'data'), 'data')]`, `labelImg.spec:9`) 사용을 권장. 또한 원시 명령은 `-c`(콘솔 창 표시)인 반면 spec은 `console=False`(`labelImg.spec:109`)라 GUI 전용 exe가 만들어진다.

### onnxruntime(ONNX 런타임) 번들

`labelImg.spec`은 빌드 환경에 `onnxruntime`이 설치돼 있으면(`pip install -e ".[ai]"` 또는 CI의
`pip install pyinstaller ".[ai]"`) `onnxruntime`/`numpy`를 exe 안에 함께 번들한다 — AI 자동 라벨링 코드
(`libs/inference/yolo_onnx.py`)뿐 아니라 그 코드가 런타임에 실제로 필요로 하는 네이티브 런타임
(`onnxruntime_pybind11_state*.pyd`, `onnxruntime.dll`, `onnxruntime_providers_shared.dll`)까지 함께
들어간다는 뜻이다. 예전에는 exe에 AI *코드*만 들어가고 이 *런타임*이 빠져 있어서, 배포된 exe로는
`build_backend()`가 항상 `MissingDependency`로 실패했다(AI 메뉴가 늘 회색으로 남았다) — 지금은 고쳐졌다.

`collect_all('onnxruntime')`으로 통째로 긁는 방식은 시도했지만 빌드 자체가 깨진다: 그 방식이 함께 끌어오는
`onnxruntime.transformers.torch_onnx_export_helper`(`import torch` 포함)를 PyInstaller가 정적 분석하려는
순간 `pyinstaller-hooks-contrib`의 torch 훅이 격리 서브프로세스로 torch 서브모듈을 수집하려다 죽는다(exit
code 3). 이 포크는 순수 CPU 추론만 쓰고 학습/양자화/변환기-내보내기 헬퍼는 전혀 쓰지 않으므로,
`labelImg.spec`은 `collect_dynamic_libs('onnxruntime')` + `collect_data_files('onnxruntime')`와 추론에
실제로 필요한 hiddenimports 몇 개만 골라 쓰고, `torch`/`onnxruntime.training`/`onnxruntime.quantization`/
`onnxruntime.transformers`는 `Analysis(excludes=[...])`로 명시적으로 제외한다(`labelImg.spec:47-79`).
`onnxruntime`이 빌드 환경에 없으면(기본 설치) 이 블록은 통째로 건너뛰고, `data/` 번들 등 나머지는 그대로인
평범한(AI 비활성) exe가 만들어진다 — 빌드 실패가 아니라 런타임에 AI 메뉴가 꺼진 채로 동작한다.

> ⚠️ **exe에는 여전히 모델 가중치가 들어있지 않다.** 이 번들은 *런타임*(onnxruntime 자체)일 뿐, `.onnx`
> 가중치 파일이 아니다(라이선스 이유는 [`data/models/README.md`](../../data/models/README.md) 참고). 배포된
> exe를 받아 AI 메뉴를 쓰려면 사용자가 직접 `.onnx` 파일을 구해 **AI 메뉴 > Model Settings...**에서
> 지정해야 한다 — 앱 안에 파일 선택 대화상자(`Browse...`)가 있는 다이얼로그이고, 적용하면 재시작 없이
> 바로 AI 액션이 켜진다(자세한 절차는 [`docs/how-to/auto-label.md`](auto-label.md) 참고). "exe만 받으면
> AI가 바로 동작한다"는 뜻은 아니다 — 모델 파일은 여전히 사용자가 직접 준비해야 한다.

> ⚠️ 작업 폴더에는 최신 `dist\labelImg.exe`와 구버전이 든 `dist_labelimg.zip`이 있다. `dist\labelImg.exe`는 **2026-07-03에 `pyinstaller labelImg.spec`으로 재빌드**된 최신 빌드로, 같은 날의 소스 수정(`open_dir_dialog` 재작성 등)과 `data/` 번들이 모두 반영돼 있다. 반면 루트의 `dist_labelimg.zip`에 들어 있는 `labelImg.exe`는 **2026-06-26 구버전 빌드**(zip 파일 자체의 최종 수정 시각은 06-29)로 이후 수정이 반영되어 있지 않다 — 특히 구 exe로 이미지 폴더를 명령줄 인자로 열면 설정 pkl의 `lastOpenDir`(이전 세션 폴더)가 우선되어 `default_save_dir`를 덮어쓰고, 연 폴더의 XML 라벨(박스)이 표시되지 않는 버그가 그대로 재현된다(현재 소스는 명령줄로 넘긴 폴더가 우선하도록 수정됨: `labelImg.py:1562-1567`, 시작 시 호출은 `labelImg.py:620`). 최신 동작이 필요하면 `dist\labelImg.exe`를 쓰거나 `python labelImg.py`로 소스 실행할 것.

## E. 테스트

```shell
python -m unittest discover tests      # 또는 make testpy3
```

테스트 범위: `test_io`(VOC/CreateML 라운드트립)·`test_settings`(설정 영속화)·`test_stringBundle`(i18n 폴백)·`test_utils`(색상/정렬)·`test_qt`(앱 부팅)·`test_classify`(포크 분류의 원자적 이동·실패 주입 롤백·undo — 실제 `MainWindow` 구동)·`test_yolo_reader`(YOLO 견고성: classes.txt 부재·불량 라인)·`test_create_ml_reader`(CreateML 리더: verified 매칭·utf-8). 총 30개(8개 파일).

> ℹ️ `test_stringBundle`은 로케일 환경변수를 `os.environ.get()`으로 안전 조회하도록 고쳐져, `LC_ALL`/`LANG`이 없는 Windows PowerShell/cmd에서도 그대로 통과한다(`tests/test_stringBundle.py`). 별도 환경변수 설정이 필요 없다.
> `test_io`가 실행 중 `tests/tests.json`(`tests/test_io.py:49`)과 `tests/test.xml`(`tests/test_io.py:19`)을 만든다. 두 산출물 모두 `tests/.gitignore`에 등록돼 작업트리를 더럽히지 않는다.

## F. 패키징/배포(PyPI)

`setup.py`가 버전을 `libs/__init__.py`(현재 `1.8.6`)에서 동적으로 읽고, 콘솔 스크립트 진입점 `labelImg=labelImg.labelImg:main`을 등록한다. 배포는 `make pip_upload`(= `python3 setup.py upload`, `UploadCommand`가 sdist/bdist_wheel 빌드 후 twine upload + git tag — 로컬 태그만 생성하며, `git push --tags`는 `setup.py:79`에서 주석 처리돼 원격에 푸시되지 않는다).

관련: [../explanation/architecture.md](../explanation/architecture.md)(리소스 파이프라인) · [reset-and-troubleshoot.md](reset-and-troubleshoot.md)
