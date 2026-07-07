# How-to: 설정 리셋 · 문제 해결

## 설정 리셋

설정(창 배치·저장 폴더·최근 파일 등)이 꼬였을 때:

1. 메뉴 `File → Reset All`, 또는
2. 홈 디렉터리의 `~/.labelImgSettings.pkl`을 직접 삭제:
   ```shell
   rm ~/.labelImgSettings.pkl        # Linux/macOS
   del %USERPROFILE%\.labelImgSettings.pkl   # Windows
   ```

`make clean`도 이 파일을 지운다. 내부적으로 `Reset All`은 `Settings.reset()`(파일 삭제 + 메모리 초기화) 후 창을 닫고 `QProcess.startDetached`로 앱을 재시작한다(`labelImg.py:1749-1753`). 단, 재시작은 `os.path.abspath(__file__)`(파이썬 스크립트 경로)를 직접 실행하는 방식이라 Windows나 빌드된 exe에서는 자동 재시작이 안 될 수 있다 — 그 경우 창이 닫힌 뒤 수동으로 다시 실행하면 된다(설정 초기화 자체는 완료된 상태).

> 설정 로더(`Settings.load`)는 손상된 pkl을 무시하고 빈 설정으로 동작한다(GUI에는 에러가 뜨지 않고, 콘솔 실행 시에만 `Loading setting failed`가 출력된다 — `libs/settings.py:37`). "왜 설정이 안 먹지" 싶으면 파일 삭제가 가장 확실하다.

## 자주 겪는 문제

### 클래스 목록이 안 뜨거나 이상하다

클래스 목록은 설정 pkl이 아니라 `predefined_classes.txt`에서 읽는다(소스 실행: `data/predefined_classes.txt`, exe 실행: exe 옆 `predefined_classes.txt` — `get_persistent_classes_file`, `labelImg.py:1835`). pkl을 지우거나 `Reset All`을 해도 클래스는 바뀌지 않는다. 메뉴 `File → Edit Default Classes`(Ctrl+Shift+E)로 수정하거나 해당 txt 파일을 직접 편집하라(한 줄에 한 클래스).

### 아이콘/문자열이 안 보이거나 import 에러

`libs/resources.py`가 없거나 옛 버전이다. 리소스를 다시 컴파일한다:
```shell
pyrcc5 -o libs/resources.py resources.qrc     # PyQt5
```
PyQt5는 `pyrcc5`, 구버전 PyQt4는 `pyrcc4`다. 섞으면 안 된다. → [install-and-build.md](install-and-build.md)

### YOLO 라벨 인덱스가 어긋난다

작업 도중 클래스 목록을 바꿨을 가능성이 크다. `classes.txt`는 저장 때마다 갱신되지만 **이전 라벨 파일의 인덱스는 갱신되지 않는다.** 한 묶음을 처리하는 동안 클래스 목록을 고정하라. → [annotate-yolo.md](annotate-yolo.md)

### difficult/verify가 사라진다

포맷별 지원이 다르다. YOLO는 difficult를 저장하지 않고, CreateML은 읽을 때 difficult를 모두 True로 만든다. verify는 YOLO에 저장되지 않는다. → [verify-and-difficult.md](verify-and-difficult.md) · [../reference/formats.md](../reference/formats.md)

### 라벨 파일이 안 열리는데 에러도 없다

`PascalVocReader`는 모든 파싱 예외를, `CreateMLReader`는 JSON 디코드 실패(ValueError)를 삼키고 **빈 결과**를 돌려준다(조용한 실패). 단 CreateML JSON이 문법은 맞지만 구조가 다르면(필수 키 누락 등) 예외가 그대로 전파된다. 라벨 파일이 손상됐거나 포맷이 안 맞으면 박스가 그냥 안 뜬다. 파일을 직접 열어 구조를 확인하라. → [../reference/formats.md](../reference/formats.md)

라벨 파일이 멀쩡한데도 박스가 안 뜨면 `default_save_dir`가 다른 폴더를 가리키는지 의심하라. `default_save_dir`가 설정돼 있으면 라벨은 그 폴더에서**만** 찾고, 설정이 없을 때만 이미지 옆에서 찾는다(`labelImg.py:1204-1231`). 설정 pkl에 남은 이전 세션의 저장 폴더가 시작 시 `default_save_dir`로 복원되므로(`labelImg.py:519-522`) 이것이 원인일 수 있다. `Change Save Dir`로 이미지 폴더를 다시 지정하거나 위의 설정 리셋을 하라. 참고: 명령줄 인자로 이미지 폴더를 넘겨도 pkl의 마지막 폴더가 우선돼 라벨을 못 찾던 버그는 소스(`python labelImg.py`)와 2026-07-03 재빌드된 `dist\labelImg.exe`에서는 수정됐다(`open_dir_dialog`, `labelImg.py:1367-1398`). 구버전 exe(루트 `dist_labelimg.zip`에 포함, 2026-06-26 빌드; zip 파일 자체의 최종 수정 시각은 06-29)에는 이 버그가 남아 있다 — 그 exe를 압축 해제해 쓸 때는 pkl 삭제가 우회책이다.

### Open으로 단일 파일을 열었더니 File List가 사라짐

`load_file`은 열려는 파일이 현재 이미지 목록(`m_img_list`)에 없으면 File List를 비운다(`labelImg.py:1136-1138`). 폴더 단위로 작업하려면 `Open Dir`을 쓰라.

### 한글(비ASCII) 클래스명이 깨진다

라벨 txt는 UTF-8로 쓰지만 YOLO의 `classes.txt`는 인코딩 지정 없이 쓰여 플랫폼 기본 인코딩을 탄다. 비ASCII 클래스명은 Windows/Linux 간 불일치 가능성이 있다.

관련: [../reference/settings.md](../reference/settings.md) · [../reference/modules.md](../reference/modules.md)
