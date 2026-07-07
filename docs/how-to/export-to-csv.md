# How-to: 라벨을 CSV로 내보내기 (AutoML)

`tools/label_to_csv.py`는 YOLO txt 또는 PASCAL VOC xml 라벨 묶음을 **Google Cloud AutoML 객체 검출용 단일 CSV**로 변환하는 독립 스크립트다. labelImg 본체(PyQt5)와는 import 관계가 없고 `pandas`만 필요하다.

## 입력 폴더 구조

스크립트는 `location` 아래를 **2단계**로 순회한다: `set 폴더(TRAINING/VALIDATION/TEST/UNASSIGNED)` → `클래스 폴더`. (자세한 예시는 `tools/README.md`.)

## 실행

```shell
cd tools
python label_to_csv.py -p <bucket_prefix> -l <labels_root> -m <txt|xml> [-c <classes.txt>]
```

| 인자 | 필수 | 설명 |
|---|---|---|
| `-p` / `--prefix` | ✅ | GCS 버킷 프리픽스. 이미지 URI를 `gs://{prefix}/...`로 구성 |
| `-l` / `--location` | ✅ | 라벨 루트 디렉터리 |
| `-m` / `--mode` | ✅ | `txt`(YOLO) 또는 `xml`(VOC) |
| `-c` / `--classes` | | 클래스 파일(기본 `../data/predefined_classes.txt`) — txt 모드의 인덱스→이름 매핑 |
| `-o` / `--output` | | (파싱되지만 아래 주의 참고) |

예:
```shell
python label_to_csv.py -p test -l /User/test/labels -m txt
```

## 출력

`res.csv`(헤더 없음). 각 행:

```
<set>, gs://<prefix>/<class>/<img>.jpg, <label>, x_min, y_min, , , x_max, y_max, ,
```

- 좌표는 모두 **[0,1] 정규화**.
  - `txt` 모드: YOLO center/size를 `cx±w/2`, `cy±h/2`로 변환 후 `[0,1]` 클램프.
  - `xml` 모드: 픽셀 bndbox를 `size`의 width/height로 나눠 정규화(**클램프 없음** — bndbox가 이미지 크기를 벗어나면 1 초과 값이 그대로 기록됨).
- AutoML 박스 정의에 쓰는 두 모서리(x_min,y_min / x_max,y_max)만 채우고 나머지 모서리는 빈 칸.
- 이미지 경로는 항상 `.jpg` 확장자로 강제된다.

## 주의사항(코드 특성)

- **출력 파일명은 `-o`를 무시하고 항상 `res.csv`로 하드코딩**돼 있다(`tools/label_to_csv.py:215`).
- `txt` 모드는 **이미 정규화된 YOLO 좌표**를 가정한다(절대 픽셀 txt를 넣으면 `[0,1]` 클램프로 뭉개진다).
- 이미지 확장자가 `.jpg`가 아니면(png 등) URI가 어긋난다.
- 빈 라벨 파일이 있으면 `pandas.read_csv`가 예외를 낼 수 있다(예외 처리 없음).
- 클래스 파일은 **xml 모드에서도 존재해야 한다** — 파일이 없으면 모드와 무관하게 `File: ... not exists`를 출력하고 종료한다(`tools/label_to_csv.py:161-168`). `cd tools` 없이 실행하면 기본 상대경로가 어긋난다.
- set 폴더 바로 아래에 **폴더가 아닌 파일**이 섞여 있으면 그 파일은 건너뛴다(각 항목의 자식 경로를 `isdir`로 검사, `tools/label_to_csv.py:185-189`). 그래도 set 폴더 아래에는 클래스 폴더만 두는 것을 권장한다.
- `-m` 값 검증은 첫 클래스 폴더 처리 시점에야 일어난다(`tools/label_to_csv.py:202-206`). `location` 아래에 순회할 폴더가 없으면 잘못된 모드라도 에러 없이 빈 `res.csv`가 만들어진다.

소스 인벤토리 → [../reference/modules.md](../reference/modules.md). YOLO 포맷 → [../reference/formats.md](../reference/formats.md).
