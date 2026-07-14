# 튜토리얼: 처음 실행

labelImg를 처음부터 띄워 이미지 한 장에 박스를 그리고 저장하는 happy path. 약 5분.

## 0. 준비

Python 3 + PyQt5 + lxml 이 필요하다. 가장 빠른 길:

```shell
pip3 install labelImg
labelImg
```

> ⚠️ 이 명령으로 설치되는 PyPI 패키지는 업스트림 `labelImg`이며, **이 포크의 기능(AI 자동 라벨링, COCO 가져오기/내보내기 등)을 전혀 포함하지 않습니다.** 추가로 업스트림 1.8.6은 최신 PyQt5에서 박스 그리기나 줌 시 충돌하는 알려진 결함이 있습니다(이슈 #987/#988/#938). **이 저장소에서 소스로 실행**하는 것을 권합니다:

소스에서 실행하려면(이 저장소) → [../how-to/install-and-build.md](../how-to/install-and-build.md). Windows에서 소스 실행 시 먼저 리소스를 컴파일해야 한다:

```shell
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```

## 1. 실행

```shell
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE] [SAVE_DIR]
```

인자 없이 실행하면 빈 창이 뜬다. 창은 가운데 **캔버스**, 우측에 **Box Labels**(라벨 리스트)와 **File List**(파일 목록) 도크로 구성된다.

`SAVE_DIR`를 주면 어노테이션 저장 폴더로 사용된다(설정에 저장된 이전 저장 폴더보다 우선). `IMAGE_PATH`가 폴더든 단일 이미지 파일이든 유효하다. 단, 실행 후 메뉴에서 `Open Dir`로 폴더를 다시 열면 저장 폴더가 새로 연 폴더로 재설정되므로, 그때는 `Change Save Dir`(`Ctrl+r`)로 다시 지정한다.

## 2. 이미지 폴더 열기

`File → Open Dir`(`Ctrl+u`)로 이미지가 든 폴더를 연다. 폴더의 이미지들이 File List에 자연 정렬되어 채워지고 첫 이미지가 캔버스에 표시된다. (한 장만 열려면 `Open`.) 폴더를 열면 **저장 폴더가 자동으로 그 폴더로 설정**되어 라벨 파일이 이미지 옆에 저장된다.

## 3. 저장 폴더 변경 (선택)

라벨을 다른 폴더에 저장하려면 **폴더를 연 뒤에** `File → Change Save Dir`(`Ctrl+r`)로 바꾼다. 주의: `Open Dir`를 다시 실행하면 저장 폴더가 새로 연 폴더로 초기화되므로, Change Save Dir는 항상 Open Dir 이후에 해야 한다.

## 4. 박스 그리기

1. `Create RectBox`(단축키 `w`)를 누른다 → 커서가 십자(+)로 바뀐다.
2. 객체의 한 모서리에서 마우스 버튼을 **누른 채** 대각 모서리까지 끌고 가서 버튼을 **놓으면** 박스가 완성된다. (누른 자리에서 움직이지 않고 그냥 클릭만 하면 그리기가 취소되고 EDIT 모드로 돌아간다.)
3. 박스를 완성하면 **라벨 입력 창**이 뜬다. 클래스명을 입력(또는 기존 라벨 클릭)하고 OK. (우측 도크의 'Use default label'이 체크되어 있으면 입력 창 없이 기본 라벨이 바로 적용된다.)

박스를 옮기거나 크기를 바꾸려면 EDIT 모드(기본)에서:
- 박스 안을 잡고 드래그 → 이동
- 모서리 정점을 잡고 드래그 → 크기 변경
- 방향키 → 1픽셀씩 이동

## 5. 저장

`Ctrl+s`로 저장한다. 기본 포맷은 **PASCAL VOC(.xml)**. 저장 폴더(또는 이미지 옆)에 `이미지이름.xml`이 생긴다.

## 6. 다음 이미지

`d`(다음) / `a`(이전)로 이미지를 넘긴다. **Auto Save**가 켜져 있으면 넘어갈 때 자동 저장된다(`View → Auto Save mode`). 단, 저장 폴더가 아직 정해지지 않았다면(Open으로 한 장만 연 경우 등) 자동 저장 대신 저장 폴더 선택 창이 먼저 뜬다.

## 다음 단계

- 포맷을 바꾸려면(YOLO/CreateML) → 툴바의 포맷 버튼을 누르거나 [../how-to/annotate-yolo.md](../how-to/annotate-yolo.md)
- 단축키 전체 → [../reference/shortcuts.md](../reference/shortcuts.md)
- 동작 원리 → [../explanation/architecture.md](../explanation/architecture.md)
