# How-to: PASCAL VOC(XML) 라벨링

PASCAL VOC는 labelImg의 **기본 포맷**이다(ImageNet에서 쓰는 형식). 이미지마다 별도 `.xml`이 생긴다.

## 절차

1. 앱을 실행한다 → [install-and-build.md](install-and-build.md).
2. `File → Open Dir`(`Ctrl+u`)로 이미지 폴더를 연다. 이때 저장 폴더는 **연 폴더로 자동 설정**된다 — 즉 기본적으로 XML이 이미지 옆에 생긴다.
3. 다른 폴더에 저장하려면 **폴더를 연 뒤에** `File → Change Save Dir`(`Ctrl+r`)로 저장 폴더를 지정한다. ⚠️ 메뉴의 Open Dir는 저장 폴더를 연 폴더로 덮어쓰므로(`open_dir_dialog` 말미, `labelImg.py:1585`), Change Save Dir를 Open Dir보다 먼저 하면 지정이 소리 없이 무효가 된다. (명령줄 3번째 인자 `save_dir`로 시작한 경우에는 시작 시점에 그 폴더가 저장 폴더로 유지된다.)
4. `Create RectBox`(`w`)로 박스를 그리고 라벨을 입력한다.
5. 마우스 우드래그로 박스를 복사·이동할 수 있다.
6. `Ctrl+s`로 저장한다.

포맷이 YOLO/CreateML로 바뀌어 있다면 툴바의 포맷 버튼을 눌러 **PascalVOC**로 되돌린다(VOC→YOLO→CreateML 순환).

## 출력 예

저장 폴더에 `이미지이름.xml`:

```xml
<annotation>
	<folder>images</folder>
	<filename>img001.jpg</filename>
	<path>/data/images/img001.jpg</path>
	<source><database>Unknown</database></source>
	<size><width>800</width><height>600</height><depth>3</depth></size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox><xmin>60</xmin><ymin>40</ymin><xmax>430</xmax><ymax>504</ymax></bndbox>
	</object>
</annotation>
```

- 좌표는 **절대 픽셀**(정수). 정규화 없음.
- 저장 시 xmin/ymin은 최소 1로 클램프되며(`libs/labelFile.py:168-172`), 박스가 이미지 경계에 닿으면 `truncated`가 자동으로 1로 기록된다(`libs/pascal_voc_io.py:94-99`).
- `difficult`는 보존된다(아래 참고).
- `verified` 플래그를 켜면 루트에 `verified="yes"`가 붙는다 → [verify-and-difficult.md](verify-and-difficult.md).

정확한 필드 규칙은 [../reference/formats.md](../reference/formats.md).

## 어노테이션 시각화(검수)

기존 라벨을 눈으로 확인하려면:
1. 라벨 `.xml`을 이미지와 **같은 폴더**, **같은 이름**으로 둔다.
2. `Open Dir`로 이미지 폴더를 연다.
3. File List에서 이미지를 고르면 박스·라벨이 표시된다.
4. `View → Display Labels`로 라벨 표시 on/off.

단, `Change Save Dir`로 저장 폴더를 바꾼 상태라면 XML은 이미지 옆이 아니라 **저장 폴더**에서 찾는다(`labelImg.py:1359-1373`). 같은 이름의 `.xml` > `.txt`(YOLO) > `.json`(CreateML) 순으로 탐색된다.

## 사전 정의 클래스

`data/predefined_classes.txt`(한 줄당 클래스명)를 편집해 라벨 입력창의 자동완성/리스트를 미리 채울 수 있다. 실행 시 두 번째 인자로 다른 클래스 파일을 줄 수도 있다:

```shell
python labelImg.py /data/images /data/my_classes.txt
```

클래스 목록은 `File → Edit Default Classes`(`Ctrl+Shift+E`)로 앱 안에서 편집·영구 저장할 수도 있다. ⚠️ 빌드된 exe에서는 `data/` 폴더가 아니라 **실행파일 옆의 `predefined_classes.txt`** 가 사용되며, 이 파일이 생성된 뒤에는 두 번째 인자가 무시된다(최초 1회 시드로만 사용) → 자세한 규칙은 [more-features.md](more-features.md#사전-정의-클래스-앱에서-편집-ctrlshifte) 참고.

관련: [annotate-yolo.md](annotate-yolo.md) · [annotate-createml.md](annotate-createml.md) · [../explanation/annotation-formats.md](../explanation/annotation-formats.md)
