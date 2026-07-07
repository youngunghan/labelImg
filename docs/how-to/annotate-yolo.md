# How-to: YOLO(txt) 라벨링

YOLO 포맷은 이미지마다 동일 이름 `.txt`를 만들고, 폴더에 `classes.txt`(클래스 인덱스 정의)를 함께 둔다. 학습기에 바로 넣기 좋은 정규화 좌표다.

## 절차

1. **먼저** 학습에 쓸 클래스 목록을 정의한다(한 줄당 하나). 앱 실행 후 **File → Edit Default Classes**(`Ctrl+Shift+E`)로 편집하는 것을 권장하며, 저장 시 영속 클래스 파일에 즉시 기록된다. 이 파일은 소스 실행(`python labelImg.py`) 시 `data/predefined_classes.txt`, **빌드된 exe 실행 시에는 exe 옆의 `predefined_classes.txt`**다(첫 실행 때 번들 기본값이 한 번 복사될 뿐, 이후에는 저장소의 `data/predefined_classes.txt`를 고쳐도 exe에 반영되지 않는다).
2. 앱을 실행한다.
3. 툴바의 **Save 버튼 바로 아래 포맷 버튼**을 눌러(또는 `Ctrl+Y`) `PascalVOC`에서 **YOLO**로 바꾼다(VOC→YOLO→CreateML 순환).
4. `Open`/`Open Dir`로 이미지를 열고 박스를 그린다.
5. 이미지마다 작업을 마치면 `Ctrl+s`로 저장한다.

## 출력

이미지와 같은 폴더에:

(`Change Save Dir`(`Ctrl+r`)로 저장 폴더를 바꾼 경우 `.txt`와 `classes.txt` 모두 그 폴더에 생성된다.)

`img001.txt` — 한 줄당 박스 하나:
```
0 0.306250 0.453333 0.462500 0.773333
```
형식은 `class_index x_center y_center width height`이고 모두 **0~1 정규화**(이미지 크기 기준).

`classes.txt` — 클래스 인덱스 정의(**첫 줄이 인덱스 0**, 이후 줄 순서대로 1, 2, …):
```
person
car
```

`classes.txt`는 **읽기 시** 클래스 이름을 결정하는 단일 출처다. 단, 이 파일은 저장할 때마다 앱의 클래스 목록(predefined classes + 세션 중 추가한 라벨)으로 **통째로 다시 생성**되므로, `classes.txt`를 직접 편집해도 다음 저장에서 덮어써진다. 클래스 정의는 1번 절차(predefined classes)에서만 하라.

## 주의사항 (중요)

- **라벨 목록을 작업 도중 바꾸지 말 것.** 이미지를 저장할 때마다 `classes.txt`가 갱신되지만, **이전에 저장한 라벨 파일의 인덱스는 갱신되지 않아** 매핑이 어긋난다. 특히 작업 도중 **Edit Default Classes**(`Ctrl+Shift+E`)로 클래스 순서를 바꾸거나 삭제하면 다음 저장부터 새 순서로 `classes.txt`와 인덱스가 기록되어, 이전에 저장한 txt의 인덱스가 즉시 어긋난다.
- **YOLO 저장 시 `difficult` 플래그는 폐기된다.** 다시 읽으면 모든 박스가 `difficult=False`가 된다.
- YOLO 저장에는 **"default class"(single-class) 기능이 참조되지 않는다.**
- `classes.txt`는 인코딩 지정 없이 쓰이므로 비ASCII 클래스명은 플랫폼 간 불일치 가능성이 있다.
- `classes.txt`가 없는 폴더의 txt를 열면 에러 대화상자가 뜨고 이미지는 라벨 없이 표시된다(크래시하지 않음). 형식이 틀린 라인(NaN/inf 좌표 포함)은 건너뛰고 상태바에 `Skipped N malformed line(s)`로 알린다(포크 견고화, `libs/yolo_io.py:104-168`).

정확한 좌표 변환식·역변환은 [../reference/formats.md](../reference/formats.md#2-yolo-txt--classestxt).

## CSV로 내보내기

YOLO txt 묶음을 Google Cloud AutoML CSV로 바꾸려면 [export-to-csv.md](export-to-csv.md).

관련: [annotate-pascal-voc.md](annotate-pascal-voc.md) · [../explanation/annotation-formats.md](../explanation/annotation-formats.md)
