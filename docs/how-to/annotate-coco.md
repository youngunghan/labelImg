# How-to: COCO(JSON) 라벨링

COCO는 이 포크가 지원하는 4번째 포맷이며, 나머지 셋(PascalVOC/YOLO/CreateML)과 근본적으로 다르다.
**COCO는 이미지별 사이드카가 아니라 데이터셋 전체를 담는 파일 하나**다 — `images[]`/`annotations[]`/`categories[]`가
여러 이미지를 한 json에 함께 담는다. 그래서 이 앱도 COCO만은 "이미지 stem에서 유도한 파일"이 아니라
**고정된 데이터셋 json 하나**로 취급한다(`libs/coco_io.py:11-17`, `labelImg.py:670-687`).

## 절차

1. 앱을 실행한다.
2. 툴바 포맷 버튼(`Ctrl+Y`)을 세 번 눌러 **COCO**로 바꾼다(PascalVOC→YOLO→CreateML→**COCO**→PascalVOC 순환,
   `labelImg.py:651-661`).
3. `Open`/`Open Dir`로 이미지를 열고 박스를 그린다.
4. `Ctrl+S`로 저장한다.

## 데이터셋 json은 어디에 쓰이나

- **기본 대상**: `<저장 폴더>/annotations.json` — `Change Save Dir`로 지정한 저장 폴더(없으면 이미지가 있는
  폴더) 밑의 `annotations.json`(`COCO_DEFAULT_DATASET_NAME`, `libs/coco_io.py:17`)이다
  (`coco_dataset_target`, `labelImg.py:670-687`).
- **File → Import COCO...** / **File → Export COCO...**(`labelImg.py:257-260`, 단축키 없음)로 다른 json을
  명시적으로 고를 수 있다. 한 번 고르면 그 파일이 이후 저장·자동저장의 대상으로 세션 내내 고정된다
  (`self.coco_dataset_path`, `labelImg.py:2287`·`2312`).
  - **Import COCO...**: 기존 COCO 데이터셋 json을 열어 그 안에서 **현재 이미지에 해당하는 항목**만 불러온다.
    데이터셋에 이 이미지가 없으면 상태바에 "No annotations for ..." 로 알리고 포맷만 COCO로 바뀐다.
  - **Export COCO...**: 저장 대상 json을 고른 뒤(없는 파일이면 새로 만들어짐) **즉시 현재 이미지를 그 파일에
    병합 저장**한다.

## 저장 시 벌어지는 일 (read-modify-write 병합)

`Ctrl+S`를 누르면(또는 다음/이전 이미지로 넘어갈 때 **Auto Save**가 켜져 있어 자동 저장이 걸리면) 다음이
일어난다(`save_labels`의 COCO 분기, `labelImg.py:1058-1069`; `save_file`의 COCO 선분기,
`labelImg.py:1665-1675`):

1. 대상 데이터셋 json이 이미 있으면 통째로 읽는다. 없으면 빈 데이터셋으로 시작한다.
2. **현재 이미지의 `images[]`/`annotations[]` 항목만** 지금 캔버스 내용으로 교체한다.
3. **다른 이미지들의 항목은 그대로 보존된다** — COCOWriter가 이미지 id로 걸러서 이 이미지의 annotations만
   지우고 다시 쓴다(`COCOWriter.save`, `libs/coco_io.py:216-256`).
4. 클래스는 `category_id`로 매핑되고, 이미 파일에 있던 카테고리 id는 유지된 채 새 클래스만 다음 id로
   추가된다(`sync_categories`, `libs/coco_io.py:140-161`) — 병합할 때마다 기존 어노테이션의 카테고리가
   바뀌는 일은 없다.
5. **다른 포맷과 달리 `Ctrl+S`/자동저장이 COCO를 건드리지 않는 예외는 없다.** "COCO는 자동저장에 안 걸린다"는
   설계는 폐기됐고, 지금은 자동저장도 같은 데이터셋 파일로 병합된다. 다만 **이미지별 `<stem>.json` 사이드카는
   COCO에서는 절대 만들어지지 않는다** — 항상 데이터셋 json 하나로 몰린다는 점은 그대로다.

## 출력 형식

`annotations.json`(요지):

```json
{
  "images": [
    { "id": 1, "file_name": "train/0001.jpg", "width": 640, "height": 480 }
  ],
  "annotations": [
    { "id": 1, "image_id": 1, "category_id": 1, "bbox": [120, 80, 200, 150], "area": 30000, "iscrowd": 0 }
  ],
  "categories": [
    { "id": 1, "name": "person", "supercategory": "" }
  ]
}
```

- `bbox`는 `[x, y, width, height]`(좌상단 기준), `area = width * height`, `iscrowd`는 항상 `0`
  (`COCOWriter.save`, `libs/coco_io.py:235-251`).
- `file_name`은 파일 basename이 아니라 **데이터셋 json이 있는 폴더 기준 상대경로**다
  (`dataset_relative_name`, `libs/coco_io.py:34-61`). `Open Dir`는 하위 폴더까지 재귀로 스캔하므로
  (`os.walk`), `train/0001.jpg`와 `val/0001.jpg`처럼 같은 basename을 가진 두 이미지가 서로 다른 항목으로
  구분되게 하기 위한 설계다. 이미지 경로와 데이터셋 json이 같은 드라이브의 절대경로가 아니면(둘 중 하나가
  상대경로거나 Windows에서 드라이브가 다르면) basename으로 대체된다.
- 파일은 `indent=2`로 저장돼 사람이 읽기 편하다(CreateML과 반대, `libs/coco_io.py:256`).

## `.json` 확장자 충돌 (COCO vs CreateML)

CreateML도 `.json`을 쓰므로, 확장자만으로는 어느 리더를 써야 할지 알 수 없다. 그래서 `.json`을 여는 모든
경로(폴더 스캔 시 자동 로드, Import 대화상자)가 **파일을 열어 내용을 스니핑**한 뒤 리더를 고른다
(`load_json_by_filename`, `labelImg.py:2231-2241`):

- 최상위가 `images`/`annotations`/`categories` 중 하나라도 가진 **dict**면 → COCO (`is_coco_dict`,
  `libs/coco_io.py:82-91`)
- 최상위가 **list**면 → CreateML

이미지를 열 때 라벨 파일을 찾는 우선순위는 **PascalVOC(.xml) > YOLO(.txt) > CreateML(`<stem>.json`) > COCO
(데이터셋 json)** 이다(`show_bounding_box_from_annotation_file`, `labelImg.py:1342-1379`) — 즉 저장 폴더에
`<stem>.xml`이나 `<stem>.json`(CreateML 형식)이 이미 있으면 COCO 데이터셋보다 그것들이 먼저 로드된다.
COCO 데이터셋 json은 **그 이미지에 대한 항목이 실제로 있을 때만** 채택된다 — 저장 폴더에 우연히 다른
용도의 `annotations.json`이 있어도 앱이 멋대로 COCO로 전환되지는 않는다(`load_coco_json_by_filename`,
`labelImg.py:2256-2281`).

## 한계 (정직하게 밝힘)

- **`difficult`와 `verified`는 COCO 스키마에 저장 슬롯이 없다.** 두 플래그 모두 저장 시 버려지고, 다시
  읽으면 항상 `difficult=False`(`COCOReader.add_shape`, `libs/coco_io.py:330-336`) /
  `verified=False`(`COCOReader.__init__`, `libs/coco_io.py:273`)로 들어온다. `verify_image`(Space)로
  검증해도 캔버스에는 반영되지만 저장 후 상태바에 "COCO stores no verified/difficult flag" 라는 안내가
  뜬다(`labelImg.py:1587-1593`) — 다음 로드에서 조용히 사라지는 대신 그 자리에서 알려준다.
- **이미지 분류(Good/Bad, `g`/`b`)로 이미지를 옮기면 COCO 데이터셋의 항목은 그대로 남는다.** COCO는
  이미지별 사이드카가 없어서(다른 이미지들의 항목까지 들어있는 공용 파일이라) 함께 옮길 파일이 없다.
  분류 후 상태바에 "COCO dataset entry left in place; boxes were not moved" 라고 명시적으로 알린다
  (`labelImg.py:1887-1891`). 즉 이미지를 옮긴 뒤에는 데이터셋 json에 더 이상 존재하지 않는 경로를 가리키는
  고아 항목이 남으므로, 필요하면 수동으로 정리해야 한다. 같은 이유로 이미지 파일을 앱 밖에서 직접
  옮기거나 이름을 바꿔도 데이터셋 항목은 따라가지 않는다 — 새 경로는 새 항목으로 다시 저장해야 한다.
- COCO 저장에는 YOLO처럼 **"default class"(single-class) 기능이 참조되지 않는다**(다른 포맷과 동일한
  `shapes` 리스트를 그대로 넘길 뿐, single-class 로직은 캔버스 쪽 관심사).

관련: [annotate-yolo.md](annotate-yolo.md) · [annotate-createml.md](annotate-createml.md) ·
[../explanation/ml-assist-architecture.md](../explanation/ml-assist-architecture.md#7-coco-저장-경로가-다르다--데이터셋-레인)
