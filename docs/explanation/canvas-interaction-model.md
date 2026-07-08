# 캔버스 상호작용 모델

`Canvas`(`libs/canvas.py:24`)는 마우스·키보드·휠로 박스를 그리고 편집하는 모든 동작의 무대다. 핵심은 **두 모드(CREATE/EDIT)** 와 **선택 상태**가 같은 이벤트 핸들러의 분기를 결정한다는 점이다.

## 두 모드

```python
CREATE, EDIT = list(range(2))   # canvas.py:33
```

- `drawing()` = `mode == CREATE`, `editing()` = `mode == EDIT` (`canvas.py:88-92`). 초기 모드는 EDIT.
- `set_editing(value)`로 전환한다. CREATE로 들어갈 때(`value=False`) 하이라이트와 선택을 해제한다(`canvas.py:94-100`).
- MainWindow의 "Create RectBox"(단축키 `w`) 액션이 CREATE로, 박스를 다 그리면 다시 EDIT로 돌아온다.

## CREATE 모드 — 박스 그리기

사각형 전용으로 최적화돼 있어, **누르고-끌고-놓기(press-drag-release) 한 번**으로 4꼭짓점 박스가 만들어진다:

1. **버튼 누름**(`mousePressEvent` → `handle_drawing` 1차, `canvas.py:261-263`, `334-340`): `current = Shape()`를 만들고 시작점을 `add_point`, 미리보기선 `line`을 `[pos, pos]`로 초기화, `drawingPolygon.emit(True)`.
2. **드래그**(`mouseMoveEvent`, `canvas.py:111-165`): `line[1]`을 현재 위치로 갱신해 고무줄 사각형을 그린다. 픽셀맵 밖이면 좌표를 `[0,max]`로 클리핑한다. (시작점 흡인 `close_enough` 분기는 `len(current) > 1`을 요구하는데 드래그 중 `current`는 시작점 1개뿐이라 이 플로우에선 도달하지 않는다 — 과거 다각형 도구의 흔적.)
3. **버튼 놓음**(`mouseReleaseEvent` → `handle_drawing` 2차, `canvas.py:292-295`, `322-333`): 시작점과 놓은 지점(`line[1]`)을 대각으로 보고 `(max_x,min_y)`,`(target)`,`(min_x,max_y)` 세 점을 추가해 4점을 채운 뒤 `finalise`. **제자리 클릭**(이동 없이 누르고 놓기)은 4점이 모두 같아져 `finalise`의 영-크기 가드(`canvas.py:576-580`)에 걸려 버려지고 `drawingPolygon(False)`가 emit되며, beginner 모드(기본)에선 `toggle_drawing_sensitive`(`labelImg.py:746-754`)가 CREATE 모드 자체를 취소하고 EDIT로 되돌린다.
4. **finalise**(`canvas.py:574-587`): 첫 점==끝 점이면(영-크기 박스) 버리고, 아니면 `current.close()` 후 `shapes`에 추가하고 `newShape.emit()`. → MainWindow가 `LabelDialog`를 띄워 라벨을 받는다. 라벨 입력을 취소하면 MainWindow가 `reset_all_lines`(`canvas.py:698-706`)를 호출해 방금 추가된 박스를 `shapes`에서 pop하고 `drawingPolygon(False)`를 emit해 그리기를 없던 일로 되돌린다.

`Esc`는 그리던 박스를 취소(`current=None`, `drawingPolygon.emit(False)`)하고, `Return`은 닫을 수 있으면 `finalise`한다(`canvas.py:629-645`).

### 정사각형 그리기(draw_square)

`Ctrl` 키를 누르고 있으면(MainWindow의 `keyPressEvent`/`keyReleaseEvent`가 `set_drawing_shape_to_square` 토글, `labelImg.py:579-586`) 그리기 중 `line[1]`이 시작점 기준 정사각형 좌표로 강제된다(`canvas.py:148-155`).

## EDIT 모드 — 선택·이동·정점편집

마우스 이동/드래그의 동작은 **무엇이 선택/하이라이트돼 있는지**로 갈린다:

- **호버**(버튼 없이 이동): `shapes`(+선택 도형)를 역순으로 훑어 `nearest_vertex`로 정점을, 실패하면 `contains_point`로 도형을 하이라이트한다(`canvas.py:218-256`). 아무것도 안 걸리면 `for-else`의 `else`에서 하이라이트를 모두 해제한다.
- **좌클릭 드래그**(`canvas.py:179-211`):
  - 정점이 잡혀 있으면 → `bounded_move_vertex`(정점 이동) 후 `shapeMoved.emit`.
  - 도형이 선택돼 있으면 → `bounded_move_shape`(도형 전체 이동) 후 `shapeMoved.emit`.
  - 둘 다 아니면 → **패닝**(`scrollRequest` 두 방향 emit).
- **우클릭 드래그**(`canvas.py:168-176`): 선택 도형의 **그림자 복사본**(`selected_shape_copy`)을 만들어 이동한다. 마우스를 떼면 컨텍스트 메뉴로 "여기에 복사/이동"을 확정하거나 취소한다(`mouseReleaseEvent`, `canvas.py:278-298`). 비파괴 편집 UX다.

정점 이동은 경계 클리핑되고, 인접 두 정점이 축별로 동반 이동해 어느 모드에서든 직사각형이 유지된다(`bounded_move_vertex`, `canvas.py:400-434`; 클리핑 403-407, 인접 정점 동반 이동 423-434). `draw_square` 모드에선 추가로 대각 반대 정점을 고정점으로 목표 위치가 정사각형 좌표로 강제된다(`canvas.py:409-417`). 방향키는 선택 도형을 1픽셀씩 이동한다(`move_one_pixel`, `canvas.py:647-674`).

> **사각형 가정**: 이동·정점 로직은 `shape[1]`/`shape[3]` 같은 인덱스로 대각 꼭짓점에 접근하고 `(index+2)%4` 산술을 쓴다 — 모두 4꼭짓점 박스를 전제한다. `drawingPolygon`·`finalise`·`set_open/close` 같은 polygon 용어는 과거 다각형 도구의 흔적이며 현재는 사실상 rectangle 전용으로 동작한다. `close_enough` 흡인 분기(`canvas.py:140`, `epsilon=24.0`)는 `len(current) > 1`을 요구하는데, 사각형 플로우에선 그리는 동안 `current`가 항상 점 1개(나머지 3점은 두 번째 `handle_drawing` 호출이 추가한 즉시 `finalise`되어 `current`가 비워짐, `canvas.py:330-333`)라 실행되지 않는다 — `drawingPolygon` 등과 마찬가지로 과거 다각형 도구의 흔적이다.

## 렌더링(paintEvent)

`paintEvent`(`canvas.py:495-555`)는 `p.scale(self.scale)`과 `p.translate(offset_to_center)`로 좌표계를 맞춘 뒤:
1. `pixmap`을 그리고, `overlay_color`가 있으면 `CompositionMode_Overlay`로 밝기 오버레이를 덮는다.
2. 보이는 `Shape`들을 그린다(`shape.fill = shape.selected or shape == h_shape`로 채움 여부 결정).
3. CREATE 모드에서 버튼 누름 전 호버 중이면 검은 십자선을(`canvas.py:540-543`), 드래그 중(`current`가 존재하는 동안)이면 실선 테두리에 빗금(`BDiagPattern`) 브러시로 채운 미리보기 사각형을(`canvas.py:530-538`) 그린다 — `prev_point`가 호버 시 설정되고 `current`가 생긴 뒤 첫 마우스 이동에서 비워지므로(`canvas.py:160-163`) 두 요소는 사실상 상호 배타적으로 나타난다.
4. 배경은 **verified면 연녹색**`QColor(184,239,38,128)`, 아니면 회색(`canvas.py:546-553`) — verify 플래그의 시각 신호다.

좌표 변환은 `transform_pos`(위젯→이미지: `point/scale - offset_to_center`)와 `out_of_pixmap`(범위 판정, 경계 포함)으로 한다(`canvas.py:557-572`).

## 줌·밝기·스크롤(wheelEvent)

`wheelEvent`(`canvas.py:605-627`)는 수정자 조합으로 분기한다:
- `Ctrl + Shift + 휠` → `lightRequest`(밝기)
- `Ctrl + 휠` → `zoomRequest`(줌)
- 그 외 → `scrollRequest`(스크롤/패닝)

실제 줌 배율과 밝기는 MainWindow의 `ZoomWidget`/`LightWidget` 값으로 결정되고, `paint_canvas`(`labelImg.py:1256`)가 `canvas.scale = 0.01*zoom_widget.value()`, `canvas.overlay_color = light_widget.color()`를 세팅한다. → [../reference/shortcuts.md](../reference/shortcuts.md) · [architecture.md](architecture.md)

## 커서

상태별 전역 override 커서로 피드백을 준다(`canvas.py:15-19`, `override_cursor`): 기본 화살표, 그리기 십자(Cross), 정점 근처 포인팅손, 도형 이동 닫힌손, 패닝 열린손.
