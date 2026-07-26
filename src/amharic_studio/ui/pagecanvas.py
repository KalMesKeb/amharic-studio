"""The scan pane: page image with word-box overlays and two-way sync with the editor.

All the boxes are painted by a single custom item rather than one QGraphicsItem each. A
dense page can carry over a thousand words, and per-item overhead makes panning stutter at
that count, which ruins the one interaction the editor performs constantly.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from ..core.project import WordBox
from . import theme

MIN_ZOOM = 0.05
MAX_ZOOM = 12.0


class _OverlayItem(QGraphicsItem):
    """Paints every word box in one go, plus the active and flagged highlights."""

    def __init__(self, palette: theme.Palette) -> None:
        super().__init__()
        self.palette = palette
        self.boxes: list[WordBox] = []
        self.active_index: int | None = None
        self.flagged: dict[int, str] = {}  # box index -> severity
        self._bounds = QRectF()
        self.show_boxes = True
        self.show_heatmap = True
        self.setAcceptHoverEvents(True)
        self.hover_index: int | None = None

    def set_boxes(self, boxes: list[WordBox], page_size: tuple[int, int]) -> None:
        self.prepareGeometryChange()
        self.boxes = boxes
        self._bounds = QRectF(0, 0, page_size[0], page_size[1])
        self.active_index = None
        self.hover_index = None
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        return self._bounds

    def index_at(self, point: QPointF) -> int | None:
        for index, box in enumerate(self.boxes):
            if QRectF(box.x, box.y, box.w, box.h).contains(point):
                return index
        return None

    def paint(self, painter: QPainter, option, widget: QWidget | None = None) -> None:  # noqa: N802
        if not self.boxes:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        visible = option.exposedRect if option is not None else self._bounds

        for index, box in enumerate(self.boxes):
            rect = QRectF(box.x, box.y, box.w, box.h)
            if not visible.intersects(rect):
                continue

            if self.show_heatmap and box.conf > 0:
                painter.fillRect(rect, theme.confidence_color(box.conf, self.palette))

            severity = self.flagged.get(index)
            if severity:
                color = theme.severity_color(self.palette, severity)
                painter.setPen(QPen(color, 0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                # A thick underline reads clearly even when the box itself is tiny.
                painter.fillRect(
                    QRectF(rect.left(), rect.bottom() - 2, rect.width(), 2.5), QBrush(color)
                )
            elif self.show_boxes:
                pen_color = QColor(self.palette.border)
                pen_color.setAlpha(110)
                painter.setPen(QPen(pen_color, 0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

        if self.hover_index is not None and 0 <= self.hover_index < len(self.boxes):
            box = self.boxes[self.hover_index]
            hover = QColor(self.palette.accent)
            hover.setAlpha(40)
            painter.fillRect(QRectF(box.x, box.y, box.w, box.h), hover)

        if self.active_index is not None and 0 <= self.active_index < len(self.boxes):
            box = self.boxes[self.active_index]
            rect = QRectF(box.x, box.y, box.w, box.h).adjusted(-2, -2, 2, 2)
            accent = QColor(self.palette.accent)
            painter.setPen(QPen(accent, 0, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            fill = QColor(accent)
            fill.setAlpha(46)
            painter.fillRect(rect, fill)


class PageCanvas(QGraphicsView):
    """Zoomable page view. Emits the character offset of a clicked word."""

    wordClicked = Signal(int, int)  # start, end character offsets
    wordDoubleClicked = Signal(int, int)
    zoomChanged = Signal(float)

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self.pixmap_item)

        self.overlay = _OverlayItem(palette)
        self.overlay.setZValue(10)
        self._scene.addItem(self.overlay)

        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(palette.surface_alt)))
        self.setMouseTracking(True)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._zoom = 1.0
        self._fit_mode = "width"
        self._boxes: list[WordBox] = []
        self._has_page = False

    # -- content -------------------------------------------------------------------------

    def set_page(self, image_path: str | None, boxes: list[WordBox]) -> None:
        self._boxes = boxes
        if image_path:
            pixmap = QPixmap(image_path)
            self._has_page = not pixmap.isNull()
        else:
            pixmap = QPixmap()
            self._has_page = False

        self.pixmap_item.setPixmap(pixmap)
        size = (pixmap.width(), pixmap.height()) if self._has_page else (0, 0)
        self.overlay.set_boxes(boxes if self._has_page else [], size)
        self._scene.setSceneRect(QRectF(0, 0, *size) if self._has_page else QRectF())
        if self._has_page:
            # Deferred: on the first page load the viewport has not been laid out yet, so
            # fitting immediately would compute the scale against a stale width.
            QTimer.singleShot(0, self.apply_fit)

    def set_image(self, image: QImage, boxes: list[WordBox] | None = None) -> None:
        pixmap = QPixmap.fromImage(image)
        self._has_page = not pixmap.isNull()
        self.pixmap_item.setPixmap(pixmap)
        self._boxes = boxes or []
        self.overlay.set_boxes(self._boxes, (pixmap.width(), pixmap.height()))
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.apply_fit()

    def clear(self) -> None:
        self.pixmap_item.setPixmap(QPixmap())
        self.overlay.set_boxes([], (0, 0))
        self._scene.setSceneRect(QRectF())
        self._has_page = False

    @property
    def has_page(self) -> bool:
        return self._has_page

    # -- overlays -------------------------------------------------------------------------

    def set_flagged(self, flagged: dict[int, str]) -> None:
        self.overlay.flagged = flagged
        self.overlay.update()

    def set_show_boxes(self, show: bool) -> None:
        self.overlay.show_boxes = show
        self.overlay.update()

    def set_show_heatmap(self, show: bool) -> None:
        self.overlay.show_heatmap = show
        self.overlay.update()

    def highlight_offset(self, offset: int, center: bool = True) -> None:
        """Highlight the word covering a character offset — the editor drives this."""
        index = next(
            (i for i, b in enumerate(self._boxes) if b.start <= offset < b.end),
            None,
        )
        if index is None:
            index = next((i for i, b in enumerate(self._boxes) if b.start >= offset), None)
        self.overlay.active_index = index
        self.overlay.update()

        if index is not None and center and self._has_page:
            box = self._boxes[index]
            self.ensure_visible(QRectF(box.x, box.y, box.w, box.h))

    def ensure_visible(self, rect: QRectF) -> None:
        """Scroll the rectangle into view, but leave the viewport alone if it already is.

        Unconditional centring makes the pane jump on every keystroke, which is
        disorienting when the editor is typing straight through a line.
        """
        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if viewport_rect.contains(rect):
            return
        self.ensureVisible(rect, 80, 120)

    # -- zoom -----------------------------------------------------------------------------

    def apply_fit(self) -> None:
        if not self._has_page:
            return
        if self._fit_mode == "page":
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()
        elif self._fit_mode == "width":
            width = self._scene.sceneRect().width()
            if width > 0:
                scale = (self.viewport().width() - 24) / width
                self.setTransform(self.transform().fromScale(scale, scale))
                self._zoom = scale
        self.zoomChanged.emit(self._zoom)

    def set_fit_mode(self, mode: str) -> None:
        self._fit_mode = mode
        self.apply_fit()

    def set_zoom(self, zoom: float) -> None:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        self._fit_mode = "manual"
        self.setTransform(self.transform().fromScale(zoom, zoom))
        self._zoom = zoom
        self.zoomChanged.emit(zoom)

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._zoom * factor)

    @property
    def zoom(self) -> float:
        return self._zoom

    # -- events ------------------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self._fit_mode = "manual"
                factor = 1.18 if delta > 0 else 1 / 1.18
                new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
                self.scale(new_zoom / self._zoom, new_zoom / self._zoom)
                self._zoom = new_zoom
                self.zoomChanged.emit(self._zoom)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode in ("width", "page"):
            self.apply_fit()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._has_page:
            index = self.overlay.index_at(self.mapToScene(event.pos()))
            if index != self.overlay.hover_index:
                self.overlay.hover_index = index
                self.overlay.update()
                if index is not None:
                    box = self._boxes[index]
                    self.setToolTip(f"{box.text}\nconfidence {box.conf:.0%}")
                else:
                    self.setToolTip("")
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._has_page:
            index = self.overlay.index_at(self.mapToScene(event.pos()))
            if index is not None:
                box = self._boxes[index]
                self.overlay.active_index = index
                self.overlay.update()
                if box.start >= 0:
                    self.wordClicked.emit(box.start, box.end)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._has_page:
            index = self.overlay.index_at(self.mapToScene(event.pos()))
            if index is not None:
                box = self._boxes[index]
                if box.start >= 0:
                    self.wordDoubleClicked.emit(box.start, box.end)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)
