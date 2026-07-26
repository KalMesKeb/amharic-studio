"""The text pane: fidel editor with inline issue marking and a suggestion popover.

Two things here are what make correction fast rather than merely possible. The popover
puts ranked candidates one keystroke away (Alt+1…5) without moving the hands off the
keyboard or the eyes off the line. And the transliterator lets a corrector type fidel
phonetically, which matters because retyping a word is often quicker than hunting for it
in a candidate list.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.suggest import Issue, Suggestion
from ..core.translit import Transliterator
from . import theme


class SuggestionPopup(QFrame):
    """Frameless candidate list anchored under the flagged word."""

    accepted = Signal(object)  # Suggestion
    dismissed = Signal()

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.palette_ = palette
        self.issue: Issue | None = None

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background: {palette.surface}; border: 1px solid {palette.accent};"
            f" border-radius: 5px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(4)

        self.header = QLabel()
        self.header.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt; border: none;")
        layout.addWidget(self.header)

        self.list = QListWidget()
        self.list.setFont(theme.ethiopic_font(13))
        self.list.setStyleSheet("border: none;")
        self.list.setMaximumHeight(180)
        self.list.setMinimumWidth(280)
        self.list.itemActivated.connect(self._on_activated)
        layout.addWidget(self.list)

        self.hint = QLabel("Alt+1…5 accept · Esc dismiss · Alt+Enter accept top")
        self.hint.setStyleSheet(f"color: {palette.text_muted}; font-size: 7.5pt; border: none;")
        layout.addWidget(self.hint)

    def show_for(self, issue: Issue, global_pos) -> None:
        self.issue = issue
        self.header.setText(f"{issue.message} — “{issue.text}”")
        self.list.clear()

        for index, suggestion in enumerate(issue.suggestions[:5], start=1):
            item = QListWidgetItem(f"{index}.  {suggestion.text}")
            detail = suggestion.explanation or suggestion.source
            item.setToolTip(f"{detail} · confidence {suggestion.confidence:.0%}")
            item.setData(Qt.ItemDataRole.UserRole, suggestion)
            if not suggestion.verified:
                item.setForeground(QColor(self.palette_.text_muted))
            self.list.addItem(item)

        if not issue.suggestions:
            self.list.addItem(QListWidgetItem("No candidates — edit by hand"))

        self.list.setCurrentRow(0)
        self.adjustSize()
        self.move(global_pos)
        self.show()

    def _on_activated(self, item: QListWidgetItem) -> None:
        suggestion = item.data(Qt.ItemDataRole.UserRole)
        if suggestion is not None:
            self.accepted.emit(suggestion)
        self.hide()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.hide()
            self.dismissed.emit()
            return
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            index = key - Qt.Key.Key_1
            if index < self.list.count():
                self.list.setCurrentRow(index)
                self._on_activated(self.list.item(index))
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            item = self.list.currentItem()
            if item:
                self._on_activated(item)
            return
        super().keyPressEvent(event)


class AmharicTextEdit(QPlainTextEdit):
    """Fidel editor with issue underlining, canvas sync and phonetic input."""

    offsetChanged = Signal(int)
    issueAccepted = Signal(object, object)  # Issue, Suggestion
    editCommitted = Signal(str, str)  # before, after — for the ledger

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = palette
        self.issues: list[Issue] = []
        self.transliterator = Transliterator(enabled=False)
        self._font_size = 15
        self._current_issue_index = -1
        self._suppress_sync = False

        self.setFont(theme.ethiopic_font(self._font_size))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setTabChangesFocus(True)
        self.setCursorWidth(2)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Fidel need extra leading; Qt has no direct line-height setter on QPlainTextEdit,
        # so the document margin does the visual work instead.
        self.document().setDocumentMargin(14)

        self.popup = SuggestionPopup(palette, self)
        self.popup.accepted.connect(self._apply_suggestion)

        self.cursorPositionChanged.connect(self._on_cursor_moved)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # -- appearance -------------------------------------------------------------------------

    def set_font_size(self, size: int) -> None:
        self._font_size = max(8, min(48, size))
        self.setFont(theme.ethiopic_font(self._font_size))
        self.refresh_highlighting()

    @property
    def font_size(self) -> int:
        return self._font_size

    def set_transliteration(self, enabled: bool) -> None:
        self.transliterator.enabled = enabled
        self.transliterator.reset()

    # -- content ------------------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        self._suppress_sync = True
        self.setPlainText(text)
        self._suppress_sync = False
        self.transliterator.reset()

    def set_issues(self, issues: list[Issue]) -> None:
        self.issues = sorted(issues, key=lambda i: i.start)
        self._current_issue_index = -1
        self.refresh_highlighting()

    def issue_at(self, offset: int) -> Issue | None:
        for issue in self.issues:
            if issue.start <= offset < issue.end:
                return issue
        return None

    # -- highlighting ----------------------------------------------------------------------------

    def refresh_highlighting(self) -> None:
        # ExtraSelection lives on QTextEdit even for a QPlainTextEdit.
        selections: list[QTextEdit.ExtraSelection] = []
        document_length = len(self.toPlainText())

        for issue in self.issues:
            if issue.start >= document_length:
                continue
            selection = QTextEdit.ExtraSelection()
            fmt = QTextCharFormat()
            color = theme.severity_color(self.palette_, issue.severity.value)
            fmt.setUnderlineColor(color)
            fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SpellCheckUnderline)
            tint = QColor(color)
            tint.setAlpha(26)
            fmt.setBackground(tint)
            fmt.setToolTip(_issue_tooltip(issue))
            selection.format = fmt

            cursor = self.textCursor()
            cursor.setPosition(issue.start)
            cursor.setPosition(min(issue.end, document_length), QTextCursor.MoveMode.KeepAnchor)
            selection.cursor = cursor
            selections.append(selection)

        # Current-line tint, kept very subtle so it never competes with issue marking.
        line_selection = QTextEdit.ExtraSelection()
        line_format = QTextCharFormat()
        line_tint = QColor(self.palette_.accent)
        line_tint.setAlpha(14)
        line_format.setBackground(line_tint)
        line_format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        line_selection.format = line_format
        line_cursor = self.textCursor()
        line_cursor.clearSelection()
        line_selection.cursor = line_cursor
        selections.append(line_selection)

        self.setExtraSelections(selections)

    # -- navigation --------------------------------------------------------------------------------

    def goto_offset(self, start: int, end: int | None = None, select: bool = True) -> None:
        cursor = self.textCursor()
        cursor.setPosition(min(start, len(self.toPlainText())))
        if select and end is not None:
            cursor.setPosition(min(end, len(self.toPlainText())), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def goto_issue(self, index: int) -> Issue | None:
        if not self.issues:
            return None
        index = max(0, min(len(self.issues) - 1, index))
        self._current_issue_index = index
        issue = self.issues[index]
        self.goto_offset(issue.start, issue.end)
        self.show_suggestions_for(issue)
        return issue

    def next_issue(self) -> Issue | None:
        if not self.issues:
            return None
        offset = self.textCursor().position()
        for index, issue in enumerate(self.issues):
            if issue.start > offset:
                return self.goto_issue(index)
        return self.goto_issue(0)  # wrap

    def previous_issue(self) -> Issue | None:
        if not self.issues:
            return None
        offset = self.textCursor().selectionStart()
        for index in range(len(self.issues) - 1, -1, -1):
            if self.issues[index].start < offset:
                return self.goto_issue(index)
        return self.goto_issue(len(self.issues) - 1)

    # -- suggestions ----------------------------------------------------------------------------------

    def show_suggestions_for(self, issue: Issue | None = None) -> None:
        issue = issue or self.issue_at(self.textCursor().position())
        if issue is None:
            self.popup.hide()
            return
        cursor = self.textCursor()
        cursor.setPosition(issue.start)
        rect = self.cursorRect(cursor)
        point = self.viewport().mapToGlobal(rect.bottomLeft())
        point.setY(point.y() + 6)
        self.popup.show_for(issue, point)

    def _apply_suggestion(self, suggestion: Suggestion) -> None:
        issue = self.popup.issue
        if issue is None:
            return
        self.replace_span(issue.start, issue.end, suggestion.text)
        self.issueAccepted.emit(issue, suggestion)

    def replace_span(self, start: int, end: int, replacement: str) -> None:
        """Replace a range as one undoable edit, and report it for the ledger."""
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        cursor.setPosition(min(end, len(self.toPlainText())), QTextCursor.MoveMode.KeepAnchor)
        before = cursor.selectedText().replace("\u2029", "\n")
        cursor.insertText(replacement)
        cursor.endEditBlock()
        self.setTextCursor(cursor)
        if before != replacement:
            self.editCommitted.emit(before, replacement)

    def accept_top_suggestion(self, rank: int = 0) -> bool:
        issue = self.issue_at(self.textCursor().position())
        if issue is None or rank >= len(issue.suggestions):
            return False
        suggestion = issue.suggestions[rank]
        self.replace_span(issue.start, issue.end, suggestion.text)
        self.issueAccepted.emit(issue, suggestion)
        self.popup.hide()
        return True

    # -- events ----------------------------------------------------------------------------------------

    def _on_cursor_moved(self) -> None:
        self.transliterator.reset()
        self.refresh_highlighting()
        if not self._suppress_sync:
            self.offsetChanged.emit(self.textCursor().position())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        modifiers = event.modifiers()
        key = event.key()

        if modifiers & Qt.KeyboardModifier.AltModifier:
            if Qt.Key.Key_1 <= key <= Qt.Key.Key_5:
                if self.accept_top_suggestion(key - Qt.Key.Key_1):
                    event.accept()
                    return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.accept_top_suggestion(0):
                    event.accept()
                    return
            if key == Qt.Key.Key_Space:
                self.show_suggestions_for()
                event.accept()
                return

        if key == Qt.Key.Key_F8:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.previous_issue()
            else:
                self.next_issue()
            event.accept()
            return

        if self.transliterator.enabled and event.text() and not (
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
        ):
            emission = self.transliterator.feed(event.text())
            if emission.handled:
                cursor = self.textCursor()
                cursor.beginEditBlock()
                for _ in range(emission.delete):
                    cursor.deletePreviousChar()
                cursor.insertText(emission.insert)
                cursor.endEditBlock()
                event.accept()
                return

        super().keyPressEvent(event)

    def event(self, event: QEvent) -> bool:
        # Tooltips come from the ExtraSelection formats, which Qt surfaces automatically.
        return super().event(event)

    def _show_context_menu(self, point) -> None:
        cursor = self.cursorForPosition(point)
        offset = cursor.position()
        issue = self.issue_at(offset)

        menu = QMenu(self)
        if issue is not None:
            header = menu.addAction(f"“{issue.text}” — {issue.message}")
            header.setEnabled(False)
            menu.addSeparator()
            for index, suggestion in enumerate(issue.suggestions[:6], start=1):
                label = f"{suggestion.text}    ({suggestion.explanation or suggestion.source})"
                action = menu.addAction(label)
                action.setShortcut(f"Alt+{index}" if index <= 5 else "")
                action.triggered.connect(
                    lambda _checked=False, s=suggestion, i=issue: (
                        self.replace_span(i.start, i.end, s.text),
                        self.issueAccepted.emit(i, s),
                    )
                )
            if not issue.suggestions:
                none_action = menu.addAction("No candidates")
                none_action.setEnabled(False)
            menu.addSeparator()

        menu.addAction("Cut", self.cut)
        menu.addAction("Copy", self.copy)
        menu.addAction("Paste", self.paste)
        menu.addSeparator()
        menu.addAction("Select all", self.selectAll)
        menu.exec(self.viewport().mapToGlobal(point))


def _issue_tooltip(issue: Issue) -> str:
    lines = [f"{issue.message} ({issue.severity.value})"]
    for index, suggestion in enumerate(issue.suggestions[:5], start=1):
        detail = suggestion.explanation or suggestion.source
        lines.append(f"  Alt+{index}  {suggestion.text}   — {detail}")
    if not issue.suggestions:
        lines.append("  no candidates")
    return "\n".join(lines)


class EditorPane(QWidget):
    """The editor plus a compact status strip."""

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.editor = AmharicTextEdit(palette)
        layout.addWidget(self.editor, 1)

        strip = QWidget()
        strip.setStyleSheet(
            f"background: {palette.surface}; border-top: 1px solid {palette.border};"
        )
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(10, 4, 10, 4)

        self.info = QLabel("")
        self.info.setStyleSheet(f"color: {palette.text_muted}; font-size: 8pt;")
        strip_layout.addWidget(self.info)
        strip_layout.addStretch(1)

        self.ime_label = QLabel("")
        self.ime_label.setStyleSheet(f"color: {palette.accent}; font-size: 8pt; font-weight: 600;")
        strip_layout.addWidget(self.ime_label)

        layout.addWidget(strip)

    def set_status(self, text: str) -> None:
        self.info.setText(text)

    def set_ime_active(self, active: bool) -> None:
        self.ime_label.setText("ፊደል  SERA input" if active else "")
