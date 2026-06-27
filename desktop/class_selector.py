"""Searchable COCO class selector for the demo.

A compact widget: a search box that filters the 80 COCO class names, a checkable
list to pick which classes the Vision Assistant should report, and All / None /
ISR-preset shortcuts. ``selected_classes()`` returns the chosen names; an empty
selection means "detect everything" (no filter).
"""
from __future__ import annotations

from typing import Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from src.detection.coco_classes import COCO_CLASSES, ISR_PRESET


class ClassSelector(QWidget):
    changed = Signal()

    def __init__(self, parent=None, initial: Optional[Set[str]] = None) -> None:
        super().__init__(parent)
        initial = initial or set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search classes to detect…")
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)

        self.list = QListWidget()
        self.list.setMaximumHeight(170)
        for name in COCO_CLASSES:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in initial else Qt.Unchecked)
            self.list.addItem(item)
        self.list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.list)

        btns = QHBoxLayout()
        for label, fn in (("All", self._select_all), ("None", self._select_none),
                          ("ISR preset", self._select_isr)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)

        self.count = QLabel()
        lay.addWidget(self.count)
        self._update_count()

    # -- public API -----------------------------------------------------
    def selected_classes(self) -> Optional[Set[str]]:
        """Checked class names, or ``None`` when nothing is selected (= all)."""
        picked = {self.list.item(i).text()
                  for i in range(self.list.count())
                  if self.list.item(i).checkState() == Qt.Checked}
        return picked or None

    # -- internals ------------------------------------------------------
    def _filter(self, text: str) -> None:
        t = text.strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def _set_all(self, state) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)
        self.list.blockSignals(False)
        self._update_count()
        self.changed.emit()

    def _select_all(self) -> None:
        self._set_all(Qt.Checked)

    def _select_none(self) -> None:
        self._set_all(Qt.Unchecked)

    def _select_isr(self) -> None:
        preset = set(ISR_PRESET)
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.text() in preset else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_count()
        self.changed.emit()

    def _on_item_changed(self, _item) -> None:
        self._update_count()
        self.changed.emit()

    def _update_count(self) -> None:
        sel = self.selected_classes()
        self.count.setText("All classes (nothing selected)" if sel is None
                           else f"{len(sel)} class(es) selected")
