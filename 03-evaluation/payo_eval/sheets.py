"""
Excel sheets for the team: post text to read, label columns with dropdowns, and strict read-back.

Dropdown lists live on a hidden `lists` sheet, so they are not limited by Excel's 255-character
inline list. Reading back never trusts the dropdowns: every value is validated again in code.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

MAIN_SHEET = "labels"
GUIDE_SHEET = "how_to_annotate"
GUIDE_WIDTHS = (20, 18, 70, 60, 60)
HEADING_FILL = PatternFill("solid", fgColor="DDE7F3")
TEXT_WIDTH = 70
INPUT_FILL = PatternFill("solid", fgColor="FFF7D6")  # pale yellow marks the columns people fill in
HEADER_FONT = Font(bold=True)


@dataclass(frozen=True)
class Column:
    """One column the person fills in, and what it may contain."""

    name: str
    choices: Sequence[str] = ()          # allowed values (dropdown); empty = free text or numeric
    int_range: tuple[int, int] | None = None
    multi: bool = False                  # several choices separated by ";" (e.g. harm flags)
    required: bool = True                # an empty cell is an error (unless `skip_if` applies)
    skip_if: Mapping[str, str] = field(default_factory=dict)  # e.g. {"exclude_flag": "yes"}: not needed then


def write_sheet(path: Path, frame: pd.DataFrame, inputs: Sequence[Column], text_columns: Sequence[str] = (),
                prefill: pd.DataFrame | None = None, overwrite: bool = False,
                guide: Sequence[Sequence[str]] | None = None) -> Path:
    """
    Write one person's sheet: read-only columns from `frame`, then the input columns.

    Args:
        path: Target .xlsx.
        frame: Read-only columns (ids and text), one row per item.
        inputs: Columns to fill in, with their dropdowns.
        text_columns: Long-text columns to wrap and widen.
        prefill: Optional values for input columns (same index as `frame`).
        overwrite: Replace an existing sheet. Off by default so no one's work is lost.
        guide: Optional instructions, written to a first tab that opens by default. One list of cells per
            row; a row whose first cell starts with "# " is a heading.

    Returns:
        The path written.
    """
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists; it may hold someone's work. Delete it to regenerate.")
    path.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    sheet = book.active
    sheet.title = MAIN_SHEET
    lists = book.create_sheet("lists")
    lists.sheet_state = "hidden"

    headers = list(frame.columns) + [c.name for c in inputs]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = HEADER_FONT
    for index, row in frame.reset_index(drop=True).iterrows():
        values = ["" if pd.isna(v) else v for v in row.tolist()]
        filled = []
        for column in inputs:
            value = ""
            if prefill is not None and column.name in prefill.columns:
                cell = prefill.reset_index(drop=True).loc[index, column.name]
                value = "" if pd.isna(cell) else cell
            filled.append(value)
        sheet.append(values + filled)

    n_rows = len(frame) + 1
    for position, name in enumerate(headers, start=1):
        letter = get_column_letter(position)
        is_text = name in text_columns
        sheet.column_dimensions[letter].width = TEXT_WIDTH if is_text else max(12, len(name) + 2)
        for row in range(2, n_rows + 1):
            cell = sheet[f"{letter}{row}"]
            cell.alignment = Alignment(wrap_text=is_text, vertical="top")
    for offset, column in enumerate(inputs):
        position = len(frame.columns) + 1 + offset
        letter = get_column_letter(position)
        for row in range(2, n_rows + 1):
            sheet[f"{letter}{row}"].fill = INPUT_FILL
        if column.choices and not column.multi and n_rows > 1:
            list_col = get_column_letter(offset + 1)
            for i, choice in enumerate(column.choices, start=1):
                lists[f"{list_col}{i}"] = choice
            rule = DataValidation(type="list", formula1=f"=lists!${list_col}$1:${list_col}${len(column.choices)}",
                                  allow_blank=True)
            rule.error, rule.errorTitle = f"Pick one of: {', '.join(column.choices)}", column.name
            sheet.add_data_validation(rule)
            rule.add(f"{letter}2:{letter}{n_rows}")
        elif column.int_range and n_rows > 1:
            low, high = column.int_range
            rule = DataValidation(type="whole", operator="between", formula1=str(low), formula2=str(high),
                                  allow_blank=True)
            rule.error, rule.errorTitle = f"Whole number {low}-{high}", column.name
            sheet.add_data_validation(rule)
            rule.add(f"{letter}2:{letter}{n_rows}")
    sheet.freeze_panes = "B2"
    if guide:
        _write_guide(book, guide)
    book.save(path)
    return path


def _write_guide(book: Workbook, rows: Sequence[Sequence[str]]) -> None:
    """A read-only instructions tab, first in the workbook and active when the file opens."""
    tab = book.create_sheet(GUIDE_SHEET, 0)
    for row in rows:
        heading = bool(row) and str(row[0]).startswith("# ")
        cells = [str(row[0])[2:], *row[1:]] if heading else list(row)
        tab.append(cells)
        for cell in tab[tab.max_row]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if heading:
                cell.font, cell.fill = HEADER_FONT, HEADING_FILL
    for position, width in enumerate(GUIDE_WIDTHS, start=1):
        tab.column_dimensions[get_column_letter(position)].width = width
    tab.protection.sheet = True  # read-only, so nobody types labels here by mistake
    book.active = 0


def read_sheet(path: Path) -> pd.DataFrame:
    """
    Read a sheet back as trimmed strings ("" for empty cells).

    Args:
        path: An .xlsx written by write_sheet (or the same layout).

    Returns:
        All columns as strings.
    """
    if not path.exists():
        raise SystemExit(f"Missing sheet: {path}")
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book[MAIN_SHEET] if MAIN_SHEET in book.sheetnames else book.worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    book.close()
    if not rows:
        return pd.DataFrame()
    header = [str(h) for h in rows[0]]
    body = [["" if v is None else _cell_text(v) for v in row] for row in rows[1:]
            if any(v not in (None, "") for v in row)]
    return pd.DataFrame(body, columns=header)


def _cell_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def validate(frame: pd.DataFrame, inputs: Sequence[Column], id_column: str, source: str,
             allow_blank_rows: bool = False) -> list[str]:
    """
    Check every input cell against its column rule.

    Args:
        frame: From read_sheet.
        inputs: The column rules used to write the sheet.
        id_column: Column that identifies a row in messages (e.g. post_id, item_id).
        source: Name used in messages (e.g. the person's id).
        allow_blank_rows: Treat rows with every input empty as "not done yet" instead of errors.

    Returns:
        Human-readable problems; empty when the sheet is valid.
    """
    problems = []
    missing = [c.name for c in inputs if c.name not in frame.columns]
    if missing:
        return [f"{source}: columns missing from the sheet: {', '.join(missing)}"]
    for _, row in frame.iterrows():
        values = {c.name: str(row[c.name]).strip() for c in inputs}
        if allow_blank_rows and not any(values.values()):
            continue
        for column in inputs:
            value = values[column.name]
            skipped = any(str(row.get(k, "")).strip().lower() == v for k, v in column.skip_if.items())
            where = f"{source}, {id_column} {row[id_column]}, {column.name}"
            if not value:
                if column.required and not skipped:
                    problems.append(f"{where}: empty")
                continue
            if column.choices:
                parts = [p.strip().lower() for p in value.split(";")] if column.multi else [value.lower()]
                allowed = {c.lower() for c in column.choices}
                bad = [p for p in parts if p not in allowed]
                if bad:
                    problems.append(f"{where}: {', '.join(bad)!r} not one of {', '.join(column.choices)}")
            elif column.int_range:
                low, high = column.int_range
                if not value.isdigit() or not low <= int(value) <= high:
                    problems.append(f"{where}: {value!r} is not a whole number {low}-{high}")
    return problems


def normalize(frame: pd.DataFrame, inputs: Sequence[Column]) -> pd.DataFrame:
    """Lowercase choice columns and join multi-choice values with ';' (after validate passed)."""
    out = frame.copy()
    for column in inputs:
        if column.choices and column.name in out:
            if column.multi:
                out[column.name] = out[column.name].map(
                    lambda v: ";".join(sorted(p.strip().lower() for p in v.split(";") if p.strip())))
            else:
                out[column.name] = out[column.name].str.lower()
    return out
