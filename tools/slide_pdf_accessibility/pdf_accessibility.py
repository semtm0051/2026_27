"""Repair only the accessibility/text layers of native PowerPoint equations.

Usage:
  python pdf_accessibility.py extract --inputs a.pdf b.pdf --output mathml.json
  python pdf_accessibility.py apply --input a.pdf --speech-map speech.json --output fixed.pdf --report report.json

The speech-map is either {sha256: "English speech"} or a list of objects with
"hash" and "speech". Each key hashes the exact stripped embedded MathML UTF-8.
No equation is inferred from an old, potentially stale Figure/LaTeX alternative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, ContentStream, DictionaryObject, FloatObject, IndirectObject,
    NameObject, NumberObject, TextStringObject,
)

ITALIAN = re.compile(
    r"\b(?:aperta|aperto|chiusa|chiuso|pedice|apice|grassetto|corsivo|"
    r"uguale|diverso|fratto|sommatoria|equazione|rettangolo|inizia|termina|"
    r"graffa|tonda|quadra|matrice|maiuscola|minuscola|circonflesso|"
    r"moltiplicazione|quadrato|limite\s+(?:inferiore|superiore)|"
    r"più|meno|appartiene|riga\s+successiva|tre\s+puntini)\b", re.I,
)


def identity(obj):
    ref = obj if isinstance(obj, IndirectObject) else getattr(obj, "indirect_reference", None)
    return (ref.idnum, ref.generation) if ref else (id(obj), None)


def struct_elements(reader):
    """Visit the authoritative K tree once, without ParentTree duplicates."""
    seen = set()
    result = []

    def visit(value):
        key = identity(value)
        if key in seen:
            return
        seen.add(key)
        obj = value.get_object()
        if not isinstance(obj, dict):
            return
        if obj.get("/Type") == "/StructElem":
            result.append(obj)
        children = obj.get("/K", [])
        if not isinstance(children, list):
            children = [children]
        for child in children:
            if isinstance(child, (dict, IndirectObject)):
                visit(child)

    visit(reader.trailer["/Root"]["/StructTreeRoot"] if isinstance(reader, PdfReader)
          else reader.root_object["/StructTreeRoot"])
    return result


def mathml_for(element):
    attributes = element.get("/A")
    if attributes:
        attributes = attributes.get_object()
        attributes = attributes if isinstance(attributes, list) else [attributes]
        for value in attributes:
            obj = value.get_object()
            if isinstance(obj, dict) and obj.get("/MSFT_MathML"):
                return str(obj["/MSFT_MathML"]).strip()
    for file_spec in element.get("/AF", []):
        spec = file_spec.get_object()
        if "/EF" in spec:
            stream = spec["/EF"]["/F"]
            if stream.get("/Subtype") == "/application/mathml+xml":
                return stream.get_data().decode("utf-8").strip()
    return None


def key_for(mathml):
    return hashlib.sha256(mathml.encode("utf-8")).hexdigest()


def mcids_for(element):
    page = element.get("/Pg")
    result = []
    children = element.get("/K", [])
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if isinstance(child, int) and page is not None:
            result.append((identity(page), int(child)))
        elif isinstance(child, (dict, IndirectObject)):
            child = child.get_object()
            if isinstance(child, dict) and "/MCID" in child:
                p = child.get("/Pg", page)
                if p is not None:
                    result.append((identity(p), int(child["/MCID"])))
    return result


def extract(inputs, output):
    unique = {}
    file_reports = []
    for file in inputs:
        reader = PdfReader(file)
        page_map = {identity(p): i + 1 for i, p in enumerate(reader.pages)}
        count = 0
        for element in struct_elements(reader):
            if element.get("/S") != "/Formula":
                continue
            mathml = mathml_for(element)
            if not mathml:
                raise ValueError(f"Formula {identity(element)} has no MathML in {file}")
            count += 1
            key = key_for(mathml)
            entry = unique.setdefault(key, {"hash": key, "mathml": mathml, "original_alts": [], "occurrences": []})
            alt = str(element.get("/Alt", ""))
            if alt not in entry["original_alts"]:
                entry["original_alts"].append(alt)
            entry["occurrences"].append({
                "file": str(Path(file).resolve()),
                "page": page_map.get(identity(element.get("/Pg"))) if element.get("/Pg") else None,
                "object": identity(element)[0],
                "mcids": [mcid for _, mcid in mcids_for(element)],
            })
        file_reports.append({"file": str(Path(file).resolve()), "pages": len(reader.pages), "formulas": count})
    result = {"schema": 1, "hash_method": "sha256 of stripped MathML UTF-8", "files": file_reports,
              "equations": list(unique.values())}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"unique_mathml": len(unique), "files": file_reports, "output": str(output)}


def load_speech(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "equations" in data:
        data = data["equations"]
    if isinstance(data, list):
        data = {item["hash"]: item["speech"] for item in data}
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = value["speech"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty or invalid speech for {key}")
        if ITALIAN.search(value):
            raise ValueError(f"Italian remains in generated speech for {key}: {value}")
        result[key] = value.strip()
    return result


def ascii_speech(text):
    """Helvetica's ASCII characters avoid all embedded-subset font dependencies."""
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                       "–": "-", "—": "-", "−": "-", "\u00a0": " "}))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Non-ASCII speech needs explicit English wording: {text}") from exc
    return text


def dictionary_walk(root):
    seen = set()

    def visit(value):
        key = identity(value)
        if key in seen:
            return
        seen.add(key)
        value = value.get_object()
        if isinstance(value, dict):
            yield value
            for child in value.values():
                if isinstance(child, (dict, list, IndirectObject)):
                    yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list, IndirectObject)):
                    yield from visit(child)

    yield from visit(root)


def apply(input_path, speech_path, output, report_path):
    speech = load_speech(speech_path)
    reader = PdfReader(input_path)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    elements = struct_elements(writer)
    page_map = {identity(p): i + 1 for i, p in enumerate(writer.pages)}
    formula_map = {}
    formula_changes = []
    old_alt_lookup = defaultdict(set)
    for element in elements:
        if element.get("/S") != "/Formula":
            continue
        mathml = mathml_for(element)
        if mathml is None or key_for(mathml) not in speech:
            raise ValueError(f"No English MathML speech for Formula {identity(element)}")
        english = speech[key_for(mathml)]
        old = str(element.get("/Alt", ""))
        element[NameObject("/Alt")] = TextStringObject(english)
        element[NameObject("/Lang")] = TextStringObject("en-GB")
        for mcid in mcids_for(element):
            formula_map[mcid] = {"english": english, "old": old, "hash": key_for(mathml)}
        if old:
            old_alt_lookup[old].add(english)
        formula_changes.append({"object": identity(element)[0], "hash": key_for(mathml),
                                "mcids": [x[1] for x in mcids_for(element)]})

    figure_changes = []
    for element in elements:
        if element.get("/S") != "/Figure" or not element.get("/Alt"):
            continue
        old = str(element["/Alt"])
        if not ITALIAN.search(old):
            continue
        candidates = []
        for page_id, mcid in mcids_for(element):
            preceding = formula_map.get((page_id, mcid - 1))
            if preceding:
                candidates.append(preceding["english"])
        method = "preceding Formula MCID"
        if not candidates:
            matches = [(a, next(iter(values))) for a, values in old_alt_lookup.items()
                       if len(values) == 1 and len(a) > 12 and a in old]
            # Longest matches first: don't accidentally replace a subformula.
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            covered = []
            for alt, english in matches:
                if not any(alt in parent for parent in covered):
                    covered.append(alt)
                    candidates.append(english)
            method = "exact native Formula alternative"
        if candidates:
            english = "; ".join(dict.fromkeys(candidates))
            # Most first lines are obsolete LaTeX; a few are genuine English
            # author comments. Keep those comments alongside the native math.
            prefix = re.split(r"[\r\n]", old, maxsplit=1)[0].strip()
            retained_author_prefix = ""
            if prefix.startswith(("Example parameter column", "Hidden layer ")):
                retained_author_prefix = prefix
                english = prefix.rstrip(". ") + ". " + english
            element[NameObject("/Alt")] = TextStringObject(english)
            element[NameObject("/Lang")] = TextStringObject("en-GB")
            figure_changes.append({"object": identity(element)[0], "method": method, "old": old,
                                   "new": english, "retained_author_prefix": retained_author_prefix})

    font_name = NameObject("/CodexEquationEnglish")
    font_ref = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"), NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
    }))
    content_changes = []
    unmatched_hidden = []
    for page in writer.pages:
        resources = page["/Resources"]
        stream = ContentStream(page.get_contents(), writer)
        states = []
        state = {"ca": 1, "CA": 1, "Tr": 0}
        marks = []
        emitted = set()
        operations = []
        replacements = 0
        removals = 0
        for args, operator in stream.operations:
            if operator == b"q":
                states.append(state.copy())
            elif operator == b"Q":
                if not states:
                    raise ValueError("Unbalanced graphics-state stack")
                state = states.pop()
            elif operator == b"gs":
                ext = resources["/ExtGState"][args[0]]
                for key in ("ca", "CA"):
                    if "/" + key in ext:
                        state[key] = ext["/" + key]
            elif operator == b"Tr":
                state["Tr"] = int(args[0])
            elif operator in (b"BDC", b"BMC"):
                prop = args[1] if operator == b"BDC" else None
                if prop is not None and not isinstance(prop, dict):
                    prop = resources.get("/Properties", {}).get(prop)
                    prop = prop.get_object() if prop else None
                if isinstance(prop, dict) and "/Lang" in prop:
                    prop[NameObject("/Lang")] = TextStringObject("en-GB")
                marks.append(prop.get("/MCID") if isinstance(prop, dict) else None)
            elif operator == b"EMC":
                if not marks:
                    raise ValueError("Unbalanced marked-content stack")
                marks.pop()

            is_zeroalpha = state["ca"] == 0 and state["CA"] == 0
            if operator in (b"Tj", b"TJ", b"'", b'"') and is_zeroalpha:
                mcid = next((m for m in reversed(marks) if m is not None), None)
                key = (identity(page), int(mcid)) if mcid is not None else None
                formula = formula_map.get(key)
                if formula is not None:
                    if key not in emitted:
                        english = ascii_speech(formula["english"])
                        # q/Q preserves the original font and graphics state;
                        # both alphas remain zero throughout these operations.
                        operations.extend([
                            ([], b"q"), ([font_name, FloatObject(1.2)], b"Tf"),
                            ([TextStringObject(english)], b"Tj"), ([], b"Q"),
                        ])
                        emitted.add(key)
                        replacements += 1
                    else:
                        removals += 1
                    continue
                unmatched_hidden.append({"page": page_map[identity(page)], "mcid": mcid})
            operations.append((args, operator))
        if states or marks:
            raise ValueError("Unbalanced content stream")
        if replacements:
            if "/Font" not in resources:
                resources[NameObject("/Font")] = DictionaryObject()
            resources["/Font"][font_name] = font_ref
        stream.operations = operations
        # All pages are serialized to persist corrected marked-content language.
        page.replace_contents(stream)
        content_changes.append({"page": page_map[identity(page)], "english_text_runs": replacements,
                                "removed_extra_italian_text_runs": removals})

    language_count = 0
    for obj in dictionary_walk(writer.root_object):
        if "/Lang" in obj:
            obj[NameObject("/Lang")] = TextStringObject("en-GB")
            language_count += 1
    writer.root_object[NameObject("/Lang")] = TextStringObject("en-GB")
    unmatched_alt = []
    for obj in dictionary_walk(writer.root_object):
        for field in ("/Alt", "/ActualText"):
            if field in obj and ITALIAN.search(str(obj[field])):
                unmatched_alt.append({"object": identity(obj)[0], "field": field,
                                      "role": str(obj.get("/S", "")), "text": str(obj[field])})

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    reopened = PdfReader(output)
    formula_after = [o for o in struct_elements(reopened) if o.get("/S") == "/Formula"]
    if len(formula_after) != len(formula_changes):
        raise ValueError("Formula count changed after write")
    original_mathml = sorted(mathml_for(o) for o in struct_elements(reader) if o.get("/S") == "/Formula")
    if original_mathml != sorted(mathml_for(o) for o in formula_after):
        raise ValueError("Embedded MathML changed")
    text = "\n".join(page.extract_text() for page in reopened.pages)
    text_italian = [line for line in text.splitlines() if ITALIAN.search(line)]
    report = {"input": str(Path(input_path).resolve()), "output": str(output.resolve()),
              "pages": len(reopened.pages), "formulas_updated": len(formula_changes),
              "figure_alternatives_updated": len(figure_changes), "language_dictionaries_updated": language_count,
              "formula_changes": formula_changes, "figure_changes": figure_changes,
              "content_changes": content_changes, "unmatched_hidden_runs_preserved": unmatched_hidden,
              "unmatched_italian_alt": unmatched_alt, "unmatched_italian_extracted_lines": text_italian,
              "mathml_unchanged": True, "render_comparison_required": True}
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k not in ("figure_changes", "formula_changes", "content_changes")}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--inputs", nargs="+", required=True)
    ext.add_argument("--output", required=True)
    app = sub.add_parser("apply")
    app.add_argument("--input", required=True)
    app.add_argument("--speech-map", required=True)
    app.add_argument("--output", required=True)
    app.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(args.inputs, args.output)
    else:
        result = apply(args.input, args.speech_map, args.output, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
