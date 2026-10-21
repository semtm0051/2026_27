# English equation descriptions in slide PDFs

PowerPoint can generate Italian equation descriptions in its PDF export even when the slide text is English. These descriptions are invisible, but appear in text extraction and accessibility tags. Changing the presentation's proofing language alone does not reliably prevent this.

After exporting a new PDF, run this tool before publishing it. It derives English descriptions from the PDF's embedded MathML, updates the invisible equation text and accessibility alternatives, and sets the document language to British English. It preserves the native visible equations, links, tags and MathML supplements. It then renders every page before and after the repair and refuses to deliver the output if any pixels differ at 1200-pixel resolution.

The PowerPoint files do not need to be rebuilt or flattened. This tool is intended for tagged PowerPoint PDFs with embedded MathML, as used in Weeks 3–5.

## Setup

Install Python 3.10 or later, Node.js 18 or later, and Poppler (`pdftoppm`). In this directory:

```sh
python -m pip install -r requirements.txt
npm install --ignore-scripts
```

## Repair an export

```sh
python fix_export.py "new-export.pdf" "checked-export.pdf"
```

Use a new output filename. The input is never overwritten. If Node.js or Poppler is not on your PATH, pass `--node /path/to/node` and `--pdftoppm /path/to/pdftoppm`.

The tool checks for residual Italian descriptions and requires a pixel-identical render before writing the requested output. Review the result and publish it in place of the new export. This repairs the identified equation-language issue; it is not a complete PDF accessibility certification.

English maths speech is generated locally using [Speech Rule Engine](https://github.com/Speech-Rule-Engine/speech-rule-engine), pinned to version 4.1.4. No slide contents are sent to an external service.
