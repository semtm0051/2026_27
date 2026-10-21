"""Normalize a PowerPoint PDF's equation accessibility and verify its appearance."""
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import shutil
import subprocess
from PIL import Image, ImageChops
from pdf_accessibility import extract, apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--node', default='node')
    parser.add_argument('--pdftoppm', default='pdftoppm')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_file():
        parser.error('Input PDF does not exist.')
    if source == output or output.exists():
        parser.error('Use a new output filename; the input and existing files are preserved.')
    script = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='slide-pdf-accessibility-',dir=output.parent) as temp:
        tmp = Path(temp).resolve()
        assert tmp.parent == output.parent, "Unexpected temporary directory"
        extract([str(source)], tmp/'mathml.json')
        subprocess.run([args.node,str(script/'generate_speech.mjs'),str(tmp/'mathml.json'),str(tmp/'speech.json')],check=True)
        report = apply(source,tmp/'speech.json',tmp/'fixed.pdf',tmp/'report.json')
        if report['unmatched_italian_alt'] or report['unmatched_italian_extracted_lines']:
            raise RuntimeError('Unmatched Italian text remains; the output was not published.')
        def render(job):
            name,pdf=job
            folder=tmp/name;folder.mkdir()
            subprocess.run([args.pdftoppm,'-scale-to','1200','-png',str(pdf),str(folder/'page')],check=True,capture_output=True)
            return sorted(folder.glob('page-*.png'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            before,after=pool.map(render,[('before',source),('after',tmp/'fixed.pdf')])
        if len(before)!=len(after):
            raise RuntimeError('Page count changed; output withheld.')
        for number,(a,b) in enumerate(zip(before,after),1):
            with Image.open(a) as ia,Image.open(b) as ib:
                if ia.size!=ib.size or ImageChops.difference(ia.convert('RGB'),ib.convert('RGB')).getbbox():
                    raise RuntimeError(f'Page {number} changed visually; output withheld.')
        output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(tmp/'fixed.pdf',output)
        print(f'Saved {output}: {len(after)} visually identical pages, English equation descriptions, original MathML preserved.')


if __name__=='__main__':
    main()

