// Convert outputs/REPORT_NOTES.md (headings, paragraphs, bullets, tables, images, code) to .docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, ImageRun, AlignmentType, LevelFormat, BorderStyle,
} = require("docx");

const [, , mdPath, outPath] = process.argv;
const dir = path.dirname(mdPath);
const lines = fs.readFileSync(mdPath, "utf8").split(/\r?\n/);

const FONT = "Calibri";
const CONTENT_W = 9026; // A4 width 11906 - 2 * 1440 margins (DXA)
const MAX_IMG_PX = 600;

function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: "Consolas", size: 19 }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function table(rows) {
  const cells = rows.map((r) => r.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()));
  const n = cells[0].length;
  // column widths proportional to the longest text in each column (min share), summing to CONTENT_W
  const len = [...Array(n)].map((_, j) => Math.max(6, ...cells.map((r) => (r[j] || "").replace(/[*`]/g, "").length)));
  const tot = len.reduce((a, b) => a + b, 0);
  const widths = len.map((l) => Math.floor((l / tot) * CONTENT_W));
  widths[n - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: cells.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: widths.map((w, j) => new TableCell({
        width: { size: w, type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "E7EEF7", color: "auto" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: runs(r[j] || "", { size: 19, bold: i === 0 ? true : undefined }) })],
      })),
    })),
  });
}

const body = [];
for (let i = 0; i < lines.length; i++) {
  const l = lines[i];
  if (!l.trim()) continue;
  if (l.startsWith("```")) {                       // code block
    const code = [];
    for (i++; i < lines.length && !lines[i].startsWith("```"); i++) code.push(lines[i]);
    code.forEach((c) => body.push(new Paragraph({
      spacing: { before: 0, after: 0 },
      shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
      children: [new TextRun({ text: c || " ", font: "Consolas", size: 18 })],
    })));
    body.push(new Paragraph({ children: [] }));
  } else if (l.startsWith("# ")) {
    body.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(l.slice(2)) }));
  } else if (l.startsWith("### ")) {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(l.slice(4)) }));
  } else if (l.startsWith("## ")) {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(l.slice(3)) }));
  } else if (l.startsWith("|")) {
    const rows = [];
    for (; i < lines.length && lines[i].startsWith("|"); i++) if (!/^\|[\s|:-]+\|$/.test(lines[i])) rows.push(lines[i]);
    i--;
    body.push(table(rows), new Paragraph({ children: [] }));
  } else if (/^!\[/.test(l)) {
    const [, alt, src] = l.match(/^!\[(.*)\]\((.*)\)/);
    const { w, h, data } = pngSize(path.join(dir, src));
    const s = Math.min(1, MAX_IMG_PX / w);
    body.push(new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(w * s), height: Math.round(h * s) },
        altText: { title: alt, description: alt, name: src } })],
    }));
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: alt, italics: true, size: 18, color: "595959" })] }));
  } else if (l.startsWith("- ") || l.startsWith("  ")) {
    // bullet; continuation lines (indented) are joined to the bullet text
    let t = l.replace(/^- /, "").trim();
    while (i + 1 < lines.length && lines[i + 1].startsWith("  ") && !lines[i + 1].trim().startsWith("- ")) t += " " + lines[++i].trim();
    body.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t) }));
  } else if (/^\d+\. /.test(l)) {
    // numbered item: keep its number, hanging indent; indented continuation lines are joined
    let t = l.trim();
    while (i + 1 < lines.length && /^ {2,}\S/.test(lines[i + 1]) && !/^\s*(- |\d+\. )/.test(lines[i + 1])) t += " " + lines[++i].trim();
    body.push(new Paragraph({ indent: { left: 360, hanging: 360 }, children: runs(t) }));
  } else if (l.startsWith(">")) {
    // block quote: join the quote's lines into one indented paragraph
    let t = l.replace(/^>\s?/, "").trim();
    while (i + 1 < lines.length && lines[i + 1].startsWith(">")) {
      const n = lines[++i].replace(/^>\s?/, "").trim();
      if (n) t += " " + n;
    }
    body.push(new Paragraph({ indent: { left: 567 }, shading: { type: ShadingType.CLEAR, fill: "EEF3F8" }, children: runs(t) }));
  } else {
    // plain paragraph: join following non-special lines
    let t = l.trim();
    while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|\||!\[|- |```|>|\d+\. )/.test(lines[i + 1])) t += " " + lines[++i].trim();
    body.push(new Paragraph({ children: runs(t) }));
  }
}

const doc = new Document({
  creator: "Renewable installation type project",
  title: "Predicting renewable installation type from geography",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { after: 120, line: 276 } } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 40, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 0, after: 240 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "2F5496" },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: body }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes"); });
