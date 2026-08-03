function splitTableRow(line) {
  let source = line.trim();
  if (source.startsWith("|")) source = source.slice(1);
  if (source.endsWith("|") && !source.endsWith("\\|")) source = source.slice(0, -1);

  const cells = [];
  let cell = "";
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === "\\" && source[index + 1] === "|") {
      cell += "|";
      index += 1;
    } else if (char === "|") {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell.trim());
  return cells;
}

function tableAlignment(marker) {
  const trimmed = marker.trim();
  if (!/^:?-{3,}:?$/.test(trimmed)) return null;
  if (trimmed.startsWith(":") && trimmed.endsWith(":")) return "center";
  if (trimmed.endsWith(":")) return "right";
  return "left";
}

function parseTableAt(lines, start) {
  if (start + 1 >= lines.length || !lines[start].includes("|")) return null;
  const headers = splitTableRow(lines[start]);
  const markers = splitTableRow(lines[start + 1]);
  const alignments = markers.map(tableAlignment);
  if (
    headers.length < 2
    || markers.length !== headers.length
    || alignments.includes(null)
  ) {
    return null;
  }

  const rows = [];
  let end = start + 2;
  while (end < lines.length && lines[end].includes("|")) {
    const cells = splitTableRow(lines[end]);
    if (cells.length < 2) break;
    rows.push(headers.map((_, index) => cells[index] || ""));
    end += 1;
  }
  return { headers, alignments, rows, end };
}

function appendMarkdownInline(parent, text) {
  const strongPattern = /\*\*(.+?)\*\*/g;
  let offset = 0;
  for (const match of text.matchAll(strongPattern)) {
    parent.appendChild(document.createTextNode(text.slice(offset, match.index)));
    const strong = document.createElement("strong");
    strong.textContent = match[1];
    parent.appendChild(strong);
    offset = match.index + match[0].length;
  }
  parent.appendChild(document.createTextNode(text.slice(offset)));
}

function makeMarkdownTable(parsed) {
  const wrapper = document.createElement("div");
  wrapper.className = "markdown-table-wrap";
  const table = document.createElement("table");
  table.className = "markdown-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");

  parsed.headers.forEach((text, index) => {
    const th = document.createElement("th");
    th.style.textAlign = parsed.alignments[index];
    appendMarkdownInline(th, text);
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  if (parsed.rows.length) {
    const tbody = document.createElement("tbody");
    for (const row of parsed.rows) {
      const tr = document.createElement("tr");
      row.forEach((text, index) => {
        const td = document.createElement("td");
        td.style.textAlign = parsed.alignments[index];
        appendMarkdownInline(td, text);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
  }

  wrapper.appendChild(table);
  return wrapper;
}

export function makeAssistantTextElement(text) {
  const div = document.createElement("div");
  div.className = "ev text";
  const lines = (text || "").split(/\r?\n/);
  let plainStart = 0;
  let index = 0;

  while (index < lines.length) {
    const table = parseTableAt(lines, index);
    if (!table) {
      index += 1;
      continue;
    }
    if (plainStart < index) {
      appendMarkdownInline(div, lines.slice(plainStart, index).join("\n"));
    }
    div.appendChild(makeMarkdownTable(table));
    index = table.end;
    plainStart = index;
  }
  if (plainStart < lines.length) {
    appendMarkdownInline(div, lines.slice(plainStart).join("\n"));
  }
  return div;
}
