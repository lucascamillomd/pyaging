// Interactive clock glossary: per-column dropdown filters, free-text search,
// click-to-sort headers (numeric-aware), and clickable DOIs. Columns are located
// by header name so the script keeps working if columns are added or reordered.
document.addEventListener('DOMContentLoaded', function () {
  const table = document.querySelector('.sortable.filterable');
  if (!table) return;

  // Map header label -> column index
  const headers = Array.from(table.querySelectorAll('thead th'));
  const colIndex = {};
  headers.forEach((th, i) => { colIndex[th.textContent.trim()] = i; });

  const NUMERIC_COLS = new Set(['N features', 'Year', 'Citations']);
  const FILTER_COLS = ['Data type', 'Species', 'Platform', 'Model type', 'Unit'];

  const bodyRows = () => Array.from(table.querySelectorAll('tbody tr'));

  // ---- wrap the (wide) table in a horizontal-scroll container ----
  const scroller = document.createElement('div');
  scroller.className = 'glossary-scroll';
  table.parentNode.insertBefore(scroller, table);
  scroller.appendChild(table);

  // ---- build filter UI ----
  const filtersDiv = document.createElement('div');
  filtersDiv.className = 'glossary-filters';
  const selects = {};

  FILTER_COLS.forEach(name => {
    if (!(name in colIndex)) return;
    const idx = colIndex[name];
    const values = [...new Set(bodyRows()
      .map(r => r.children[idx] ? r.children[idx].textContent.trim() : '')
      .filter(v => v !== ''))].sort((a, b) => a.localeCompare(b));
    const wrap = document.createElement('div');
    wrap.className = 'glossary-filter';
    const label = document.createElement('label');
    label.textContent = name + ':';
    const select = document.createElement('select');
    const all = document.createElement('option');
    all.value = ''; all.textContent = 'All';
    select.appendChild(all);
    values.forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.textContent = v.length > 40 ? v.slice(0, 38) + '…' : v;
      select.appendChild(o);
    });
    select.addEventListener('change', applyFilters);
    wrap.appendChild(label); wrap.appendChild(select);
    filtersDiv.appendChild(wrap);
    selects[name] = { idx, select };
  });

  // search box
  const searchWrap = document.createElement('div');
  searchWrap.className = 'glossary-filter';
  const searchLabel = document.createElement('label');
  searchLabel.textContent = 'Search:';
  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = 'Type to search…';
  searchInput.addEventListener('input', applyFilters);
  searchWrap.appendChild(searchLabel); searchWrap.appendChild(searchInput);
  filtersDiv.appendChild(searchWrap);

  // reset
  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset';
  resetBtn.className = 'glossary-reset';
  resetBtn.addEventListener('click', () => {
    Object.values(selects).forEach(s => { s.select.value = ''; });
    searchInput.value = '';
    applyFilters();
  });
  filtersDiv.appendChild(resetBtn);

  const countDiv = document.createElement('div');
  countDiv.className = 'glossary-count';
  filtersDiv.appendChild(countDiv);

  scroller.parentNode.insertBefore(filtersDiv, scroller);

  // ---- linkify DOIs ----
  if ('DOI' in colIndex) {
    const idx = colIndex['DOI'];
    bodyRows().forEach(r => {
      const cell = r.children[idx];
      if (!cell) return;
      const t = cell.textContent.trim();
      if (t.startsWith('http')) {
        cell.innerHTML = `<a href="${t}" target="_blank" rel="noopener">link</a>`;
      }
    });
  }

  // ---- sortable headers ----
  headers.forEach((header, index) => {
    header.style.cursor = 'pointer';
    header.title = 'Click to sort';
    header.addEventListener('click', () => sortTable(index, NUMERIC_COLS.has(header.textContent.trim())));
  });

  function sortTable(columnIndex, numeric) {
    const th = headers[columnIndex];
    const current = th.getAttribute('aria-sort');
    headers.forEach(h => h.removeAttribute('aria-sort'));
    const dir = current === 'ascending' ? 'descending' : 'ascending';
    th.setAttribute('aria-sort', dir);
    const tbody = table.querySelector('tbody');
    const rows = bodyRows();
    rows.sort((a, b) => {
      const at = a.children[columnIndex].textContent.trim();
      const bt = b.children[columnIndex].textContent.trim();
      if (numeric) {
        const an = parseFloat(at.replace(/[^0-9.eE+-]/g, ''));
        const bn = parseFloat(bt.replace(/[^0-9.eE+-]/g, ''));
        const av = isNaN(an) ? -Infinity : an;
        const bv = isNaN(bn) ? -Infinity : bn;
        return dir === 'ascending' ? av - bv : bv - av;
      }
      return dir === 'ascending' ? at.localeCompare(bt) : bt.localeCompare(at);
    });
    rows.forEach(r => tbody.appendChild(r));
  }

  // ---- filtering ----
  function applyFilters() {
    const search = searchInput.value.toLowerCase();
    let shown = 0;
    const rows = bodyRows();
    rows.forEach(row => {
      let ok = true;
      for (const name in selects) {
        const { idx, select } = selects[name];
        if (select.value !== '') {
          const cell = row.children[idx];
          if (!cell || cell.textContent.trim() !== select.value) { ok = false; break; }
        }
      }
      if (ok && search !== '' && !row.textContent.toLowerCase().includes(search)) ok = false;
      row.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });
    countDiv.textContent = `${shown} / ${rows.length} clocks`;
  }

  applyFilters();
});
