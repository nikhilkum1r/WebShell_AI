/* ═══════════════════════════════════════════════════════════
   UI & DOM Helpers
   ═══════════════════════════════════════════════════════════ */

let scans = 0, threats = 0, curTab = 'text';
const MIN_CODE_LEN = 20; // Minimum characters for code analysis

const id = s => document.getElementById(s);
const show = s => id(s).classList.remove('gone');
const hide = s => id(s).classList.add('gone');

/* ── Code panel helpers ──────────────────────────────────────────── */
function onCodeInput(ta) {
    const len = ta.value.trim().length;
    const lines = ta.value.split('\n').length;
    const counter = id('code-counter');
    const hint = id('code-hint');
    if (counter) counter.textContent = `${len} chars · ${lines} line${lines !== 1 ? 's' : ''}`;
    if (hint) {
        if (len === 0) {
            hint.textContent = '';
            hint.className = 'code-hint';
        } else if (len < MIN_CODE_LEN) {
            hint.textContent = `⚠ Min ${MIN_CODE_LEN} chars required (${MIN_CODE_LEN - len} more)`;
            hint.className = 'code-hint';
            ta.classList.add('input-error');
        } else {
            hint.textContent = `✓ Ready to analyse`;
            hint.className = 'code-hint ok';
            ta.classList.remove('input-error');
        }
    }
}

function setLangHint(el, lang) {
    document.querySelectorAll('.lang-tag').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    const ta = id('code-input');
    if (ta) ta.setAttribute('data-lang', lang);
}

function clearCode() {
    const ta = id('code-input');
    if (ta) { ta.value = ''; onCodeInput(ta); ta.focus(); }
}

async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        const ta = id('code-input');
        if (ta) { ta.value = text; onCodeInput(ta); ta.focus(); }
    } catch {
        toast('Clipboard permission denied. Paste manually (Ctrl+V).', 'err');
    }
}

const fmt_bytes = b => b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)} MB`
    : b > 1024 ? `${(b / 1024).toFixed(1)} KB`
        : `${b} B`;

function tab(name) {
    curTab = name;
    hide('result-box'); hide('features-box'); hide('votes-box');
    id('bulk-table-wrap').style.display = 'none';
    ['text', 'file', 'bulk'].forEach(t => {
        id(`tab-${t}`).classList.toggle('on', t === name);
        id(`panel-${t}`).classList.toggle('on', t === name);
    });
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

function dzOver(e, dzId) { e.preventDefault(); id(dzId).classList.add('drag-over'); }
function dzLeave(dzId) { id(dzId).classList.remove('drag-over'); }
function dzDrop(e, inputId) {
    e.preventDefault(); e.currentTarget.classList.remove('drag-over');
    const inp = id(inputId);
    inp.files = e.dataTransfer.files; inp.dispatchEvent(new Event('change'));
}
function dzChange(inp, nameId) {
    const files = [...inp.files];
    id(nameId).textContent = files.length > 1 ? `${files.length} files selected` : (files[0]?.name ?? '');
}

function setLoading(on) {
    id('spinner').style.display = on ? 'block' : 'none';
    id('go-icon').style.display = on ? 'none' : 'inline';
    id('go-txt').textContent = on ? 'Scanning…' : 'Analyse Now';
    id('go-btn').disabled = on;
}

function toast(msg, type = 'suc') {
    const c = id('toasts');
    const div = document.createElement('div');
    div.className = `toast ${type}`;
    const col = type === 'err' ? '#f87171' : '#34d399';
    div.innerHTML = `
<svg xmlns='http://www.w3.org/2000/svg' width='17' height='17' viewBox='0 0 24 24' fill='none'
     stroke='${col}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>
  ${type === 'err'
            ? "<circle cx='12' cy='12' r='10'/><line x1='15' y1='9' x2='9' y2='15'/><line x1='9' y1='9' x2='15' y2='15'/>"
            : "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><polyline points='9 12 11 14 15 10'/>"}
</svg> ${msg}`;
    c.appendChild(div);
    setTimeout(() => div.remove(), 4500);
}

function showResult(data, src) {
    const ws = data.is_webshell;
    const banner = id('banner');
    banner.className = `banner ${ws ? 'ws' : 'ok'}`;
    id('cbar').className = `cbar-wrap ${ws ? 'ws' : 'ok'}`;

    id('b-icon').innerHTML = ws
        ? `<svg xmlns='http://www.w3.org/2000/svg' width='27' height='27' viewBox='0 0 24 24' fill='none' stroke='#f87171' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><polygon points='7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2'/><line x1='15' y1='9' x2='9' y2='15'/><line x1='9' y1='9' x2='15' y2='15'/></svg>`
        : `<svg xmlns='http://www.w3.org/2000/svg' width='27' height='27' viewBox='0 0 24 24' fill='none' stroke='#34d399' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><polyline points='9 12 11 14 15 10'/></svg>`;

    id('b-label').textContent = ws ? '⚠ Webshell Detected' : '✓ Normal Content';
    id('b-sub').textContent = `Model: ${data.model} | Raw: ${data.raw_score} | Source: ${src}`;
    id('cbar-pct').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    setTimeout(() => { id('cbar-fill').style.width = `${data.confidence * 100}%`; }, 50);
    show('result-box');

    if (data.votes?.length) {
        const vl = id('votes-list'); vl.innerHTML = '';
        data.votes.forEach(v => {
            const div = document.createElement('div');
            div.className = 'vote-row';
            const col = v.is_webshell ? '#ef4444' : '#10b981';
            div.innerHTML = `
<span class="vote-model">${v.model}</span>
<div class="vote-bar-wrap">
  <div class="vote-bar-track">
    <div class="vote-bar-fill" style="width:${v.confidence * 100}%;background:${col}"></div>
  </div>
</div>
<span class="vote-pct" style="color:${col}">${(v.confidence * 100).toFixed(1)}%</span>
<span class="vtag ${v.is_webshell ? 'ws' : 'ok'}" style="margin-left:10px">${v.is_webshell ? 'Webshell' : 'Normal'}</span>
`;
            vl.appendChild(div);
        });
        show('votes-box');
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    } else {
        hide('votes-box');
    }

    if (data.features) showFeatures(data.features);
    if (data.explanation) {
        id('exp-text').innerHTML = data.explanation.replace(/`(.*?)`/g, '<code>$1</code>');
        show('explanation-box');
    } else {
        hide('explanation-box');
    }

    // Handle code highlighting
    if (curTab === 'text') {
        const text = id('code-input').value;
        annotateCode(text, data.features?.dangerous_funcs_found || []);
        show('code-box');
    } else {
        hide('code-box');
    }

    addHistory(src, data);
    updateStats([data]);
}

function annotateCode(text, dangerousFuncs) {
    const codeView = id('code-view');
    // Escape HTML
    let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Highlight dangerous functions
    dangerousFuncs.forEach(fn => {
        const regex = new RegExp(`\\b${fn}\\b`, 'gi');
        html = html.replace(regex, `<span class="hl-danger">$&</span>`);
    });

    // Simple common patterns like eval if not in find
    const common = ['eval', 'base64_decode', 'system', 'exec', 'shell_exec', 'passthru'];
    common.forEach(fn => {
        const regex = new RegExp(`\\b${fn}\\b`, 'gi');
        if (!html.includes('hl-danger') || !dangerousFuncs.includes(fn)) {
            html = html.replace(regex, `<span class="hl-danger">$&</span>`);
        }
    });

    codeView.innerHTML = html;
}

function toggleCode() {
    const wrap = id('code-pre-wrap');
    const btn = id('code-toggle-btn');
    if (wrap.classList.contains('gone')) {
        wrap.classList.remove('gone');
        btn.textContent = 'Hide Annotated Code';
    } else {
        wrap.classList.add('gone');
        btn.textContent = 'Show Highlighted Code';
    }
}

function showFeatures(f) {
    const chips = id('feat-chips');
    chips.innerHTML = '';
    const add = (lbl, val, cls = '') => {
        const d = document.createElement('div');
        d.className = 'feat-chip';
        d.innerHTML = `<span class="fc-label">${lbl}</span><span class="fc-val ${cls}">${val}</span>`;
        chips.appendChild(d);
    };
    add('Entropy', f.entropy.toFixed(3) + ' bits', `risk-${f.entropy_risk[0]}`);
    add('Entropy Risk', f.entropy_risk, `risk-${f.entropy_risk[0]}`);
    add('File Size', fmt_bytes(f.file_size_bytes));
    add('Lines', f.line_count.toLocaleString());
    add('Danger Funcs', f.dangerous_func_count, f.dangerous_func_count > 0 ? 'risk-H' : 'risk-L');
    add('Heuristic Risk', f.heuristic_risk.toFixed(1) + '/100', `risk-${f.risk_label[0]}`);

    const dfw = id('dfuncs-wrap');
    const dfl = id('dfuncs-list');
    if (f.dangerous_funcs_found?.length) {
        dfl.innerHTML = f.dangerous_funcs_found.map(fn => `<span class="func-tag">${fn}</span>`).join('');
        dfw.style.display = 'block';
    } else {
        dfl.innerHTML = '<span class="no-funcs">None detected.</span>';
        dfw.style.display = 'block';
    }
    show('features-box');
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

function renderBulk(results) {
    const tb = id('bulk-tbody'); tb.innerHTML = '';
    results.forEach((item, i) => {
        const r = item.result;
        const ws = r.is_webshell;
        const entropy = r.features?.entropy ?? '—';
        const tr = document.createElement('tr');
        tr.innerHTML = `
<td style="color:var(--muted)">${i + 1}</td>
<td class="bulk-fname" title="${item.filename}">${item.filename}</td>
<td><span class="tag ${ws ? 'ws' : 'ok'}">${ws ? '⚠ Webshell' : '✓ Normal'}</span></td>
<td class="bulk-conf" style="text-align:right; color:${ws ? '#f87171' : '#34d399'}">${(r.confidence * 100).toFixed(1)}%</td>
<td style="text-align:right; color:var(--muted)">${typeof entropy === 'number' ? entropy.toFixed(2) : entropy}</td>
`;
        tb.appendChild(tr);
    });
    id('bulk-table-wrap').style.display = 'block';
    hide('result-box'); hide('features-box'); hide('votes-box');
}

function addHistory(name, result) {
    id('no-hist').style.display = 'none';
    const hl = id('hist-list');
    const div = document.createElement('div');
    div.className = 'hist-item';
    div.innerHTML = `
<div>
  <div class="hist-name" title="${name}">${name}</div>
  <div class="hist-model">Model: ${result.model}</div>
</div>
<span class="tag ${result.is_webshell ? 'ws' : 'ok'}">${result.is_webshell ? 'Webshell' : 'Normal'}</span>
`;
    hl.prepend(div);
}

function updateStats(results) {
    scans += results.length;
    threats += results.filter(r => r.is_webshell).length;
    id('s-scans').textContent = scans;
    id('s-threats').textContent = threats;
}
