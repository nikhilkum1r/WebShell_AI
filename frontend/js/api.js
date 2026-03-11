/* ═══════════════════════════════════════════════════════════
   API Service Layer
   ═══════════════════════════════════════════════════════════ */

let API_BASE;
if (window.WEBSHELL_CONFIG?.API_BASE) {
    API_BASE = window.WEBSHELL_CONFIG.API_BASE;
} else {
    const host = window.location.hostname || 'localhost';
    API_BASE = `http://${host}:8000`;
}

// Ensure protocol matching (no mixed content)
if (window.location.protocol === 'https:' && API_BASE.startsWith('http:')) {
    console.warn("API_BASE is insecure but page is secure. Connectivity might fail.");
}

async function checkHealth() {
    try {
        const dot = id('sdot'), txt = id('stext');
        if (!dot || !txt) return;

        const r = await fetch(`${API_BASE}/health`);
        const d = await r.json();
        dot.className = 'sdot on'; txt.textContent = 'Online — Ready';
        id('s-device').textContent = `Device: ${d.device}`;
        id('s-ver').textContent = `API v${d.version}`;
    } catch {
        const dot = id('sdot'), txt = id('stext');
        if (dot && txt) {
            dot.className = 'sdot off'; txt.textContent = 'Offline — Start server';
        }
    }
}

async function fetchBenchmarks() {
    try {
        const r = await fetch(`${API_BASE}/stats`);
        if (!r.ok) return;
        const data = await r.json();
        const bTable = id('bench-table');
        if (!bTable) return;
        const tb = bTable.querySelector('tbody');
        if (!tb) return;
        tb.innerHTML = data.map(m => `
<tr>
  <td>${m.model}</td>
  <td class="b-perc ${m.accuracy >= .98 ? 'b-hi' : 'b-md'}">${(m.accuracy * 100).toFixed(1)}%</td>
  <td class="b-perc ${m.f1 >= .98 ? 'b-hi' : 'b-md'}">${(m.f1 * 100).toFixed(1)}%</td>
</tr>
`).join('');
    } catch { /* api not started, static fallback shown */ }
}

async function run() {
    const mSel = id('model-sel');
    if (!mSel) return;
    const model = mSel.value;
    setLoading(true);
    const t0 = performance.now();
    try {
        if (curTab === 'bulk') {
            await runBulk(model);
        } else if (curTab === 'file') {
            const file = id('fi-input').files[0];
            if (!file) { toast('Select a file first.', 'err'); return; }
            const fd = new FormData(); fd.append('file', file); fd.append('model_name', model);
            const r = await fetch(`${API_BASE}/predict/file`, { method: 'POST', body: fd });
            if (!r.ok) throw new Error((await r.json()).detail);
            showResult(await r.json(), file.name);
        } else {
            const text = id('code-input').value.trim();
            if (!text) { toast('Paste some code first.', 'err'); return; }
            if (text.length < MIN_CODE_LEN) {
                toast(`⚠ Too short! Paste at least ${MIN_CODE_LEN} characters of PHP/ASP code for accurate analysis.`, 'err');
                id('code-input').classList.add('input-error');
                return;
            }
            const r = await fetch(`${API_BASE}/predict`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model_name: model }),
            });
            if (!r.ok) throw new Error((await r.json()).detail);
            showResult(await r.json(), 'Code snippet');
        }
        id('s-latency').textContent = `${(performance.now() - t0).toFixed(0)}ms`;
    } catch (e) {
        toast(`Error: ${e.message}`, 'err');
    } finally {
        setLoading(false);
    }
}

async function runBulk(model) {
    const bInp = id('bulk-input');
    if (!bInp) return;
    const files = bInp.files;
    if (!files || !files.length) { toast('Select files for bulk scan.', 'err'); return; }
    const fd = new FormData();
    [...files].forEach(f => fd.append('files', f));
    fd.append('model_name', model);
    const r = await fetch(`${API_BASE}/predict/bulk`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail);
    const results = await r.json();
    renderBulk(results);
    results.forEach(it => addHistory(it.filename, it.result));
    updateStats(results.map(i => i.result));
}
