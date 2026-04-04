/* distill course reader */

var $ = function (s) { return document.querySelector(s); };
var $$ = function (s) { return document.querySelectorAll(s); };

var courseName = '';
var courseData = null;
var currentModuleIdx = 0;
var staticModules = null;

/* ── Theme ────────────────────────────────────────── */

function getTheme() {
    var stored = localStorage.getItem('distill_theme');
    if (stored === 'dark' || stored === 'light') return stored;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
}

(function () {
    var stored = localStorage.getItem('distill_theme');
    if (stored) document.documentElement.setAttribute('data-theme', stored);
})();

$('#theme-toggle').addEventListener('click', function () {
    var next = getTheme() === 'dark' ? 'light' : 'dark';
    localStorage.setItem('distill_theme', next);
    applyTheme(next);
});

/* ── Init ─────────────────────────────────────────── */

async function init() {
    var params = new URLSearchParams(location.search);

    // Static mode: course data embedded in the page
    var embedded = document.getElementById('course-data');
    if (embedded) {
        try {
            var data = JSON.parse(embedded.textContent);
            courseName = data.courseName;
            courseData = data.detail;
            staticModules = data.modules;
        } catch (e) {
            $('#article').innerHTML = '<p>Failed to load course data.</p>';
            return;
        }
    } else {
        // API mode: fetch from local server
        courseName = params.get('course') || '';

        if (!courseName) {
            $('#article').innerHTML = '<p>No course specified. <a href="/">&larr; Back to Distill</a></p>';
            return;
        }

        try {
            var resp = await fetch('/api/courses/' + encodeURIComponent(courseName) + '/detail');
            if (!resp.ok) throw new Error('not found');
            courseData = await resp.json();
        } catch (e) {
            $('#article').innerHTML = '<p>Could not load course. <a href="/">&larr; Back to Distill</a></p>';
            return;
        }
    }

    document.title = courseData.title + ' \u2014 Distill';
    $('#course-title').textContent = courseData.title;

    renderSidebar();
    renderModuleSelect();

    var moduleParam = parseInt(params.get('module') || '0', 10);
    if (moduleParam < 0 || moduleParam >= courseData.modules.length) moduleParam = 0;
    await loadModule(moduleParam);
}

/* ── Navigation ───────────────────────────────────── */

function renderSidebar() {
    var sb = $('#module-sidebar');
    sb.innerHTML = courseData.modules.map(function (m, i) {
        return '<a href="?course=' + encodeURIComponent(courseName) + '&module=' + i +
            '" class="sidebar-item" data-index="' + i + '" title="' + escAttr(m.title) + '">' +
            '<span class="sidebar-num">' + i + '</span></a>';
    }).join('');

    sb.addEventListener('click', async function (e) {
        var item = e.target.closest('.sidebar-item');
        if (!item) return;
        e.preventDefault();
        await loadModule(parseInt(item.dataset.index, 10));
    });
}

function renderModuleSelect() {
    var sel = $('#module-select');
    sel.innerHTML = courseData.modules.map(function (m, i) {
        return '<option value="' + i + '">' + i + '. ' + escHtml(m.title) + '</option>';
    }).join('');

    sel.addEventListener('change', async function () {
        await loadModule(parseInt(sel.value, 10));
    });
}

function renderPager() {
    var pager = $('#module-pager');
    var prev = currentModuleIdx > 0 ? courseData.modules[currentModuleIdx - 1] : null;
    var next = currentModuleIdx < courseData.modules.length - 1 ? courseData.modules[currentModuleIdx + 1] : null;

    var html = '';

    if (prev) {
        html += '<a href="#" class="pager-link pager-prev" data-index="' + (currentModuleIdx - 1) + '">' +
            '<span class="pager-dir">&larr; previous</span>' +
            '<span class="pager-title">' + escHtml(prev.title) + '</span></a>';
    } else {
        html += '<span class="pager-link disabled"></span>';
    }

    if (next) {
        html += '<a href="#" class="pager-link pager-next" data-index="' + (currentModuleIdx + 1) + '">' +
            '<span class="pager-dir">next &rarr;</span>' +
            '<span class="pager-title">' + escHtml(next.title) + '</span></a>';
    } else {
        html += '<span class="pager-link disabled"></span>';
    }

    pager.innerHTML = html;

    pager.querySelectorAll('.pager-link[data-index]').forEach(function (btn) {
        btn.addEventListener('click', async function (e) {
            e.preventDefault();
            await loadModule(parseInt(btn.dataset.index, 10));
        });
    });
}

/* ── Module loading ───────────────────────────────── */

async function loadModule(index) {
    if (index < 0 || index >= courseData.modules.length) return;
    currentModuleIdx = index;
    var mod = courseData.modules[index];

    if (!mod || !mod.dir_name) {
        $('#article').innerHTML = '<p>Module directory not found.</p>';
        return;
    }

    // Update URL without reload
    var url = new URL(location);
    url.searchParams.set('module', index);
    history.replaceState(null, '', url);

    // Update active sidebar item
    $$('.sidebar-item').forEach(function (el) {
        el.classList.toggle('active', parseInt(el.dataset.index, 10) === index);
    });
    $('#module-select').value = index;

    // Loading state
    $('#article').innerHTML = '<p class="loading">Loading\u2026</p>';

    try {
        var data;
        if (staticModules && staticModules[mod.dir_name]) {
            data = staticModules[mod.dir_name];
        } else {
            var resp = await fetch(
                '/api/courses/' + encodeURIComponent(courseName) +
                '/module/' + encodeURIComponent(mod.dir_name)
            );
            if (!resp.ok) throw new Error('Failed to load module');
            data = await resp.json();
        }

        renderContent(data.content, mod.dir_name);
        if (data.exercises && data.exercises.length > 0) {
            renderExercises(data.exercises);
        }
        renderPager();

        // Scroll to hash target if present, otherwise scroll to top
        var hash = location.hash;
        if (hash) {
            var target = document.querySelector(hash);
            if (target) {
                target.scrollIntoView({ behavior: 'instant' });
            } else {
                window.scrollTo({ top: 0, behavior: 'instant' });
            }
        } else {
            window.scrollTo({ top: 0, behavior: 'instant' });
        }
    } catch (e) {
        $('#article').innerHTML = '<p>Error loading module content.</p>';
    }
}

/* ── Markdown rendering pipeline ──────────────────── */

function renderContent(markdown, dirName) {
    // 1. Protect math blocks from marked processing
    var mathResult = protectMath(markdown);
    var md = mathResult.md;
    var blocks = mathResult.blocks;

    // 2. Convert footnotes to Tufte sidenote HTML
    md = convertFootnotes(md);

    // 3. Parse markdown to HTML
    var html = marked.parse(md);

    // 4. Restore math blocks (render via KaTeX)
    html = restoreMath(html, blocks);

    // 5. Rewrite relative image paths to API endpoint
    html = rewriteImages(html, dirName);

    // 6. Wrap standalone images in <figure> elements
    html = wrapImagesInFigures(html);

    // 7. Insert
    $('#article').innerHTML = html;

    // 8. Add IDs to headings so TOC fragment links work
    var headingMap = {};
    $$('#article h1, #article h2, #article h3, #article h4, #article h5, #article h6').forEach(function (el) {
        if (!el.id) {
            el.id = el.textContent
                .toLowerCase()
                .replace(/['\u2018\u2019]/g, '')
                .replace(/[^\w\s-]/g, '-')
                .trim()
                .replace(/\s+/g, '-')
                .replace(/-{2,}/g, '-')
                .replace(/^-|-$/g, '');
        }
        headingMap[el.id] = el;
    });

    // 9. Fix TOC links whose short slugs don't exactly match heading IDs
    var headingIds = Object.keys(headingMap);
    $$('#article a[href^="#"]').forEach(function (a) {
        var target = a.getAttribute('href').slice(1);
        if (headingMap[target]) return; // exact match
        // Find a heading ID that contains the slug as a whole-word segment
        for (var i = 0; i < headingIds.length; i++) {
            var hid = headingIds[i];
            var pos = hid.indexOf(target);
            if (pos === -1) continue;
            var before = pos === 0 || hid[pos - 1] === '-';
            var afterIdx = pos + target.length;
            var after = afterIdx >= hid.length || hid[afterIdx] === '-';
            if (before && after) {
                a.setAttribute('href', '#' + hid);
                break;
            }
        }
    });

    // 9. Syntax-highlight all code blocks
    $$('#article pre code').forEach(function (el) {
        hljs.highlightElement(el);
    });
}

/* ── Math protection ──────────────────────────────── */

function protectMath(md) {
    var blocks = [];

    // Display math: $$...$$ (possibly multiline)
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, function (_, math) {
        blocks.push({ display: true, content: math });
        return '\x00MATH' + (blocks.length - 1) + '\x00';
    });

    // Inline math: $...$
    // Not preceded by \ or $ (prevents \$ and $$), content has no $ or newlines
    md = md.replace(/(?<![\\$])\$(?!\$)([^\$\n]+?)\$/g, function (_, math) {
        blocks.push({ display: false, content: math });
        return '\x00MATH' + (blocks.length - 1) + '\x00';
    });

    return { md: md, blocks: blocks };
}

function restoreMath(html, blocks) {
    return html.replace(/\x00MATH(\d+)\x00/g, function (_, idx) {
        var b = blocks[parseInt(idx, 10)];
        try {
            var rendered = katex.renderToString(b.content.trim(), {
                displayMode: b.display,
                throwOnError: false,
                trust: true,
            });
            if (b.display) {
                return '<div class="math-display">' + rendered + '</div>';
            }
            return rendered;
        } catch (e) {
            // Fallback: show raw math
            return b.display ? '$$' + b.content + '$$' : '$' + b.content + '$';
        }
    });
}

/* ── Footnotes → Sidenotes ────────────────────────── */

function convertFootnotes(md) {
    // Extract footnote definitions: [^id]: text (possibly multiline with indentation)
    var defs = {};
    md = md.replace(/^\[\^(\w+)\]:\s*([\s\S]+?)(?=\n\[\^|\n\n|$)/gm, function (_, id, text) {
        defs[id] = text.trim();
        return '';
    });

    // Replace inline references with sidenote HTML
    var counter = 0;
    md = md.replace(/\[\^(\w+)\]/g, function (_, id) {
        var text = defs[id];
        if (!text) return '';
        counter++;
        return '<label for="sn-' + counter + '" class="margin-toggle sidenote-number"></label>' +
            '<input type="checkbox" id="sn-' + counter + '" class="margin-toggle"/>' +
            '<span class="sidenote">' + text + '</span>';
    });

    return md;
}

/* ── Image handling ───────────────────────────────── */

function rewriteImages(html, dirName) {
    // Rewrite relative src attributes
    return html.replace(/src="(?!https?:\/\/|\/|data:)([^"]+)"/g, function (_, src) {
        if (staticModules) {
            // Static mode: images are at ./files/{dirName}/
            return 'src="files/' + dirName + '/' + src + '"';
        }
        return 'src="/api/courses/' + courseName + '/files/' + dirName + '/' + src + '"';
    });
}

function wrapImagesInFigures(html) {
    // Convert <p><img ...></p> to <figure><img ...><figcaption>...</figcaption></figure>
    return html.replace(/<p>\s*(<img\s[^>]+>)\s*<\/p>/g, function (_, img) {
        var altMatch = img.match(/alt="([^"]*)"/);
        var alt = altMatch ? altMatch[1] : '';
        var caption = alt ? '<figcaption>' + alt + '</figcaption>' : '';
        return '<figure>' + img + caption + '</figure>';
    });
}

/* ── Exercises ────────────────────────────────────── */

function renderExercises(exercises) {
    var section = document.createElement('section');
    section.className = 'exercises-section';
    section.innerHTML = '<h2>Exercises</h2>';

    exercises.forEach(function (ex) {
        var card = document.createElement('div');
        card.className = 'exercise-card';

        var header = document.createElement('div');
        header.className = 'exercise-header';
        header.innerHTML = '<span class="arrow">&#9656;</span> <span>' + escHtml(ex.filename) + '</span>';

        var body = document.createElement('div');
        body.className = 'exercise-body';
        var pre = document.createElement('pre');
        var code = document.createElement('code');
        code.className = 'language-python';
        code.textContent = ex.content;
        pre.appendChild(code);
        body.appendChild(pre);

        header.addEventListener('click', function () {
            header.classList.toggle('expanded');
            body.classList.toggle('visible');
            // Highlight on first expand
            if (body.classList.contains('visible') && !code.classList.contains('hljs')) {
                hljs.highlightElement(code);
            }
        });

        card.appendChild(header);
        card.appendChild(body);
        section.appendChild(card);
    });

    $('#article').appendChild(section);
}

/* ── Utilities ────────────────────────────────────── */

function escHtml(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function escAttr(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ── Start ────────────────────────────────────────── */

init();
