// Measures the replay controls at the window widths they have to work at and
// fails when they take more rows than they should, when two of them overlap or
// when the playback controls are not centered in the bar. The layout has
// regressed on phones more than once, and looking at it is not enough.
//
//   npm install playwright && npx playwright install chromium
//   NODE_PATH=$(npm root) node check-controls.js path/to/index.html
//   NODE_PATH=$(npm root) node check-controls.js https://ddnet.org/watch/?uuid=...
const { chromium } = require('playwright');

const WIDTHS = [320, 344, 360, 375, 390, 393, 412, 430, 540, 600, 601, 667, 801, 844, 880, 980, 981, 1280, 1920];

(async () => {
	const browser = await chromium.launch();
	let bad = 0;
	for (const width of WIDTHS) {
		const page = await browser.newPage({ viewport: { width, height: 800 } });
		await page.goto(process.argv[2].startsWith('http') ? process.argv[2] : 'file://' + process.argv[2]);
		// Show the bar the way a running replay does
		await page.evaluate(() => {
			document.getElementById('demobar').style.display = 'flex';
			document.getElementById('overlay').style.display = 'none';
			document.getElementById('demotitle').textContent = 'Sunny Side Up [07:33.76] - Aoe & Cloudly';
			document.getElementById('aspectbutton').hidden = false;
		});
		const result = await page.evaluate(() => {
			const rect = element => {
				const r = element.getBoundingClientRect();
				return { left: Math.round(r.left), right: Math.round(r.right), top: Math.round(r.top), bottom: Math.round(r.bottom), w: Math.round(r.width) };
			};
			const bar = document.getElementById('demobar');
			const buttons = bar.querySelector('.buttons');
			const groups = buttons.querySelectorAll(':scope > .group');
			const items = [...bar.querySelectorAll('button, a#downloadbutton, #volume')].filter(e => e.offsetParent !== null && !e.closest('#cameramenu'));
			const rows = new Map();
			for (const item of items) {
				const r = rect(item);
				const key = Math.round((r.top + r.bottom) / 2 / 24) * 24;
				if (!rows.has(key)) rows.set(key, []);
				rows.get(key).push({ id: item.id || item.title || item.textContent.trim().slice(0, 8), ...r });
			}
			// Overlap: any two items on the same row whose boxes cross
			let overlaps = [];
			for (const [, row] of rows) {
				row.sort((a, b) => a.left - b.left);
				for (let i = 1; i < row.length; i++) {
					if (row[i].left < row[i - 1].right - 1) {
						overlaps.push(`${row[i - 1].id} x ${row[i].id}`);
					}
				}
			}
			const play = rect(groups[1]);
			const barRect = rect(bar);
			return {
				rows: rows.size,
				overlaps,
				playCenterOffset: Math.round((play.left + play.right) / 2 - (barRect.left + barRect.right) / 2),
				barWidth: barRect.w,
				overflow: Math.round(Math.max(0, buttons.scrollWidth - buttons.clientWidth)),
				rowTops: [...rows.keys()].sort((a, b) => a - b),
			};
		});
		const ok = result.rows <= 2 && result.overlaps.length === 0 && Math.abs(result.playCenterOffset) <= 1 && result.overflow === 0;
		if (!ok) bad++;
		console.log(`${width}px  rows=${result.rows}  overlaps=${result.overlaps.length ? result.overlaps.join(', ') : 'none'}  play off center=${result.playCenterOffset}px  overflow=${result.overflow}px  ${ok ? 'OK' : 'BAD'}`);
		await page.close();
	}
	await browser.close();
	process.exit(bad === 0 ? 0 : 1);
})();
