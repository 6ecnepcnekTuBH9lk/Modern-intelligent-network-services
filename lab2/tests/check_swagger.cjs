const path = require('path');
const fs = require('fs');
const moduleRoot = process.env.NODE_PATH;
const { chromium } = moduleRoot ? require(require.resolve('playwright', { paths: [moduleRoot] })) : require('playwright');

(async () => {
    const output = path.resolve(__dirname, '..', 'checks');
    fs.mkdirSync(output, { recursive: true });
    const browser = await chromium.launch({ headless: true, channel: 'msedge' });
    const results = [];
    try {
        const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
        await page.goto('http://127.0.0.1:8002/docs');
        await page.locator('.opblock').nth(13).waitFor();
        await page.screenshot({ path: path.join(output, 'swagger-ui.png'), fullPage: true });
        const farm = { farm_id: 'FARM-001', region: 'Krasnodar', crop_type: 'wheat', area_ha: 2500,
            temperature_avg: 24.3, precipitation_mm: 320, payment_delay_days: 45, previous_defaults: 1, debt: 6500000 };
        const escape = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        async function execute(endpoint, filename, body, params, expected) {
            await page.goto('http://127.0.0.1:8002/docs');
            const block = page.locator('.opblock').filter({ has:
                page.locator('.opblock-summary-path').filter({ hasText: new RegExp('^' + escape(endpoint) + '$') }) });
            await block.locator('.opblock-summary').click();
            await block.getByRole('button', { name: 'Try it out' }).click();
            if (body) await block.locator('textarea').fill(JSON.stringify(body, null, 2));
            for (const [name, value] of Object.entries(params || {})) {
                const row = block.locator('tr').filter({ has:
                    page.locator('.parameter__name').filter({ hasText: new RegExp('^' + escape(name) + '(?:\\s|$)') }) });
                await row.locator('input').fill(String(value));
            }
            await block.getByRole('button', { name: 'Execute', exact: true }).click();
            const table = block.locator('.live-responses-table');
            await table.waitFor({ timeout: 70000 });
            const status = parseInt(await table.locator('.response-col_status').last().innerText(), 10);
            if (!expected.includes(status)) throw new Error(endpoint + ' returned ' + status + ': ' + await table.innerText());
            await block.screenshot({ path: path.join(output, filename + '.png') });
            results.push({ endpoint, status });
            console.log(endpoint + ': HTTP ' + status);
        }
        await execute('/health', 'health', null, {}, [200]);
        await execute('/predict', 'predict', farm, {}, [200]);
        await execute('/predictions', 'history', null, { limit: 2 }, [200]);
        await execute('/batch-predict', 'batch-predict', { items: [farm, { ...farm, farm_id: 'FARM-002' }] }, {}, [200]);
        await execute('/predict-queued', 'predict-queued', farm, {}, [200]);
        await execute('/queue-status', 'queue-status', null, {}, [200]);
        await execute('/demo/seed-weekdays', 'seed-weekdays', null, { count: 70 }, [200]);
        await execute('/predictions/by-weekday', 'pagination', null, { weekday: 'monday', page: 1, per_page: 5 }, [200]);
        await execute('/llm/chat', 'llm-chat', { prompt: 'Ответь одним словом: работает', model: 'auto', temperature: 0.3, max_tokens: 30 }, {}, [200, 502, 503, 504]);
        await execute('/predict-with-llm', 'predict-with-llm', farm, {}, [200, 502, 503, 504]);
        fs.writeFileSync(path.join(output, 'swagger-results.json'), JSON.stringify({ mock_used: false, results }, null, 2));
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
