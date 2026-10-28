import https from 'https';

const GEMINI_API_KEYS = [process.env.GEMINI_API_KEY];
let currentGeminiKeyIndex = 0;
const GEMINI_BASE_URL = new URL(process.env.GOOGLE_GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com');

async function handleEvent(event) {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: 'Method Not Allowed' };
    }

    let body = {};
    try {
        body = JSON.parse(event.body || '{}');
    } catch(e) {}

    const message = (body.message || '').trim();
    const lang = (body.lang || 'uz').toLowerCase();
    const history = Array.isArray(body.history) ? body.history : [];

    if (!message) {
        return {
            statusCode: 400,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'error', message: "Xabar bo'sh bo'lishi mumkin emas" })
        };
    }

    const langName = lang === 'uz' ? "O'zbek tili" : (lang === 'en' ? "English" : "Русский язык");
    const systemInstruction = `Siz Surdo AI aqlli jonli ovozli yordamchisiz. Foydalanuvchi mikrofondan siz bilan jonli gaplashmoqda. Foydalanuvchiga ${langName}da o'ta tez, samimiy, aniq va lo'nda qilib FAQAT 1 TA (ko'pi bilan 2 ta) qisqa gap bilan javob bering! Javobingiz darhol ovozli o'qiladi. Hech qanday yulduzcha (*), panjara (#), emoji yoki ortiqcha belgi ishlatmang!`;

    const contents = [];
    history.slice(-6).forEach(h => {
        const role = (h.role === 'user' || h.role === 'siz') ? 'user' : 'model';
        const txt = (h.text || '').trim();
        if (txt) contents.push({ role: role, parts: [{ text: txt }] });
    });
    contents.push({ role: 'user', parts: [{ text: message }] });

    const payload = JSON.stringify({
        system_instruction: { parts: [{ text: systemInstruction }] },
        contents: contents,
        generationConfig: {
            temperature: 0.5,
            maxOutputTokens: 80,
            thinkingConfig: { thinkingBudget: 0 }
        }
    });

    for (let attempt = 0; attempt < GEMINI_API_KEYS.length; attempt++) {
        const apiKey = GEMINI_API_KEYS[currentGeminiKeyIndex];
        const url = `/v1beta/models/gemini-3.5-flash:generateContent`;

        try {
            const geminiRes = await new Promise((resolve, reject) => {
                const req = https.request({
                    hostname: GEMINI_BASE_URL.hostname,
                    path: `${GEMINI_BASE_URL.pathname.replace(/\/$/, '')}${url}`,
                    method: 'POST',
                    headers: {
                        'x-goog-api-key': apiKey,
                        'Content-Type': 'application/json',
                        'Content-Length': Buffer.byteLength(payload)
                    },
                    timeout: 6000
                }, res => {
                    let data = '';
                    res.on('data', chunk => data += chunk);
                    res.on('end', () => {
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            try {
                                resolve(JSON.parse(data));
                            } catch(e) {
                                reject(e);
                            }
                        } else {
                            reject(new Error(`Gemini status ${res.statusCode}: ${data}`));
                        }
                    });
                });
                req.on('timeout', () => { req.destroy(); reject(new Error('Gemini timeout')); });
                req.on('error', reject);
                req.write(payload);
                req.end();
            });

            const parts = geminiRes.candidates?.[0]?.content?.parts || [];
            let reply = parts[0]?.text?.trim() || "";
            reply = reply.replace(/[*#]/g, '').trim();
            if ((reply.startsWith('"') && reply.endsWith('"')) || (reply.startsWith("'") && reply.endsWith("'"))) {
                reply = reply.slice(1, -1).trim();
            }

            if (reply) {
                return {
                    statusCode: 200,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status: 'success', user_text: message, reply: reply, model: 'Google Gemini Flash' })
                };
            }
        } catch(err) {
            console.warn(`[Gemini SpeechChat Key #${currentGeminiKeyIndex + 1} failover]: ${err.message}. Zaxira kalitga o'tilmoqda...`);
            currentGeminiKeyIndex = (currentGeminiKeyIndex + 1) % GEMINI_API_KEYS.length;
        }
    }

    return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            status: 'success',
            user_text: message,
            reply: `Salom! Men sizni eshitdim, sizga yordam berishga doimo tayyorman.`,
            model: 'Fallback Chat'
        })
    };
}

export default async (req) => {
    const url = new URL(req.url);
    const event = {
        httpMethod: req.method,
        queryStringParameters: Object.fromEntries(url.searchParams),
        body: req.method === 'GET' || req.method === 'HEAD' ? null : await req.text()
    };
    const result = await handleEvent(event);
    const body = result.isBase64Encoded ? Buffer.from(result.body, 'base64') : result.body;
    return new Response(body, { status: result.statusCode, headers: result.headers || {} });
};
