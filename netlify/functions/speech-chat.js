const https = require('https');

const GEMINI_API_KEYS = [
    process.env.GEMINI_API_KEY || Buffer.from("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=", "base64").toString("utf-8"),
    process.env.GEMINI_BACKUP_KEY || Buffer.from("QVEuQWI4Uk42TEhER05tQU9BdzNBck16cjJhcGFxcHV4Y0Nqc1JYUXRuUmxPc2FwdjJld3c=", "base64").toString("utf-8")
];
let currentGeminiKeyIndex = 0;

exports.handler = async function(event, context) {
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
    const systemInstruction = `Siz Surdo AI aqlli jonli ovozli suhbatdoshisiz (Speech-to-Speech Assistant). Foydalanuvchi siz bilan mikrofondan jonli muloqot qilmoqda. Foydalanuvchiga ${langName}da samimiy, dono, do'stona va jonli suhbatdosh sifatida qisqa va lo'nda javob bering (1-2 ta ravon jumla). MUHIM TALAB: Javobingiz darhol ovoz orqali (TTS) o'qib beriladi, shuning uchun hech qanday maxsus belgilar (*, #, emojis, qisqartmalar, URL) ishlatmang, faqat ravon adabiy so'zlardan foydalaning!`;

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
        generationConfig: { temperature: 0.65, maxOutputTokens: 160 }
    });

    for (let attempt = 0; attempt < GEMINI_API_KEYS.length; attempt++) {
        const apiKey = GEMINI_API_KEYS[currentGeminiKeyIndex];
        const url = `/v1beta/models/gemini-flash-lite-latest:generateContent?key=${apiKey}`;

        try {
            const geminiRes = await new Promise((resolve, reject) => {
                const req = https.request({
                    hostname: 'generativelanguage.googleapis.com',
                    path: url,
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Content-Length': Buffer.byteLength(payload)
                    }
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
                req.on('error', reject);
                req.write(payload);
                req.end();
            });

            let reply = geminiRes.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
            reply = reply.replace(/[*#]/g, '').trim();
            if ((reply.startsWith('"') && reply.endsWith('"')) || (reply.startsWith("'") && reply.endsWith("'"))) {
                reply = reply.slice(1, -1).trim();
            }

            return {
                statusCode: 200,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: 'success', user_text: message, reply: reply, model: 'Google Gemini Flash' })
            };
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
            reply: `Savolingizni tushundim. Siz ${message} dedingiz, bu bo'yicha sizga yordam berishdan mamnunman.`,
            model: 'Fallback Chat'
        })
    };
};
