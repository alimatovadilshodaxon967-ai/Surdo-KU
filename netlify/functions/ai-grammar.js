const https = require('https');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || Buffer.from("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=", "base64").toString("utf-8");

exports.handler = async function(event, context) {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: 'Method Not Allowed' };
    }

    let body = {};
    try {
        body = JSON.parse(event.body || '{}');
    } catch(e) {}

    const words = Array.isArray(body.words) ? body.words : (body.words ? String(body.words).split(',') : []);
    const cleanWords = words.map(w => String(w).trim()).filter(Boolean);

    if (cleanWords.length === 0) {
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                status: 'success',
                sentence: "Assalomu alaykum! Iltimos, avval imo-ishoralarni ko'rsating."
            })
        };
    }

    const wordsStr = cleanWords.join(', ');
    const systemInstruction = "Siz imo-ishoralarni nutqqa aylantiruvchi Surdo AI yordamchisisiz. Foydalanuvchi ko'rsatgan uzuq-yuluq imo-ishora so'zlaridan O'zbek tilida odobli, to'liq, ravon va madaniy bitta adabiy jumla tuzib bering. QAT'IY QOIDA: Faqat va faqat yakuniy gapni yozing! Kirish so'zlari, tushuntirishlar, qo'shtirnoq yoki '**' belgilarini aslo qo'shmang!";
    const prompt = `Imo-ishoralar so'zlari: ${wordsStr}. Til: O'zbek tili`;

    const payload = JSON.stringify({
        system_instruction: { parts: [{ text: systemInstruction }] },
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.25 }
    });

    const url = `/v1beta/models/gemini-flash-lite-latest:generateContent?key=${GEMINI_API_KEY}`;

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
                    try {
                        resolve(JSON.parse(data));
                    } catch(e) {
                        reject(e);
                    }
                });
            });
            req.on('error', reject);
            req.write(payload);
            req.end();
        });

        let sentence = geminiRes.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
        if ((sentence.startsWith('"') && sentence.endsWith('"')) || (sentence.startsWith("'") && sentence.endsWith("'"))) {
            sentence = sentence.slice(1, -1).trim();
        }

        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'success', sentence: sentence || wordsStr })
        };
    } catch(err) {
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'success', sentence: `Hurmat bilan: ${wordsStr}. Rahmat!` })
        };
    }
};
