const https = require('https');

const VOICELAB_API_KEYS = [
    "vlk_x7y5QaqzTurjgFKNHJ5nFAK9D8One9XaJJV7x9FGdu4",
    "vlk_qQ4O4WlnQ-cPw23_PiBkZ8NK1Tj8Ywmu-GeltcH6VqQ",
    "vlk_WkCzoxnsKdJ3ffT3Gt-abjO9YNeyLtGY1TZlCS1VhYg",
    "vlk_sS-lIYDMWPqIaO8_iegg9D1n9SePqvNy0q6vthHk4YM",
    "vlk_Syrv5hJm2hEgsUxLNQ3cKhhmxClgFMICUduHf0EZgcc",
    "vlk_-Xishu-yP9pksI8QS_EXV8G6vJNu7VTI3Jr9NGy1s6k"
];
let currentKeyIndex = 0;
const VOICELAB_VOICES = {
    "abdulhay": "voice_M-p5iz9z_kPWNm_C0cFqbC6X",
    "madina": "voice_4ef06e6b-2229-4023-ad66-c2a7a2b736ff",
    "amir": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",
    "javohir": "voice_fa56cecf-9b0b-45ba-9ec3-494efb5b301f",
    "kamola": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",
    "dilnoza": "voice_f7179d1c-07ab-4b3d-b5be-84dcf2935388",
    "jasur": "voice_ba303813-3023-48aa-a454-4b5705669d35",
    "safiya": "voice_2785aabd-21b5-4cc7-9845-2230186ef7d9",
    "shohruxmirzo": "voice_jABVrmH-0x6pbYefNCuYfH6H",
    "nozima": "voice_3602ec76-5393-43b6-9e0d-784328cb77ba",
    "sardor": "voice_xdQMoRJkjHbPIB-jounXzKq8",
    "botir": "voice_5e592ada-eca8-42e3-98fa-469c1c8fc999",
    "gulya": "voice_7b350d91-4e56-4bc3-869c-8cd6ba646c3c",
    "rustam": "voice_be56058a-8f15-48bb-904b-4fe92531d5b1"
};

exports.handler = async function(event, context) {
    const params = event.queryStringParameters || {};
    const text = (params.text || '').trim();
    const voice = (params.voice || 'madina').toLowerCase();

    if (!text) {
        return { statusCode: 400, body: 'Text query parameter is required' };
    }

    const voiceId = VOICELAB_VOICES[voice] || VOICELAB_VOICES["madina"];
    const payload = JSON.stringify({
        text: text,
        voice_id: voiceId,
        language: "uz"
    });

    let lastError = null;

    for (let attempt = 0; attempt < VOICELAB_API_KEYS.length; attempt++) {
        const apiKey = VOICELAB_API_KEYS[currentKeyIndex];
        try {
            const audioBuffer = await new Promise((resolve, reject) => {
                const req = https.request({
                    hostname: 'api.voicelab.uz',
                    path: '/v1/tts',
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${apiKey}`,
                        'Content-Type': 'application/json',
                        'Content-Length': Buffer.byteLength(payload)
                    }
                }, (res) => {
                    const chunks = [];
                    res.on('data', chunk => chunks.push(chunk));
                    res.on('end', () => {
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            resolve(Buffer.concat(chunks));
                        } else {
                            reject(new Error(`VoiceLab status ${res.statusCode}: ${Buffer.concat(chunks).toString()}`));
                        }
                    });
                });
                req.on('error', reject);
                req.write(payload);
                req.end();
            });

            return {
                statusCode: 200,
                headers: {
                    'Content-Type': 'audio/wav',
                    'Cache-Control': 'public, max-age=86400'
                },
                body: audioBuffer.toString('base64'),
                isBase64Encoded: true
            };
        } catch (err) {
            lastError = err;
            console.warn(`[VoiceLab Key #${currentKeyIndex + 1} failover]: ${err.message}. Keyingi kalitga o'tilmoqda...`);
            currentKeyIndex = (currentKeyIndex + 1) % VOICELAB_API_KEYS.length;
        }
    }

    console.error('All VoiceLab API keys failed:', lastError);
    return {
        statusCode: 502,
        body: JSON.stringify({ error: lastError ? lastError.message : 'All API keys exhausted' })
    };
};
