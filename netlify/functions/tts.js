const https = require('https');

const VOICELAB_API_KEY = "vlk_3ZT5rV4nRPDlS-Tj4Gajmv9J-y0y04qlWc0-dSXPFio";
const VOICELAB_VOICES = {
    "madina": "voice_4ef06e6b-2229-4023-ad66-c2a7a2b736ff",
    "amir": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",
    "sardor": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",
    "javohir": "voice_fa56cecf-9b0b-45ba-9ec3-494efb5b301f",
    "kamola": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",
    "mohira": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",
    "abdulhay": "voice_M-p5iz9z_kPWNm_C0cFqbC6X"
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

    try {
        const audioBuffer = await new Promise((resolve, reject) => {
            const req = https.request({
                hostname: 'api.voicelab.uz',
                path: '/v1/tts',
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${VOICELAB_API_KEY}`,
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
        console.error('Netlify TTS error:', err);
        return {
            statusCode: 502,
            body: JSON.stringify({ error: err.message })
        };
    }
};
