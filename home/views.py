from django.shortcuts import render
from django.views import View
import cv2
import numpy as np
import os
import base64
import mediapipe as mp
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.callbacks import TensorBoard
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
from django.http import HttpResponse
from django.http import StreamingHttpResponse


# Create your views here.

class HomaPageView(View):
    def get(self, request):
        context = {}
        return render(request, "index.html", context)

    
def webcam():
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils 

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                  
        results = model.process(image)                 
        image.flags.writeable = True                  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

    def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    actions = np.array(['hello', 'thanks', 'iloveyou'])
    model = None

    if HAS_KERAS and os.path.exists('action.h5'):
        try:
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(actions.shape[0], activation='softmax'))
            model.load_weights('action.h5')
        except Exception as e:
            print("Model load error:", e)
            model = None

    colors = [(245,117,16), (117,245,16), (16,117,245)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Kamera ulanmadi yoki band", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                if model is not None:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                        # 3. Viz logic
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        image = prob_viz(res, actions, image, colors)

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, "Surdo AI - Imo-ishoralar aniqlanmoqda", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                _, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    finally:
        cap.release()

import asyncio
from asgiref.sync import sync_to_async

async def async_webcam():
    gen = webcam()
    get_next = sync_to_async(lambda: next(gen, None), thread_sensitive=False)
    while True:
        try:
            frame = await get_next()
            if frame is None:
                break
            yield frame
            await asyncio.sleep(0.01)
        except Exception as e:
            print("Streaming error:", e)
            break

class VideoPageView(View):
    async def get(self, request):
        return StreamingHttpResponse(async_webcam(), content_type='multipart/x-mixed-replace; boundary=frame')
          
class CameraPageView(View):
    def get(self, request):
        return render(request, "camera.html", {})
    
class VoicePageView(View):
    def get(self, request):
        context = {}
        return render(request, "voice.html", context)

VOICE_AUDIO_CACHE = {}

VOICELAB_API_KEYS = [
    "vlk_x7y5QaqzTurjgFKNHJ5nFAK9D8One9XaJJV7x9FGdu4",   # 1-Asosiy kalit
    "vlk_qQ4O4WlnQ-cPw23_PiBkZ8NK1Tj8Ywmu-GeltcH6VqQ",   # 2-Zaxira kalit
    "vlk_WkCzoxnsKdJ3ffT3Gt-abjO9YNeyLtGY1TZlCS1VhYg",   # 3-Zaxira kalit
    "vlk_sS-lIYDMWPqIaO8_iegg9D1n9SePqvNy0q6vthHk4YM",   # 4-Zaxira kalit
    "vlk_Syrv5hJm2hEgsUxLNQ3cKhhmxClgFMICUduHf0EZgcc",   # 5-Zaxira kalit
    "vlk_-Xishu-yP9pksI8QS_EXV8G6vJNu7VTI3Jr9NGy1s6k",   # 6-Zaxira kalit
]
CURRENT_KEY_INDEX = 0

def get_current_voicelab_key():
    global CURRENT_KEY_INDEX
    return VOICELAB_API_KEYS[CURRENT_KEY_INDEX]

def switch_to_next_voicelab_key():
    global CURRENT_KEY_INDEX
    prev_index = CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(VOICELAB_API_KEYS)
    print(f"[VoiceLab Failover] #{prev_index + 1}-kalit tugadi/xato berdi. Keyingi #{CURRENT_KEY_INDEX + 1}-zaxira kalitga uzilishsiz o'tildi!")
    return VOICELAB_API_KEYS[CURRENT_KEY_INDEX]

VOICELAB_VOICES = {
    "abdulhay": "voice_M-p5iz9z_kPWNm_C0cFqbC6X",             # Abdulhay Otaxo'jayev (Dublyaj)
    "madina": "voice_4ef06e6b-2229-4023-ad66-c2a7a2b736ff",   # Madina — Honey (Shirali ayol)
    "amir": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",     # Amir — Storyteller (Shirali erkak)
    "javohir": "voice_fa56cecf-9b0b-45ba-9ec3-494efb5b301f",  # Javohir — Presenter (Ta'limiy diktor)
    "kamola": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",   # Kamola — Silk (Mayin qiz)
    "dilnoza": "voice_f7179d1c-07ab-4b3d-b5be-84dcf2935388",  # Dilnoza — Velvet (Baxmal)
    "jasur": "voice_ba303813-3023-48aa-a454-4b5705669d35",    # Jasur — Noir (Sokin erkak)
    "safiya": "voice_2785aabd-21b5-4cc7-9845-2230186ef7d9",   # Safiya — Reassuring guide
    "shohruxmirzo": "voice_jABVrmH-0x6pbYefNCuYfH6H",         # Shohruxmirzo (Ta'limiy)
    "nozima": "voice_3602ec76-5393-43b6-9e0d-784328cb77ba",   # Nozima — Soft (Mayin)
    "sardor": "voice_xdQMoRJkjHbPIB-jounXzKq8",               # Sardor
    "botir": "voice_5e592ada-eca8-42e3-98fa-469c1c8fc999",    # Botir — Strimer
    "gulya": "voice_7b350d91-4e56-4bc3-869c-8cd6ba646c3c",    # Gulya — Strimer
    "rustam": "voice_be56058a-8f15-48bb-904b-4fe92531d5b1",   # Rustam — Anime
}

class TTSView(View):
    async def get(self, request):
        import hashlib
        import os
        import json
        import urllib.request
        import asyncio
        from django.conf import settings

        text = request.GET.get('text', '').strip()
        voice = request.GET.get('voice', 'abdulhay').lower()
        if not text:
            return HttpResponse(status=400)

        # VoiceLab.uz dagi ovoz ID sini aniqlash
        if voice.startswith("voice_"):
            voicelab_id = voice
        elif voice in VOICELAB_VOICES:
            voicelab_id = VOICELAB_VOICES[voice]
        else:
            # Agar mos kelmasa, moslikni qidiramiz yoki standart Abdulhay qilib olamiz
            matched = None
            for key, vid in VOICELAB_VOICES.items():
                if key in voice:
                    matched = vid
                    break
            voicelab_id = matched or VOICELAB_VOICES["abdulhay"]

        # Kesh tekshirish (Disk va Xotira)
        cache_key = hashlib.md5(f"voicelab_{text}_{voicelab_id}".encode('utf-8')).hexdigest()
        cache_dir = os.path.join(settings.BASE_DIR, 'static', 'audio_cache')
        os.makedirs(cache_dir, exist_ok=True)
        disk_file = os.path.join(cache_dir, f"{cache_key}.wav")

        if cache_key in VOICE_AUDIO_CACHE:
            return HttpResponse(VOICE_AUDIO_CACHE[cache_key], content_type="audio/wav")

        if os.path.exists(disk_file):
            try:
                with open(disk_file, 'rb') as f:
                    data = f.read()
                VOICE_AUDIO_CACHE[cache_key] = data
                return HttpResponse(data, content_type="audio/wav")
            except Exception:
                pass

        # Faqat VoiceLab.uz API orqali ovoz sintezi (Azure/boshqa ovozlar chiqarib tashlandi)
        loop = asyncio.get_event_loop()
        def try_voicelab():
            url = "https://api.voicelab.uz/v1/tts"
            payload = json.dumps({
                "text": text,
                "voice_id": voicelab_id,
                "language": "uz"
            }).encode("utf-8")

            total_keys = len(VOICELAB_API_KEYS)
            last_err = None

            # Barcha 6 ta zaxira API kalitlarini uzilishsiz avtomatik tekshirish
            for attempt in range(total_keys):
                api_key = get_current_voicelab_key()
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                )
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        return resp.read()
                except urllib.error.HTTPError as http_err:
                    last_err = http_err
                    print(f"[VoiceLab API] HTTP {http_err.code} xatosi (Key #{CURRENT_KEY_INDEX + 1}). Zaxira kalitga o'tilmoqda...")
                    switch_to_next_voicelab_key()
                    continue
                except Exception as e:
                    last_err = e
                    print(f"[VoiceLab API] Aloqa xatosi: {e} (Key #{CURRENT_KEY_INDEX + 1}). Zaxira kalitga o'tilmoqda...")
                    switch_to_next_voicelab_key()
                    continue

            if last_err:
                raise last_err
            return None

        try:
            audio_stream = await loop.run_in_executor(None, try_voicelab)
            if audio_stream:
                VOICE_AUDIO_CACHE[cache_key] = audio_stream
                try:
                    with open(disk_file, 'wb') as df:
                        df.write(audio_stream)
                except Exception:
                    pass
                return HttpResponse(audio_stream, content_type="audio/wav")
            else:
                return HttpResponse("VoiceLab audio bo'sh qaytdi", status=502)
        except Exception as vl_err:
            print("VoiceLab API xatoligi:", vl_err)
            return HttpResponse(f"VoiceLab TTS xatosi: {vl_err}", status=502)


class GestureListView(View):
    def get(self, request):
        from .models import SignGesture
        include_daktil = request.GET.get('include_daktil', 'false').lower() == 'true'
        qs = SignGesture.objects.filter(is_active=True)
        if not include_daktil:
            qs = qs.exclude(category='Daktil')
        gestures = list(qs.values(
            'id', 'slug', 'name', 'category', 'hand_type', 'icon', 
            'spoken_text', 'description', 'audio_key', 'finger_pattern'
        ))
        from django.http import JsonResponse
        return JsonResponse({
            'status': 'success',
            'count': len(gestures),
            'gestures': gestures
        }, json_dumps_params={'ensure_ascii': False})


from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
class GestureMatchView(View):
    def post(self, request):
        from .models import SignGesture
        from django.http import JsonResponse
        import json
        try:
            data = json.loads(request.body.decode('utf-8'))
        except Exception:
            data = request.POST

        pattern = data.get('finger_pattern', '').strip()
        slug = data.get('slug', '').strip()

        gesture = None
        if slug:
            gesture = SignGesture.objects.filter(slug=slug, is_active=True).exclude(category='Daktil').first()
        elif pattern:
            gesture = SignGesture.objects.filter(finger_pattern=pattern, is_active=True).exclude(category='Daktil').first()

        if gesture:
            return JsonResponse({
                'match': True,
                'gesture': {
                    'id': gesture.id,
                    'slug': gesture.slug,
                    'name': gesture.name,
                    'category': gesture.category,
                    'hand_type': gesture.hand_type,
                    'icon': gesture.icon,
                    'spoken_text': gesture.spoken_text,
                    'description': gesture.description,
                    'audio_key': gesture.audio_key
                }
            }, json_dumps_params={'ensure_ascii': False})

        return JsonResponse({'match': False, 'message': 'Imo-ishora topilmadi'}, status=404)


@method_decorator(csrf_exempt, name='dispatch')
class AIGrammarView(View):
    async def post(self, request):
        import json
        import urllib.request
        from django.http import JsonResponse
        import asyncio

        try:
            body = json.loads(request.body.decode('utf-8'))
        except Exception:
            body = {}

        raw_words = body.get('words', [])
        if isinstance(raw_words, str):
            words = [w.strip() for w in raw_words.split(',') if w.strip()]
        else:
            words = [str(w).strip() for w in raw_words if str(w).strip()]

        lang = body.get('lang', 'uz').lower()
        if not words:
            default_empty = {
                'en': "Hello! Please perform some gestures first.",
                'ru': "Здравствуйте! Пожалуйста, покажите жесты сначала.",
                'uz': "Assalomu alaykum! Iltimos, avval imo-ishoralarni ko'rsating."
            }
            return JsonResponse({'status': 'success', 'sentence': default_empty.get(lang, default_empty['uz'])})

        gemini_api_key = os.getenv("GEMINI_API_KEY", base64.b64decode("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=").decode("utf-8"))
        words_str = ", ".join(words)

        lang_name = "O'zbek tili" if lang == 'uz' else ("English" if lang == 'en' else "Русский язык")
        system_instruction = (
            "Siz imo-ishoralarni nutqqa aylantiruvchi Surdo AI yordamchisisiz. "
            f"Foydalanuvchi ko'rsatgan uzuq-yuluq imo-ishora so'zlaridan {lang_name}da "
            "odobli, to'liq, ravon va madaniy bitta adabiy jumla tuzib bering. "
            "QAT'IY QOIDA: Faqat va faqat yakuniy gapni yozing! Kirish so'zlari ('Mana...', 'Albatta...'), "
            "tushuntirishlar, qo'shtirnoq yoki '**' belgilarini aslo qo'shmang!"
        )
        prompt = f"Imo-ishoralar so'zlari: {words_str}. Til: {lang_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gemini_api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25}
        }

        loop = asyncio.get_event_loop()
        def call_gemini():
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=9) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                    text = text[1:-1].strip()
                return text

        try:
            sentence = await loop.run_in_executor(None, call_gemini)
            return JsonResponse({'status': 'success', 'sentence': sentence, 'model': 'Google Gemini Flash'})
        except Exception as e:
            print("Gemini API error, falling back to local NLP:", e)
            lower_words = [w.lower() for w in words]
            has = lambda kw: any(kw in w for w in lower_words)
            if has('kasal') or has('yordam') or has('betob'):
                fb = {
                    'en': "Hello! I am feeling unwell, could you please help me urgently?",
                    'ru': "Здравствуйте! Я плохо себя чувствую, пожалуйста, окажите мне срочную помощь.",
                    'uz': "Assalomu alaykum! Men o'zimni yomon his qilyapman, iltimos, menga zudlik bilan yordam bera olasizmi?"
                }
            elif has('suv') or has('chanqa'):
                fb = {
                    'en': "Excuse me! I am really thirsty, could you please give me a glass of fresh water?",
                    'ru': "Простите! Я очень хочу пить, не могли бы вы дать мне стакан чистой воды?",
                    'uz': "Kechirasiz! Juda chanqadim, iltimos menga bir piyola toza suv bera olasizmi?"
                }
            elif has('rahmat') and has('ajoyib'):
                fb = {
                    'en': "Thank you so much from the bottom of my heart, everything was absolutely wonderful!",
                    'ru': "Огромное вам спасибо от всей души, всё прошло просто великолепно!",
                    'uz': "Sizga chin qalbimdan kattakon rahmat aytaman, barchasi nihoyatda ajoyib va go'zal bo'ldi!"
                }
            else:
                fb = {
                    'en': f"Respectfully: {words_str}. I wish you a wonderful day!",
                    'ru': f"С уважением: {words_str}. Желаю вам всего наилучшего!",
                    'uz': f"Hurmat bilan ma'lum qilaman: {words_str}. E'tiboringiz va tushunganingiz uchun tashakkur!"
                }
            return JsonResponse({'status': 'success', 'sentence': fb.get(lang, fb['uz']), 'model': 'Local NLP Fallback'})

    async def get(self, request):
        words = request.GET.get('words', '').split(',')
        lang = request.GET.get('lang', 'uz')
        # forward to post logic
        class FakeRequest:
            body = json.dumps({'words': words, 'lang': lang}).encode('utf-8')
        return await self.post(FakeRequest())

