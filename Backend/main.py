import os
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import aiofiles
from gtts import gTTS
from dotenv import load_dotenv
import subprocess
import asyncio


load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def save_uploaded_file(uploaded_file: UploadFile, destination: str):
    async with aiofiles.open(destination, "wb") as f:
        content = await uploaded_file.read()
        await f.write(content)
        logger.info(f"File created: {destination}")


async def convert_text_to_speech(answer_text: str, output_audio_file: str):
    try:
        tts = gTTS(answer_text, lang='ko')
        tts.save(output_audio_file)
        logger.info(f"File created: {output_audio_file}")
    except Exception as e:
        logger.error(f"Error during TTS conversion: {str(e)}")


@app.post("/api/ask")
async def ask_question(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an audio file.")
    
    input_file = "input_audio.webm"
    output_audio_file = "output_audio.mp3"

    try:
       
        await save_uploaded_file(file, input_file)
        
       
        wav_file = "input_audio.wav"
        try:
            command = [
                'ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_file, '-y'
            ]
            subprocess.run(command, check=True)
            logger.info(f"File converted to WAV: {wav_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error during conversion to WAV: {e}")
            raise HTTPException(status_code=500, detail="FFmpeg error during conversion to WAV.")

        
        with open(wav_file, "rb") as audio:
            logger.info(f"Attempting to transcribe file: {wav_file}")
            transcript_response = openai.Audio.transcribe("whisper-1", audio)
            question_text = transcript_response["text"].strip()
            logger.info(f"Transcription result: {question_text}")

       
        if not question_text or len(question_text) < 0:  
            warning_message = "입력이 인식되지 않았습니다. 다시 시도해주세요."
            logger.warning(warning_message)
            return JSONResponse(content={"status": "no_meaningful_input", "message": warning_message}, status_code=200)

        
        logger.info(f"Generating response for the question: {question_text}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question_text}],
            max_tokens=150
        )

        answer_text = response.choices[0].message['content'].strip()
        logger.info(f"Generated answer: {answer_text}")

        
        background_tasks.add_task(convert_text_to_speech, answer_text, output_audio_file)
        
        return JSONResponse(content={"question": question_text, "answer": answer_text, "audio_url": f"/api/audio/{output_audio_file}"})

    except FileNotFoundError as fnf_error:
        logger.error(f"File processing error: {str(fnf_error)}")
        raise HTTPException(status_code=500, detail="File processing error. File not found.")
    except openai.error.OpenAIError as openai_error:
        logger.error(f"OpenAI API error: {str(openai_error)}")
        raise HTTPException(status_code=500, detail="OpenAI API error.")
    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Audio processing error. Please check the server logs.")
    finally:
        
        background_tasks.add_task(delayed_cleanup, input_file, wav_file)

async def delayed_cleanup(*files):
    await asyncio.sleep(60)  
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"File deleted: {file}")

@app.get("/api/audio/{audio_file}")
async def get_audio_file(audio_file: str):
    file_path = f"./{audio_file}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename=audio_file)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
