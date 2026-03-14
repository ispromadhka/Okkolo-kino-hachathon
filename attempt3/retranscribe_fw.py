import static_ffmpeg; static_ffmpeg.add_paths()
import os, re, pickle, logging, time, subprocess, tempfile
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

DATA_DIR = Path('data')

def convert_to_wav(input_path, output_path):
    r = subprocess.run(['ffmpeg','-y','-i',str(input_path),'-ar','16000','-ac','1','-f','wav',str(output_path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0

def main():
    t0 = time.time()
    import csv
    audio_files = []
    with open(DATA_DIR / 'audio_files.csv') as f:
        for row in csv.DictReader(f):
            audio_files.append(row['audio_path'])

    existing = [(af, DATA_DIR / 'video-rag' / af) for af in audio_files if (DATA_DIR / 'video-rag' / af).exists()]
    log.info(f'Found {len(existing)}/{len(audio_files)} audio files')

    log.info('Loading faster-whisper large-v3...')
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="int8")
    log.info('Model loaded')

    new_transcripts = {}
    tmpdir = tempfile.mkdtemp()

    for audio_rel, audio_full in tqdm(existing, desc='Transcribing'):
        m = re.search(r'(audio_[a-f0-9]+)', audio_rel)
        if not m: continue
        audio_id = m.group(1)
        video_hash = audio_id.replace('audio_', 'video_')

        wav_path = os.path.join(tmpdir, audio_id + '.wav')
        if not convert_to_wav(audio_full, wav_path):
            log.warning(f'Failed to convert {audio_rel}')
            continue

        try:
            segs_iter, info = model.transcribe(wav_path, beam_size=1, vad_filter=True)
            segments = []
            for seg in segs_iter:
                segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip()
                })

            ext = Path(audio_rel).suffix.lstrip('.')
            key = 'videos/' + video_hash + '.' + ext
            new_transcripts[key] = segments

            if len(new_transcripts) % 50 == 0:
                log.info(f'Progress: {len(new_transcripts)} done, lang={info.language}')
        except Exception as e:
            log.error(f'Failed {audio_rel}: {e}')
            continue

        if os.path.exists(wav_path):
            os.remove(wav_path)

    with open('new_transcripts.pkl', 'wb') as f:
        pickle.dump(new_transcripts, f)
    log.info(f'Saved {len(new_transcripts)} transcripts in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
