import static_ffmpeg; static_ffmpeg.add_paths()
import os, re, pickle, logging, time, subprocess, tempfile, sys
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

DATA_DIR = Path('data')
GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOTAL_GPUS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

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

    # Split files across GPUs
    my_files = [existing[i] for i in range(len(existing)) if i % TOTAL_GPUS == GPU_ID]
    log.info(f'GPU {GPU_ID}: processing {len(my_files)}/{len(existing)} files')

    log.info(f'GPU {GPU_ID}: Loading faster-whisper large-v3-turbo...')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    log.info(f'GPU {GPU_ID}: Model loaded')

    new_transcripts = {}
    tmpdir = tempfile.mkdtemp()

    for audio_rel, audio_full in tqdm(my_files, desc=f'GPU{GPU_ID}'):
        m = re.search(r'(audio_[a-f0-9]+)', audio_rel)
        if not m: continue
        audio_id = m.group(1)
        video_hash = audio_id.replace('audio_', 'video_')

        wav_path = os.path.join(tmpdir, audio_id + '.wav')
        if not convert_to_wav(audio_full, wav_path):
            continue

        try:
            segs_iter, info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
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
        except Exception as e:
            log.error(f'Failed {audio_rel}: {e}')
            continue

        if os.path.exists(wav_path):
            os.remove(wav_path)

    output_path = f'new_transcripts_gpu{GPU_ID}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(new_transcripts, f)
    log.info(f'GPU {GPU_ID}: Saved {len(new_transcripts)} transcripts to {output_path} in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
