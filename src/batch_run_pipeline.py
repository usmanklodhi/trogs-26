
# import os
# import subprocess
# import time

# RAW_DIR = '../data/raw/Test PNG Images'
# OUT_DIR = '../results/batch_transcripts'
# VIS_DIR = '../results/batch_visualizations'
# DIAG_FILE = '../results/batch_diagnostics.txt'
# PIPELINE_SCRIPT = 'run_pipeline.py'

# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(VIS_DIR, exist_ok=True)

# # List all files ending with _Rotation1_300dpi.png
# all_files = os.listdir(RAW_DIR)
# rotation1_files = [f for f in all_files if f.endswith('_Rotation1_300dpi.png')]

# diagnostics = []

# for file1 in rotation1_files:
#     base = file1.replace('_Rotation1_300dpi.png', '')
#     file2 = f'{base}_Rotation2_300dpi.png'
#     path1 = os.path.join(RAW_DIR, file1)
#     path2 = os.path.join(RAW_DIR, file2)
#     if not os.path.exists(path2):
#         msg = f"Warning: {file2} not found, skipping {file1}"
#         print(msg)
#         diagnostics.append({
#             'base': base,
#             'img1': path1,
#             'img2': path2,
#             'status': 'missing_rotation2',
#             'time_sec': None,
#             'error': msg
#         })
#         continue
#     out_txt = os.path.join(OUT_DIR, f'{base}_Transcript.txt')
#     out_vis = os.path.join(VIS_DIR, f'{base}_Visualized.png')
#     cmd = [
#         'python', PIPELINE_SCRIPT,
#         '--img1', path1,
#         '--img2', path2,
#         '--out', out_txt,
#         '--vis_out', out_vis
#     ]
#     print(f"Running: {' '.join(cmd)}")
#     start = time.time()
#     try:
#         subprocess.run(cmd, check=True)
#         elapsed = time.time() - start
#         diagnostics.append({
#             'base': base,
#             'img1': path1,
#             'img2': path2,
#             'status': 'success',
#             'time_sec': elapsed,
#             'error': None
#         })
#         print(f"Done {base} in {elapsed:.2f} seconds.")
#     except subprocess.CalledProcessError as e:
#         elapsed = time.time() - start
#         diagnostics.append({
#             'base': base,
#             'img1': path1,
#             'img2': path2,
#             'status': 'error',
#             'time_sec': elapsed,
#             'error': str(e)
#         })
#         print(f"Error processing {base}: {e}")

# # Save diagnostics to file
# with open(DIAG_FILE, 'w', encoding='utf-8') as f:
#     f.write('base\timg1\timg2\tstatus\ttime_sec\terror\n')
#     for d in diagnostics:
#         f.write(f"{d['base']}\t{d['img1']}\t{d['img2']}\t{d['status']}\t{d['time_sec']}\t{d['error']}\n")

# print(f"Batch processing complete. Diagnostics saved to {DIAG_FILE}")
