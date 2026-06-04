import os, csv, glob

DATA_ROOT = '/home/vt_ai_test1/KarenHE/contrast-phys/RPPG_data_benny_eric/Recorded_3 Cameras'

cam_files = {
    'C920': 'frames_timestamp.csv',
    'Android_311': 'android_311YJP3P3080D200020_frames_timestamp.csv',
    'Android_RFCN': 'android_RFCN3050F7T_frames_timestamp.csv',
}

results = []
for subj_dir in sorted(glob.glob(os.path.join(DATA_ROOT, '*'))):
    subj = os.path.basename(subj_dir)
    for ver_dir in sorted(glob.glob(os.path.join(subj_dir, 'v*'))):
        ver = os.path.basename(ver_dir)
        for cam_name, csv_file in cam_files.items():
            fpath = os.path.join(ver_dir, csv_file)
            if not os.path.isfile(fpath):
                results.append((subj, ver, cam_name, 0, 0, 'MISSING'))
                continue
            with open(fpath, 'r') as f:
                reader = csv.reader(f)
                rows = [r for r in reader if r]
            # skip header if present
            if rows and not rows[0][0].replace('.','').replace('-','').isdigit():
                rows = rows[1:]
            n = len(rows)
            if n < 2:
                results.append((subj, ver, cam_name, n, 0, 'TOO_FEW'))
                continue
            # Column 1 is the timestamp (column 0 is frame index)
            ts_col = 1 if len(rows[0]) > 1 else 0
            t0 = float(rows[0][ts_col])
            t1 = float(rows[-1][ts_col])
            dur = t1 - t0
            if dur <= 0:
                results.append((subj, ver, cam_name, n, 0, 'BAD_DUR'))
                continue
            fps = (n - 1) / dur
            status = 'OK' if fps >= 25 else 'LOW_FPS'
            results.append((subj, ver, cam_name, n, round(fps, 2), status))

print(f'{"Subject":<20} {"Ver":<5} {"Camera":<15} {"Frames":>7} {"FPS":>8} {"Status":<10}')
print('-' * 70)
for subj, ver, cam, n, fps, st in results:
    print(f'{subj:<20} {ver:<5} {cam:<15} {n:>7} {fps:>8.2f} {st:<10}')

print('\n--- Summary ---')
for cam_name in cam_files:
    fps_vals = [r[4] for r in results if r[2] == cam_name and r[5] in ('OK', 'LOW_FPS')]
    if fps_vals:
        avg = sum(fps_vals) / len(fps_vals)
        mn, mx = min(fps_vals), max(fps_vals)
        low = sum(1 for v in fps_vals if v < 25)
        print(f'{cam_name}: avg={avg:.2f} fps, range=[{mn:.2f}, {mx:.2f}], low_fps_count={low}/{len(fps_vals)}')
