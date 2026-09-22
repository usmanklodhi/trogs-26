import os
import glob
import pandas as pd
import math
import numpy as np
from Levenshtein import distance as levenshtein

# Copy necessary functions from contest_evaluation.py

import Levenshtein

def compute_edit_operations(pred, gt):
    if not gt: return 0, 0, 0
    
    proc_pred = convert(pred).upper().replace(' ','').replace('\t','')
    proc_gt = gt.replace(' ','').replace('\t','')
    
    # editops(source, destination) tells us how to turn GT into Pred
    ops = Levenshtein.editops(proc_gt, proc_pred)
    
    ins = sum(1 for op in ops if op[0] == 'insert')
    dels = sum(1 for op in ops if op[0] == 'delete')
    subs = sum(1 for op in ops if op[0] == 'replace')
    
    return ins, dels, subs

class Box:
    def __init__(self,xnw,ynw,xsw,ysw,xse,yse,xne,yne):
        self.xnw = xnw
        self.ynw = ynw
        self.xsw = xsw
        self.ysw = ysw
        self.xse = xse
        self.yse = yse
        self.xne = xne
        self.yne = yne
        
    def bd(self):
        return [self.xnw,self.ynw,self.xsw,self.ysw,self.xse,self.yse,self.xne,self.yne]

    def center(self):
        return((self.xnw+self.xne+self.xse+self.xsw)/4,(self.ynw+self.yne+self.yse+self.ysw)/4)
    
    def radius(self):
        return (math.sqrt((self.xnw-self.xse)*(self.xnw-self.xse)+(self.ynw-self.yse)*(self.ynw-self.yse))+math.sqrt((self.xne-self.xsw)*(self.xne-self.xsw)+(self.yne-self.ysw)*(self.yne-self.ysw)))/4
    
    def area(self):
        # uses top and left side, assuming orthogonal rectangle
        return math.sqrt(((self.xne-self.xnw)**2+(self.yne-self.ynw)**2)*((self.xsw-self.xnw)**2+(self.ysw-self.ynw)**2))

    def angle(self):
        return math.atan2(self.yne-self.ynw,self.xne-self.xnw)

    def dist(self,box):
        c = self.center()
        c2 = box.center()
        return math.sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]))

def convert(stringin):
    eqv = {
        913:"A", 7944:"A", 914:"B", 915:"G", 916:"D", 917:"E", 918:"Z", 919:"H", 920:"Q", 921:"I", 922:"K", 923:"L", 924:"M", 925:"N", 926:"X", 927:"O", 928:"P", 929:"R", 931:"S", 932:"T", 933:"U", 934:"F", 935:"Y", 936:"C", 937:"W", 945:"a", 8118:"a", 8119:"a", 7942:"a", 8070:"a", 8071:"a", 8048:"a", 8049:"a", 7936:"a", 7937:"a", 7940:"a", 7941:"a", 7938:"a", 7939:"a", 946:"b", 947:"g", 948:"d", 949:"e", 7952:"e", 7953:"e", 8050:"e", 8051:"e", 7956:"e", 7957:"e", 7954:"e", 7955:"e", 950:"z", 951:"h", 7974:"h", 7975:"h", 8086:"h", 8087:"h", 8134:"h", 8135:"h", 7970:"h", 7971:"h", 8016:"h", 8020:"h", 7968:"h", 7969:"h", 7972:"h", 7973:"h", 8052:"h", 8053:"h", 952:"q", 953:"i", 7990:"i", 7991:"i", 8150:"i", 8054:"i", 8055:"i", 7984:"i", 7985:"i", 7988:"i", 7989:"i", 7986:"i", 7987:"i", 954:"k", 955:"l", 956:"m", 957:"n", 834:"n", 958:"x", 959:"o", 8056:"o", 8057:"o", 8000:"o", 8001:"o", 8004:"o", 8005:"o", 8002:"o", 8003:"o", 960:"p", 961:"r", 962:"v", 963:"s", 964:"t", 965:"u", 8022:"u", 8023:"u", 8166:"u", 8016:"u", 8017:"u", 8020:"u", 8021:"u", 8018:"u", 8019:"u", 8058:"u", 8059:"u", 966:"f", 967:"y", 968:"c", 969:"w", 8102:"w", 8103:"w", 8038:"w", 8039:"w", 8182:"w", 8183:"w", 8060:"w", 8061:"w", 8032:"w", 8033:"w", 8036:"w", 8037:"w", 8034:"w", 8035:"w", 8217:"'", 8025:"`", 803:"_", 8195:" ", 903:".", 8228:".", 8311:"7", 10:"\n", 3:" ", 32:" "
    }
    for i in range(0, 254):
        eqv[i] = chr(i)
            
    txt = []
    for char in stringin:
        if ord(char) in eqv:
            txt.append(eqv[ord(char)])
        else:
            txt.append(char)               
    txt = "".join(txt)
    return txt.upper()

def readBoxFile(fname):
    angle = None
    boxes = []
    transcript = []
    lines = []
    try:
        with open(fname, "r") as f:
            phase = 0
            count = 0
            for line in f:
                count = count+1
                line2 = line.replace('*','').replace(' ','')
                if line2=='' or line2=='\n':
                    continue
                elif line[0]=='#':
                    phase = phase+1
                elif phase==0:
                    print("Header problem in file {}.".format(fname))
                elif phase==1:
                    if angle==None:
                        angle = float(line)
                    else:
                        print("Multiple angle data found in {}.".format(fname))
                elif phase==2:
                    n = line.split()
                    if len(n)==8:
                        box = Box(*[float(c) for c in n])
                        if box.area() < 9:
                            print("Skipping box with very small area (line {}).".format(count))
                        else:
                            boxes.append(box)
                    else:
                        print('Incorrect parameter count for box line ({}) in {}.'.format(len(n),fname))   
                elif phase==3:
                    transcript.append(line.replace('*','').replace('\n','').replace('\r','').replace(' ',''))
                elif phase==4:
                    n = line.split()
                    if len(n)==4:
                        lines.append([float(c) for c in n])
                        if np.any(np.isnan(lines[-1])):
                            print('Bad values found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                        elif lines[-1][0]==lines[-1][2] and lines[-1][1]==lines[-1][3]:
                            print('Zero-length line found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                    else:
                        print('Incorrect parameter count for line specifiers ({}) in {}.'.format(len(n),fname))   
            f.close()
    except FileNotFoundError:
        print(f"File '{fname}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return angle,boxes,transcript,lines

def getRowBoxes(boxes,gangle=None,lines=[]):
    if len(lines)==0:
        nboxes = len(boxes)
        bclust = np.full((nboxes),-1)
        bseq = np.zeros((nboxes))
        lnbr = np.full((nboxes),-1)
        rnbr = np.full((nboxes),-1)
        xcent = [b.center()[0] for b in boxes]
        xsort = np.sort(xcent)
        xord = np.argsort(xcent)
        xlut = np.argsort(xord)
        for i in range(nboxes):
            ci = boxes[i].center()
            if (gangle==None):
                angle = boxes[i].angle()
            else:
                angle = gangle
            ax = math.cos(angle)
            ay = math.sin(angle)
            rd = np.inf
            ld = np.inf
            klo = xlut[i]
            khi = klo
            assert(ci[0] == xsort[klo])
            while min(ci[0]-xsort[klo],xsort[khi]-ci[0]) < max(rd,ld) and (klo>0 or khi < nboxes-1):
                if (ci[0]-xsort[klo] < xsort[khi]-ci[0] or khi >= nboxes-1) and klo > 0:
                    klo = klo-1
                    j = xord[klo]
                elif khi < nboxes-1:
                    khi = khi+1
                    j = xord[khi]
                else:
                    print('Should not reach this!')
                d = boxes[i].dist(boxes[j])
                cj = boxes[j].center()
                rj = boxes[j].radius();
                rjsq = rj*rj
                dcx = cj[0]-ci[0]
                dcy = cj[1]-ci[1]
                dotprod = dcx*ax+dcy*ay
                isRight = dotprod > 0
                if (i!=j) and (d<ld) and not isRight:
                    c = (ci[0]-d*ax, ci[1]-d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        ld = d
                        lnbr[i] = j
                if (i!=j) and (d<rd) and isRight:
                    c = (ci[0]+d*ax, ci[1]+d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        rd = d
                        rnbr[i] = j
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]] >= 0 and rnbr[lnbr[rnbr[i]]] == rnbr[i]:
                    rnbr[i] = -2
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]] >= 0 and lnbr[rnbr[lnbr[i]]] == lnbr[i]:
                    lnbr[i] = -2
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]]<0:
                    lnbr[rnbr[i]] = i
                else:
                    print('Left of right mismatch at {}: {} to {}.'.format(i,rnbr[i],lnbr[rnbr[i]]))
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]]<0:
                    rnbr[lnbr[i]] = i
                else:
                    print('Right of left mismatch at {}: {} to {}.'.format(i,lnbr[i],rnbr[lnbr[i]]))
        clust = 0
        cid = []
        cseq = []
        for i in range(nboxes):
            if (gangle==None):
                angle = boxes[i].angle()
            else:
                angle = gangle
            if (bclust[i] == -1):
                c = [i]
                bclust[i] = clust
                bseq[i] = boxes[i].center()[0]*math.cos(angle)+boxes[i].center()[1]*math.sin(angle)
                j = lnbr[i]
                while j >= 0:
                    if bclust[j] == -1:
                        c.append(j)
                        bclust[j] = clust
                        bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = lnbr[j]
                j = rnbr[i]
                while j >= 0:
                    if bclust[j] == -1:
                        c.append(j)
                        bclust[j] = clust
                        bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = rnbr[j]
                cid.append(c)
                cseq.append([bseq[k] for k in c])
                clust = clust+1
        rowlist = []
        idlist = []
        for i in range(len(cid)):
            rowlist.append(sorted(cid[i], key=lambda x: bseq[x]))
            idlist.append(rowlist[-1])
        return rowlist, idlist

def getRowTranscript(fname):
    gangle, boxes, transcript, lines = readBoxFile(fname)
    boxes2 = boxes  # assume order is fine
    rowlist, idlist = getRowBoxes(boxes2,gangle=gangle)
    lintran = ''.join(transcript)
    return ''.join([''.join([lintran[i] for i in rowid])+'\n' for rowid in idlist])

def evaluate(prediction,gt):
    n_error = levenshtein(convert(prediction).upper().replace(' ','').replace('\t',''),gt)
    n_char = len(gt)
    cer = n_error/n_char
    return cer, n_error, n_char


# data/raw/test_set.txt
# results/batch_run_20260402_164529/transcripts
# Paths
TEST_SET_FILE = "../data/raw/test_set.txt"
GT_DIR = "../data/raw/Annotations"
CRNN_PRED_DIR = "../outputs/crnn/batch_transcripts"
YOLO_PRED_DIR = "../results/batch_run_20260514_163624"
CRNN_DIAG_FILE = "../outputs/crnn/batch_transcripts/batch_diagnostics.tsv"
YOLO_DIAG_FILE = "../results/batch_run_20260514_163624/diagnostics.txt"

def load_test_squeezes():
    with open(TEST_SET_FILE, 'r') as f:
        squeezes = [line.strip() for line in f if line.strip()]
    return squeezes

def load_ground_truth(squeeze):
    gt_file = os.path.join(GT_DIR, f"{squeeze}_Rotation1_300dpi_letters.txt")
    print(f"Loading GT for {squeeze}: {gt_file}")
    if not os.path.exists(gt_file):
        print(f"GT file not found: {gt_file}")
        return None
    gt = getRowTranscript(gt_file)
    print(f"GT length: {len(gt)}, first 100 chars: {repr(gt[:100])}")
    return gt

def load_prediction(pred_dir, squeeze, is_crnn=True):
    if is_crnn:
        pred_file = os.path.join(pred_dir, f"{squeeze}_Rotation1_300dpi.txt")
    else:
        # For YOLO, find in subfolders
        pattern = os.path.join(pred_dir, "**", "transcripts", f"{squeeze}_Rotation1_300dpi_transcript.txt")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            pred_file = matches[0]  # Take the first match
        else:
            pred_file = None
    print(f"Loading pred for {squeeze} from {pred_dir}: {pred_file}")
    if not pred_file or not os.path.exists(pred_file):
        print(f"Pred file not found: {pred_file}")
        return ""
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred = f.read()
    print(f"Pred length: {len(pred)}, first 100 chars: {repr(pred[:100])}")
    return pred

def compute_cer(pred, gt):
    if not gt:
        print("No GT, returning 0.0")
        return 0.0
    processed_pred = convert(pred).upper().replace(' ','').replace('\t','')
    processed_gt = gt  # gt is already processed in getRowTranscript?
    print(f"Processed pred: {repr(processed_pred[:100])}")
    print(f"Processed gt: {repr(processed_gt[:100])}")
    cer, _, _ = evaluate(pred, gt)
    print(f"CER: {cer}")
    return cer

def load_diagnostics(diag_file):
    if not os.path.exists(diag_file):
        return []
    if diag_file.endswith('.txt'):
        # YOLO format
        df = pd.read_csv(diag_file, sep='\t')
        times = df['time_sec'].dropna().tolist()
    else:
        # CRNN format
        df = pd.read_csv(diag_file, sep='\t')
        times = df['Time_Sec'].dropna().tolist()
    return times

# def main():
    squeezes = load_test_squeezes()
    print(f"Evaluating {len(squeezes)} test squeezes")

    crnn_cers = []
    yolo_cers = []

    for squeeze in squeezes:
        gt = load_ground_truth(squeeze)
        if not gt:
            print(f"No GT for {squeeze}")
            continue

        crnn_pred = load_prediction(CRNN_PRED_DIR, squeeze, is_crnn=True)
        yolo_pred = load_prediction(YOLO_PRED_DIR, squeeze, is_crnn=False)

        crnn_cer = compute_cer(crnn_pred, gt)
        yolo_cer = compute_cer(yolo_pred, gt)

        crnn_cers.append(crnn_cer)
        yolo_cers.append(yolo_cer)

        print(f"{squeeze}: CRNN CER {crnn_cer:.4f}, YOLO CER {yolo_cer:.4f}")

    mean_crnn_cer = sum(crnn_cers) / len(crnn_cers) if crnn_cers else 0
    mean_yolo_cer = sum(yolo_cers) / len(yolo_cers) if yolo_cers else 0

    print(f"\nMean CRNN CER: {mean_crnn_cer:.4f}")
    print(f"Mean YOLO CER: {mean_yolo_cer:.4f}")

    # Speed
    crnn_times = load_diagnostics(CRNN_DIAG_FILE)
    yolo_times = load_diagnostics(YOLO_DIAG_FILE)

    if crnn_times:
        mean_crnn_time = sum(crnn_times) / len(crnn_times)
        print(f"Mean CRNN inference time: {mean_crnn_time:.2f} seconds")
    if yolo_times:
        mean_yolo_time = sum(yolo_times) / len(yolo_times)
        print(f"Mean YOLO inference time: {mean_yolo_time:.2f} seconds")

    if crnn_times and yolo_times:
        speedup = mean_yolo_time / mean_crnn_time
        print(f"CRNN is {speedup:.1f}x faster than YOLO")

def main():
    squeezes = load_test_squeezes()
    print(f"Evaluating {len(squeezes)} test squeezes")

    crnn_cers = []
    yolo_cers = []
    
    # --- NEW: Track Edit Operations ---
    tot_crnn_ins, tot_crnn_dels, tot_crnn_subs = 0, 0, 0
    tot_yolo_ins, tot_yolo_dels, tot_yolo_subs = 0, 0, 0

    for squeeze in squeezes:
        gt = load_ground_truth(squeeze)
        if not gt:
            print(f"No GT for {squeeze}")
            continue

        crnn_pred = load_prediction(CRNN_PRED_DIR, squeeze, is_crnn=True)
        yolo_pred = load_prediction(YOLO_PRED_DIR, squeeze, is_crnn=False)

        crnn_cer = compute_cer(crnn_pred, gt)
        yolo_cer = compute_cer(yolo_pred, gt)

        crnn_cers.append(crnn_cer)
        yolo_cers.append(yolo_cer)
        
        # --- NEW: Compute and accumulate edit operations ---
        c_ins, c_dels, c_subs = compute_edit_operations(crnn_pred, gt)
        tot_crnn_ins += c_ins
        tot_crnn_dels += c_dels
        tot_crnn_subs += c_subs
        
        y_ins, y_dels, y_subs = compute_edit_operations(yolo_pred, gt)
        tot_yolo_ins += y_ins
        tot_yolo_dels += y_dels
        tot_yolo_subs += y_subs

        print(f"{squeeze}: CRNN CER {crnn_cer:.4f}, YOLO CER {yolo_cer:.4f}")

    mean_crnn_cer = sum(crnn_cers) / len(crnn_cers) if crnn_cers else 0
    mean_yolo_cer = sum(yolo_cers) / len(yolo_cers) if yolo_cers else 0

    print(f"\nMean CRNN CER: {mean_crnn_cer:.4f}")
    print(f"Mean YOLO CER: {mean_yolo_cer:.4f}")
    
    # --- NEW: Print Edit Operation Stats ---
    print("\n--- Error Analysis (Edit Operations) ---")
    print(f"CRNN -> Insertions (Hallucinations): {tot_crnn_ins} | Deletions (Blindness): {tot_crnn_dels} | Substitutions: {tot_crnn_subs}")
    print(f"YOLO -> Insertions (Hallucinations): {tot_yolo_ins} | Deletions (Blindness): {tot_yolo_dels} | Substitutions: {tot_yolo_subs}")

    # Speed
    crnn_times = load_diagnostics(CRNN_DIAG_FILE)
    yolo_times = load_diagnostics(YOLO_DIAG_FILE)

    if crnn_times:
        mean_crnn_time = sum(crnn_times) / len(crnn_times)
        print(f"\nMean CRNN inference time: {mean_crnn_time:.2f} seconds")
    if yolo_times:
        mean_yolo_time = sum(yolo_times) / len(yolo_times)
        print(f"Mean YOLO inference time: {mean_yolo_time:.2f} seconds")

    if crnn_times and yolo_times:
        speedup = mean_yolo_time / mean_crnn_time
        print(f"CRNN is {speedup:.1f}x faster than YOLO")

if __name__ == "__main__":
    main()
