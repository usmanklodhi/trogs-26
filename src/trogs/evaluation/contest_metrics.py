"""Ground-truth reading order and character error rate, as in the competition evaluation script.

``Box``, ``convert``, ``read_box_file`` and ``get_row_boxes`` follow the
official ``contest_evaluation.py`` closely so CER numbers stay comparable; only
names and formatting were tidied.
"""

from __future__ import annotations

import math

import Levenshtein
import numpy as np


class Box:
    """Letter box given by its north-west, south-west, south-east and north-east corners."""

    def __init__(self, xnw, ynw, xsw, ysw, xse, yse, xne, yne):
        self.xnw, self.ynw = xnw, ynw
        self.xsw, self.ysw = xsw, ysw
        self.xse, self.yse = xse, yse
        self.xne, self.yne = xne, yne

    def bd(self):
        return [self.xnw, self.ynw, self.xsw, self.ysw, self.xse, self.yse, self.xne, self.yne]

    def center(self):
        return (
            (self.xnw + self.xne + self.xse + self.xsw) / 4,
            (self.ynw + self.yne + self.yse + self.ysw) / 4,
        )

    def radius(self):
        return (
            math.sqrt(
                (self.xnw - self.xse) * (self.xnw - self.xse) + (self.ynw - self.yse) * (self.ynw - self.yse)
            )
            + math.sqrt(
                (self.xne - self.xsw) * (self.xne - self.xsw) + (self.yne - self.ysw) * (self.yne - self.ysw)
            )
        ) / 4

    def area(self):
        # Uses the top and left sides, assuming an orthogonal rectangle.
        return math.sqrt(
            ((self.xne - self.xnw) ** 2 + (self.yne - self.ynw) ** 2)
            * ((self.xsw - self.xnw) ** 2 + (self.ysw - self.ynw) ** 2)
        )

    def angle(self):
        return math.atan2(self.yne - self.ynw, self.xne - self.xnw)

    def dist(self, box):
        c, c2 = self.center(), box.center()
        return math.sqrt((c[0] - c2[0]) * (c[0] - c2[0]) + (c[1] - c2[1]) * (c[1] - c2[1]))


def convert(stringin):
    # fmt: off
    # Unicode code point -> ASCII transliteration used by the competition ground truth.
    eqv = {
        913: "A", 7944: "A", 914: "B", 915: "G", 916: "D", 917: "E", 918: "Z", 919: "H", 920: "Q", 921: "I",
        922: "K", 923: "L", 924: "M", 925: "N", 926: "X", 927: "O", 928: "P", 929: "R", 931: "S", 932: "T",
        933: "U", 934: "F", 935: "Y", 936: "C", 937: "W", 945: "a", 8118: "a", 8119: "a", 7942: "a", 8070: "a",
        8071: "a", 8048: "a", 8049: "a", 7936: "a", 7937: "a", 7940: "a", 7941: "a", 7938: "a", 7939: "a", 946: "b",
        947: "g", 948: "d", 949: "e", 7952: "e", 7953: "e", 8050: "e", 8051: "e", 7956: "e", 7957: "e", 7954: "e",
        7955: "e", 950: "z", 951: "h", 7974: "h", 7975: "h", 8086: "h", 8087: "h", 8134: "h", 8135: "h", 7970: "h",
        7971: "h", 7968: "h", 7969: "h", 7972: "h", 7973: "h", 8052: "h", 8053: "h", 952: "q", 953: "i", 7990: "i",
        7991: "i", 8150: "i", 8054: "i", 8055: "i", 7984: "i", 7985: "i", 7988: "i", 7989: "i", 7986: "i", 7987: "i",
        954: "k", 955: "l", 956: "m", 957: "n", 834: "n", 958: "x", 959: "o", 8056: "o", 8057: "o", 8000: "o",
        8001: "o", 8004: "o", 8005: "o", 8002: "o", 8003: "o", 960: "p", 961: "r", 962: "v", 963: "s", 964: "t",
        965: "u", 8022: "u", 8023: "u", 8166: "u", 8016: "u", 8017: "u", 8020: "u", 8021: "u", 8018: "u", 8019: "u",
        8058: "u", 8059: "u", 966: "f", 967: "y", 968: "c", 969: "w", 8102: "w", 8103: "w", 8038: "w", 8039: "w",
        8182: "w", 8183: "w", 8060: "w", 8061: "w", 8032: "w", 8033: "w", 8036: "w", 8037: "w", 8034: "w", 8035: "w",
        8217: "'", 8025: "`", 803: "_", 8195: " ", 903: ".", 8228: ".", 8311: "7", 10: "\n", 3: " ", 32: " ",
    }
    # fmt: on
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


def read_box_file(fname):
    angle = None
    boxes = []
    transcript = []
    lines = []
    try:
        with open(fname) as f:
            phase = 0
            count = 0
            for line in f:
                count = count + 1
                line2 = line.replace("*", "").replace(" ", "")
                if line2 == "" or line2 == "\n":
                    continue
                elif line[0] == "#":
                    phase = phase + 1
                elif phase == 0:
                    print(f"Header problem in file {fname}.")
                elif phase == 1:
                    if angle is None:
                        angle = float(line)
                    else:
                        print(f"Multiple angle data found in {fname}.")
                elif phase == 2:
                    n = line.split()
                    if len(n) == 8:
                        box = Box(*[float(c) for c in n])
                        if box.area() < 9:
                            print(f"Skipping box with very small area (line {count}).")
                        else:
                            boxes.append(box)
                    else:
                        print(f"Incorrect parameter count for box line ({len(n)}) in {fname}.")
                elif phase == 3:
                    transcript.append(
                        line.replace("*", "").replace("\n", "").replace("\r", "").replace(" ", "")
                    )
                elif phase == 4:
                    n = line.split()
                    if len(n) == 4:
                        lines.append([float(c) for c in n])
                        if np.any(np.isnan(lines[-1])):
                            print(f"Bad values found in line specification (file line {count}) -- skipping.")
                            lines.pop()
                        elif lines[-1][0] == lines[-1][2] and lines[-1][1] == lines[-1][3]:
                            print(
                                f"Zero-length line found in line specification (file line {count}) -- skipping."
                            )
                            lines.pop()
                    else:
                        print(f"Incorrect parameter count for line specifiers ({len(n)}) in {fname}.")
            f.close()
    except FileNotFoundError:
        print(f"File '{fname}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return angle, boxes, transcript, lines


def get_row_boxes(boxes, gangle=None, lines=()):
    if len(lines) == 0:
        nboxes = len(boxes)
        bclust = np.full((nboxes), -1)
        bseq = np.zeros(nboxes)
        lnbr = np.full((nboxes), -1)
        rnbr = np.full((nboxes), -1)
        xcent = [b.center()[0] for b in boxes]
        xsort = np.sort(xcent)
        xord = np.argsort(xcent)
        xlut = np.argsort(xord)
        for i in range(nboxes):
            ci = boxes[i].center()
            if gangle is None:
                angle = boxes[i].angle()
            else:
                angle = gangle
            ax = math.cos(angle)
            ay = math.sin(angle)
            rd = np.inf
            ld = np.inf
            klo = xlut[i]
            khi = klo
            assert ci[0] == xsort[klo]
            while min(ci[0] - xsort[klo], xsort[khi] - ci[0]) < max(rd, ld) and (klo > 0 or khi < nboxes - 1):
                if (ci[0] - xsort[klo] < xsort[khi] - ci[0] or khi >= nboxes - 1) and klo > 0:
                    klo = klo - 1
                    j = xord[klo]
                elif khi < nboxes - 1:
                    khi = khi + 1
                    j = xord[khi]
                else:
                    print("Should not reach this!")
                d = boxes[i].dist(boxes[j])
                cj = boxes[j].center()
                rj = boxes[j].radius()
                rjsq = rj * rj
                dcx = cj[0] - ci[0]
                dcy = cj[1] - ci[1]
                dotprod = dcx * ax + dcy * ay
                isRight = dotprod > 0
                if (i != j) and (d < ld) and not isRight:
                    c = (ci[0] - d * ax, ci[1] - d * ay)
                    if (cj[0] - c[0]) ** 2 + (cj[1] - c[1]) ** 2 < rjsq:
                        ld = d
                        lnbr[i] = j
                if (i != j) and (d < rd) and isRight:
                    c = (ci[0] + d * ax, ci[1] + d * ay)
                    if (cj[0] - c[0]) ** 2 + (cj[1] - c[1]) ** 2 < rjsq:
                        rd = d
                        rnbr[i] = j
        for i in range(len(rnbr)):
            if rnbr[i] >= 0 and lnbr[rnbr[i]] != i:
                if lnbr[rnbr[i]] >= 0 and rnbr[lnbr[rnbr[i]]] == rnbr[i]:
                    rnbr[i] = -2
            if lnbr[i] >= 0 and rnbr[lnbr[i]] != i:
                if rnbr[lnbr[i]] >= 0 and lnbr[rnbr[lnbr[i]]] == lnbr[i]:
                    lnbr[i] = -2
        for i in range(len(rnbr)):
            if rnbr[i] >= 0 and lnbr[rnbr[i]] != i:
                if lnbr[rnbr[i]] < 0:
                    lnbr[rnbr[i]] = i
                else:
                    print(f"Left of right mismatch at {i}: {rnbr[i]} to {lnbr[rnbr[i]]}.")
            if lnbr[i] >= 0 and rnbr[lnbr[i]] != i:
                if rnbr[lnbr[i]] < 0:
                    rnbr[lnbr[i]] = i
                else:
                    print(f"Right of left mismatch at {i}: {lnbr[i]} to {rnbr[lnbr[i]]}.")
        clust = 0
        cid = []
        cseq = []
        for i in range(nboxes):
            if gangle is None:
                angle = boxes[i].angle()
            else:
                angle = gangle
            if bclust[i] == -1:
                c = [i]
                bclust[i] = clust
                bseq[i] = boxes[i].center()[0] * math.cos(angle) + boxes[i].center()[1] * math.sin(angle)
                j = lnbr[i]
                while j >= 0:
                    if bclust[j] == -1:
                        c.append(j)
                        bclust[j] = clust
                        bseq[j] = boxes[j].center()[0] * math.cos(angle) + boxes[j].center()[1] * math.sin(
                            angle
                        )
                    j = lnbr[j]
                j = rnbr[i]
                while j >= 0:
                    if bclust[j] == -1:
                        c.append(j)
                        bclust[j] = clust
                        bseq[j] = boxes[j].center()[0] * math.cos(angle) + boxes[j].center()[1] * math.sin(
                            angle
                        )
                    j = rnbr[j]
                cid.append(c)
                cseq.append([bseq[k] for k in c])
                clust = clust + 1
        rowlist = []
        idlist = []
        for i in range(len(cid)):
            rowlist.append(sorted(cid[i], key=lambda x: bseq[x]))
            idlist.append(rowlist[-1])
        return rowlist, idlist


def get_row_transcript(fname):
    gangle, boxes, transcript, lines = read_box_file(fname)
    boxes2 = boxes  # assume order is fine
    rowlist, idlist = get_row_boxes(boxes2, gangle=gangle)
    lintran = "".join(transcript)
    return "".join(["".join([lintran[i] for i in rowid]) + "\n" for rowid in idlist])


def normalise_prediction(text: str) -> str:
    """Map a prediction onto the ground-truth alphabet: transliterate, upper-case, drop spaces/tabs."""
    return convert(text).upper().replace(" ", "").replace("\t", "")


def evaluate(prediction: str, gt: str) -> tuple[float, int, int]:
    """Return ``(CER, edit distance, ground-truth length)``."""
    n_error = Levenshtein.distance(normalise_prediction(prediction), gt)
    n_char = len(gt)
    return n_error / n_char, n_error, n_char


def edit_operations(prediction: str, gt: str) -> tuple[int, int, int]:
    """Count ``(insertions, deletions, substitutions)`` needed to turn ``gt`` into the prediction."""
    if not gt:
        return 0, 0, 0
    ops = Levenshtein.editops(gt.replace(" ", "").replace("\t", ""), normalise_prediction(prediction))
    return (
        sum(1 for op in ops if op[0] == "insert"),
        sum(1 for op in ops if op[0] == "delete"),
        sum(1 for op in ops if op[0] == "replace"),
    )
