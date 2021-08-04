
import os
import logging
import oct2py
from pesq import pesq
#import pypesq
from scipy.io import wavfile

logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "composite.m")


def pesq_mos(clean: str, enhanced: str):
    sr1, clean_wav = wavfile.read(clean)
    sr2, enhanced_wav = wavfile.read(enhanced)
    assert sr1 == sr2
    mode = "nb" if sr1 < 16000 else "wb"
    #print("pesq1 : ",pesq(sr1, clean_wav, enhanced_wav, mode))
    #print("pesq2 : ",pypesq.pesq(clean_wav, enhanced_wav, sr1))
    return pesq(sr1, clean_wav, enhanced_wav, mode)


def composite(clean: str, enhanced: str):
    pesq_score = pesq_mos(clean, enhanced)
    csig, cbak, covl, ssnr = oc.feval(COMPOSITE, clean, enhanced, nout=4)
    csig += 0.603 * pesq_score
    cbak += 0.478 * pesq_score
    covl += 0.805 * pesq_score
    return pesq_score, csig, cbak, covl, ssnr