import numpy as np
import numm
import time
import scipy.signal

def peak_indices(data, gap):
    """
    For now: A "peak" is defined as a datapoint which is the max
    in an interval +-gap around it.
    """

    peak_indices = set()
    for i in range(len(data)-gap):
        peak_indices.add(i+np.argmax(data[i:i+gap]))

    return set(i for i in peak_indices
               if data[i] == np.max(data[np.clip(i-gap,0,len(data)):np.clip(i+gap,0,len(data))]))

def golay(n):
    if n == 0:
        return ([1], [-1])
    else:
        a, b = golay(n-1)
        return (a + [-x for x in b], a + b)

play_pos = 0
to_play = None
end_play = 0
notes = set()
last = 0
fuck_mode = False
def audio_out(a):
    global play_pos, to_play, note, end_play
    
    chunk_size = a.shape[0]
    if to_play != None:
        print "PLAYING"
        end_pos = min(play_pos+chunk_size, to_play.shape[0])
        a[:end_pos-play_pos,:] = to_play[play_pos:end_pos,:]
        
        if end_pos == to_play.shape[0]:
            to_play = None
            play_pos = 0
            end_play = time.time()
        else:
            play_pos = end_pos
    elif len(notes) > 0:
        #return
        global last
        tone = np.zeros(len(a), np.int16)
        for note in notes:
            if fuck_mode: note = note+10
            tone += 32767*np.sin((last+np.arange(len(a)))*2*np.pi*note/44100)/len(notes)
        a[:,0] = tone
        a[:,1] = tone
        last += len(a)
        #a[::(44100/note)] = 32767
        
def take_measurement(out):
    global draw_color
    # first, wait for things to quiet down

    draw_color = [0, 255, 0]
    while std_over_mean > 0.2:
        time.sleep(0.1)

    draw_color = [255,0,0]
    global to_play, idx
    to_play = out
    start_recording()
    time.sleep(2*len(out)/44100)
    while (std_over_mean > 0.2
           or to_play != None or (time.time() - end_play) < 0.3):
        time.sleep(0.1)
    print to_play
    recording = end_recording()

    draw_color = [255]*3
    return recording[:,0].T

spectra = []

note_expansion = 50

golay_order = 15

rhs = np.zeros((240,160,3), np.uint8)

ffting = False

def correlate(x, y):
    return scipy.signal.fftconvolve(x, y[::-1])

class Scale:
    def __init__(self, in1, in2, out1, out2):
        self.in1, self.in2, self.out1, self.out2 = in1, in2, out1, out2
        self.slope = float(out2-out1)/(in2-in1)
        self.out_min = min(out1, out2)
        self.out_max = max(out1, out2)

    def __call__(self, x):
        return np.clip(self.out1 + self.slope*(x-self.in1),
                       self.out_min, self.out_max)
    
class ProcessedPair:
    def __init__(self, a, b, golay_order):
        seq = golay(golay_order)
        a_corr = correlate(a, np.array(seq[0]))
        b_corr = correlate(b, np.array(seq[1]))
        a_corr_argmax = np.argmax(a_corr)
        b_corr_argmax = np.argmax(b_corr)
        a_corr_max = a_corr[a_corr_argmax]
        b_corr_max = b_corr[b_corr_argmax]
        a_corr_short_len = len(a_corr) - a_corr_argmax
        b_corr_short_len = len(b_corr) - b_corr_argmax
        self.short_len = min(a_corr_short_len, b_corr_short_len)
        self.a_corr_short = a_corr[a_corr_argmax:a_corr_argmax+self.short_len]
        self.b_corr_short = b_corr[b_corr_argmax:b_corr_argmax+self.short_len]
        self.sum_corr = self.a_corr_short + self.b_corr_short
        print self.sum_corr

        self.spectrum = np.abs(np.fft.rfft(self.sum_corr))**2
        self.spectrum_max = self.spectrum.max()

def plot_pp(pair):
    global rhs

    rhs[:] = 0
    
    x_scale = Scale(0, pair.short_len-1, 0, 159)
    y_scale = Scale(-100000, 100000, 0, 239)
    for i in range(0, pair.short_len, 50):
        x = x_scale(i)
        rhs[y_scale(pair.sum_corr[i]):120,x,:] = [255]*3
        rhs[120:y_scale(pair.sum_corr[i]),x,:] = [255]*3
        rhs[y_scale(pair.a_corr_short[i]),x,:] = [255,0,0]
        rhs[y_scale(pair.b_corr_short[i]),x,:] = [0,255,0]

    rhs = rhs/2
        
    x_scale = Scale(0, (len(pair.spectrum)-1)/note_expansion, 0, 159)
    y_scale = Scale(0, 100*np.mean(pair.spectrum), 239, 0)
    for i in range(0,len(pair.spectrum)):
        x = x_scale(i)
        rhs[y_scale(pair.spectrum[i]):240,x,:] = [0,0,255]

    global peaks, len_spectrum
    len_spectrum = len(pair.spectrum)
    peaks = sorted(peak_indices(pair.spectrum[:len(pair.spectrum)/note_expansion], 100))
    for i in peaks:
        x = x_scale(i)
        rhs[y_scale(pair.spectrum[i]):240,x,:] = [0,255,255]

pair = ([],[])
pair_golay_order = 0

def keyboard_in(type, key):
    global golay_order, notes, fuck_mode
    tone_keys = 'qwertyuiop'
    if key in tone_keys:
        i = [i for i, x in enumerate(tone_keys) if x == key][0]
        note = 22050*float(peaks[i])/len_spectrum
        if type == 'key-release':
            #notes.remove(note)

            if note in notes:
                notes.remove(note)
            else:
                notes.add(note)

        elif type == 'key-press':
            pass
            #notes.add(note)
        print notes
    elif key == 'f' and type == 'key-release':
        fuck_mode = not fuck_mode
        print "fuck_mode =", fuck_mode
    elif key == 'm' and type == 'key-release':
        global rhs, spectra
        rhs = np.zeros((240,160,3), np.uint8)
        spectra = []
    elif key == '1' and type == 'key-release':
        golay_order -= 1
        print "golay_order = %i" % golay_order
    elif key == '2' and type == 'key-release':
        golay_order += 1
        print "golay_order = %i" % golay_order
    elif key == 'x' and type == 'key-release':
        seq = golay(golay_order)
        a = take_measurement(np.array([seq[0]]*2, dtype=np.int16).T * 32767)
        time.sleep(0.2)
        b = take_measurement(np.array([seq[1]]*2, dtype=np.int16).T * 32767)

        global pair, pair_golay_order
        pair = (a,b)
        ppair = ProcessedPair(a, b, golay_order)
        plot_pp(ppair)
        pair_golay_order = golay_order
    elif key == 'c' and type == 'key-release':
        a, b = pair
        np.savez("pair", a=pair[0], b=pair[1], golay_order=pair_golay_order)
    elif key == 'v' and type == 'key-release':
        global pair, pair_golay_order
        f = np.load("pair.npz")
        a, b, golay_order = f['a'], f['b'], f['golay_order']
        pair = (a,b)
        ppair = ProcessedPair(a, b, golay_order)
        plot_pp(ppair)
        
        
draw_color = [255]*3

idx = 0
recent_audio = np.zeros(770, np.int16)
recent_video = np.zeros((240,160,3), np.uint8)

recent_norms = np.zeros(30)
std_over_mean = 0

def video_out(a):
    global idx, recent_norms, std_over_mean
    norm = np.linalg.norm(recent_audio)
    if np.isnan(norm):
        norm = 100000
    recent_norms = np.roll(recent_norms, -1)
    recent_norms[-1] = norm
    std_over_mean = np.std(recent_norms)/np.mean(recent_norms)
    if np.isnan(std_over_mean): std_over_mean = 0

    recent_video[:,idx,:] = 0

    
    spot = np.clip(239-30*(np.log(norm)-4),0,239)
    spot2 = np.clip(239-30*std_over_mean,0,239)
    recent_video[spot:240,idx,:] = draw_color
    recent_video[spot2,idx,:] = [0,0,255]
    idx = (idx + 1) % recent_video.shape[1]
    a[:,:recent_video.shape[1],:] = np.roll(recent_video, -idx, axis=1)
    if ffting:
        spectrum = np.abs(np.fft.rfft(recent_audio))**2
        x_scale = Scale(0, (len(spectrum)-1)/note_expansion, 0, 159)
        y_scale = Scale(0, 1000000000, 239, 0)
        for i in range(0,len(spectrum)):
            x = x_scale(i)
            a[y_scale(spectrum[i]):240,x+160:x+160+x_scale.slope,:] = [0,0,128]
    else:
        a[:,160:,:] = rhs

is_recording = False
record = []

def start_recording():
    global record, is_recording
    record = []
    is_recording = True

def end_recording():
    global is_recording
    is_recording = False
    return np.concatenate(record)

def audio_in(a):
    global record, recent_audio
    if recent_audio.shape != a.shape:
        recent_audio = np.zeros(a.shape[0], np.int16)
    recent_audio[:] = np.roll(recent_audio, -len(a))
    recent_audio[:len(a)] = a.mean(axis=1)
    if is_recording:
        record.append(a)

def mouse_in(t, x, y, b):
    global note, ffting
    if x <= 0.5:
        notes = set()
    else:
        notes = set([22050*(x-0.5)/0.5/note_expansion])

    if t == 'mouse-button-press':
        ffting = True
    elif t == 'mouse-button-release':
        ffting = False
