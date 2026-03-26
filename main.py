

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from lab1_proto import enframe, preemp, windowing, powerSpectrum, logMelSpectrum, cepstrum
    from lab1_tools import lifter

    # Load the input signal (you can replace this with your own signal)
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    samples = example['samples']

    samplingrate = example['samplingrate']  # typically 20000 Hz

    # Convert ms to samples
    winlen = 20 * samplingrate // 1000  # 20 ms window length
    winshift = 10 * samplingrate // 1000  # 10 ms window shift

    # Enframe the signal
    frames = enframe(samples, winlen, winshift)
    emphasised_frames = preemp(frames)
    windowed_frames, window = windowing(emphasised_frames)
    spec = powerSpectrum(windowed_frames, nfft=512)
    mspec, filters = logMelSpectrum(spec, samplingrate)
    mfcc = cepstrum(mspec, nceps=13)
    lmfcc = lifter(mfcc)

    plt.figure(figsize=(12, 6))
    plt.plot(window, label='Hamming Window')
    plt.savefig('hamming_window.png')
    
    plt.figure(figsize=(12, 6))
    plt.plot(filters.T, label='Mel Filterbank')
    plt.savefig('melFilterbank.png')
    
    # Plot using pcolormesh
    fig, axes = plt.subplots(8, 1, figsize=(10, 18))

    axes[0].plot(samples)
    axes[0].set_title('samples: speech samples')

    axes[1].pcolormesh(frames.T)
    axes[1].set_title('frames: enframed samples')

    axes[2].pcolormesh(emphasised_frames.T)
    axes[2].set_title('preemph: preemphasis')

    axes[3].pcolormesh(windowed_frames.T)
    axes[3].set_title('windowed: hamming window')

    axes[4].pcolormesh(spec.T)
    axes[4].set_title('spec: abs(FFT)^2')

    axes[5].pcolormesh(mspec.T)
    axes[5].set_title('mspec: Mel Filterbank')

    axes[6].pcolormesh(mfcc.T)
    axes[6].set_title('mfcc: MFCCs')

    axes[7].pcolormesh(lmfcc.T)
    axes[7].set_title('lmfcc: Liftered MFCCs')

    plt.tight_layout()
    plt.savefig('output.png')
    # Now you can proceed with further processing on windowed_frames, such as computing the power spectrum, log Mel spectrum, etc.
    
if __name__ == "__main__":
    main()