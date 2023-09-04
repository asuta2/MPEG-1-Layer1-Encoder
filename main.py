import os
import wave
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import pyaudio as pa
import threading

import psychoacoustic_model as psycho
from utils import *

CHUNK = 1024
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pa.PyAudio()
frames = []


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)

    def on_enter(self, event):
        self.widget.tooltip = tk.Toplevel()
        self.widget.tooltip.geometry(
            "{}x{}+{}+{}".format(self.calculate_max_width(self.text) + 2, 15, event.x_root, event.y_root))
        self.widget.tooltip.overrideredirect(True)
        self.widget.tooltip_label = tk.Label(self.widget.tooltip, text=self.text, border=1, relief="solid")
        self.widget.tooltip_label.pack()

    def on_leave(self, event):
        self.widget.tooltip.destroy()

    def calculate_max_width(self, text):
        font_object = tkfont.Font(font=self.widget.cget("font"))
        return max(font_object.measure(line) for line in text.split('\n'))


def create_tooltip(widget, text):
    return Tooltip(widget, text)


class MainApplication(tk.Frame):
    window_height = 500
    window_width = 460

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.LabelFrame1 = None
        self.parent = parent
        # Set window size
        self.parent.geometry(str(self.window_width) + "x" + str(self.window_height) + "+" + str(
            self.get_window_x()) + "+" + str(self.get_window_y()))
        self.parent.resizable(False, False)
        self.parent.title("MPEG-1 Layer 1 Encoder GUI")

        self.label_frame_top = tk.LabelFrame(self.parent, text="Record audio and encode")
        self.label_frame_top.pack(side=tk.TOP, padx=10, pady=30, fill=tk.X)
        self.label_frame = tk.LabelFrame(self.parent, text="Or select a file to encode")
        self.label_frame.pack(side=tk.TOP, padx=10, pady=30)

        # Create a record button that will record audio and save it to a file
        self.top_frame = tk.Frame(self.label_frame_top)
        self.choose_source_directory_frame = tk.Frame(self.label_frame)
        self.choose_output_directory_frame = tk.Frame(self.label_frame)
        self.middle_frame = tk.Frame(self.label_frame)
        self.bottom_frame = tk.Frame(self.label_frame)

        self.top_frame.pack(side=tk.TOP)
        self.choose_source_directory_frame.pack(side=tk.TOP)
        self.choose_output_directory_frame.pack(side=tk.TOP)
        self.bottom_frame.pack(side=tk.BOTTOM)
        self.middle_frame.pack(side=tk.BOTTOM)

        # Section for recording audio and encoding it
        self.record_button = tk.Button(self.top_frame, text="Record", width=10, height=2, command=record)
        create_tooltip(self.record_button, "Record audio for 5 seconds.")
        self.save_button = tk.Button(self.top_frame, text="Save", width=10, height=2, command=save_audio)
        create_tooltip(self.save_button, "Save the recorded audio to a file.")
        self.play_audio_button = tk.Button(self.top_frame, text="Play", width=10, height=2, command=play_audio)
        create_tooltip(self.play_audio_button, "Play the recorded audio.")

        # Change the button to be pointer on hover
        self.record_button.config(cursor="hand2")
        self.save_button.config(cursor="hand2")
        self.play_audio_button.config(cursor="hand2")

        self.record_button.pack(side=tk.LEFT, padx=5, pady=10)
        self.play_audio_button.pack(side=tk.LEFT, padx=5, pady=10)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Section for choosing the input file
        self.input_file_label = tk.Label(self.choose_source_directory_frame, text="Input:")
        self.label_choose_input_directory = tk.Label(self.choose_source_directory_frame, text="", width=45,
                                                     height=1,
                                                     border=1, relief='solid', bg='#f0f0f0')
        self.choose_input_button = tk.Button(self.choose_source_directory_frame, text='Open', height=1, width=10,
                                             command=self.directory_pick_input)
        create_tooltip(self.choose_input_button, "Choose a file to encode.")

        self.input_file_label.pack(side=tk.LEFT, padx=4, pady=10)
        self.label_choose_input_directory.pack(side=tk.LEFT, padx=6, pady=10)
        self.choose_input_button.pack(side=tk.RIGHT, padx=5, pady=10)

        # Section for choosing the output directory
        self.output_directory_label = tk.Label(self.choose_output_directory_frame, text="Output:")
        self.label_choose_output_directory = tk.Label(self.choose_output_directory_frame, text="", width=45,
                                                      height=1,
                                                      border=1, relief='solid', highlightbackground='light grey')
        self.choose_output_button = tk.Button(self.choose_output_directory_frame, text='Open', height=1, width=10,
                                              command=self.directory_pick_output)
        create_tooltip(self.choose_output_button, "Choose a directory to save the encoded file.")

        self.output_directory_label.pack(side=tk.LEFT, pady=10)
        self.label_choose_output_directory.pack(side=tk.LEFT, padx=7, pady=10)
        self.choose_output_button.pack(side=tk.RIGHT, padx=5, pady=10)

        # Section for choosing the bitrate
        self.label_bitrate = tk.Label(self.middle_frame, text="Bitrate:")

        # Create a combobox that will allow the user to select the bitrate
        self.bitrate_combobox = ttk.Combobox(self.middle_frame, width=20, height=20)
        self.bitrate_combobox['values'] = (64, 96, 128, 160, 192, 224, 256, 288, 320)
        self.bitrate_combobox.current(8)  # set default value to 320

        self.label_bitrate.pack(side=tk.LEFT, padx=10, pady=10)
        self.bitrate_combobox.pack(side=tk.LEFT, padx=10, pady=10)

        self.pb = ttk.Progressbar(self.bottom_frame, orient="horizontal", length=460, mode="determinate")
        self.start_button = tk.Button(self.parent, text="Start", width=20, height=3,
                                      command=lambda: self.button_callback())
        self.start_button.config(cursor="hand2")

        self.pb.pack(side=tk.TOP, padx=10, pady=10)
        self.start_button.pack(side=tk.TOP, padx=10, pady=20)

    def get_window_y(self):
        return (root.winfo_screenheight() - self.window_height) // 2

    def get_window_x(self):
        return (root.winfo_screenwidth() - self.window_width) // 2

    def directory_pick_output(self):
        directory = filedialog.asksaveasfilename(initialdir="./", title="Select file",
                                                 filetypes=(("MP3 files", "*.mp3"), ("all files", "*.*")))
        if directory:
            if not directory.endswith('.mp3'):
                directory += '.mp3'
            self.label_choose_output_directory.config(text=directory)

    def directory_pick_input(self):
        directory = filedialog.askopenfilename(initialdir="./", title="Select file",
                                               filetypes=(("WAVE files", "*.wav"), ("all files", "*.*")))
        if directory:
            self.label_choose_input_directory.config(text=directory)

    def button_callback(self):
        thread = threading.Thread(target=self.main(self.label_choose_input_directory.cget("text"),
                                                   self.label_choose_output_directory.cget("text"),
                                                   int(self.bitrate_combobox.get())))
        thread.start()

    def main(self, inputmp3file, outputmp3file, bitrate):

        # Reset the progress bar
        self.pb['value'] = 0
        self.pb.update()

        # Check if the strings are empty
        if inputmp3file == "" or outputmp3file == "":
            messagebox.showerror("Error", "Please select an input and output file.")
            return
        # Check if output file already exists
        if os.path.exists(outputmp3file):
            messagebox.showerror("Error", "Output file already exists.")
            return

        # Read WAVE file and set MPEG encoder parameters.
        try:
            input_buffer = WavRead(inputmp3file)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading input file: {e}")
            return
        try:
            params = EncoderParameters(input_buffer.fs, input_buffer.nch, bitrate)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating encoder parameters: {e}")
            return
        
        self.parent.update()
        
        # Precompute the cosine window
        n = np.arange(512)
        cosine_window = np.cos((2 * np.arange(32)[:, None] + 1) * (n - 16) * np.pi / 64)

        # Compute the filterbank using broadcasting
        filterbank = cosine_window * filter_coeffs()[None, :]

        # Allocate the subband_samples array
        subband_samples = np.zeros((params.nch, 32, 12), dtype='float32')

        while input_buffer.nprocessed_samples < input_buffer.nsamples:

            # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
            for frm in range(FRAMES_PER_BLOCK):
                samples_read = input_buffer.read_samples(SHIFT_SIZE)

                # If all samples have been read, perform zero padding.
                if samples_read < SHIFT_SIZE:
                    zeros_array = np.zeros((params.nch, SHIFT_SIZE - samples_read))
                    for ch in range(params.nch):
                        input_buffer.audio[ch].insert(zeros_array[ch])

                # Filtering = dot product with reversed buffer.
                for ch in range(params.nch):
                    subband_samples[ch, :, frm] = np.dot(filterbank, input_buffer.audio[ch].reversed())

            # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
            # Number of bits allocated in subband is either 0 or in range [2,15].
            scfindices = np.zeros((params.nch, N_SUBBANDS), dtype='uint8')
            subband_bit_allocation = np.zeros((params.nch, N_SUBBANDS), dtype='uint8')

            # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although
            # scaling is done later, its result is necessary for the psychoacoustic model and calculation of
            # sound pressure levels.
            for ch in range(params.nch):
                scfindices[ch, :] = get_scalefactors(subband_samples[ch, :, :], params.table.scalefactor)
                subband_bit_allocation[ch, :] = psycho.model1(input_buffer.audio[ch].ordered(), params, scfindices)

            # Scaling subband samples with determined scalefactors.
            subband_samples /= params.table.scalefactor[scfindices][:, :, np.newaxis]

            # Update the progress bar with the percentage of the file that has been processed.
            self.pb["value"] = input_buffer.nprocessed_samples / input_buffer.nsamples * 100
            self.parent.update()

            # Subband samples quantization. Multiplication with coefficients 'a' and adding coefficients 'b' is
            # defined in the ISO standard.
            for ch in range(params.nch):
                for sb in range(N_SUBBANDS):
                    if subband_bit_allocation[ch, sb] != 0:
                        subband_samples[ch, sb, :] *= params.table.qca[subband_bit_allocation[ch, sb] - 2]
                        subband_samples[ch, sb, :] += params.table.qcb[subband_bit_allocation[ch, sb] - 2]
                        subband_samples[ch, sb, :] *= 1 << subband_bit_allocation[ch, sb] - 1

            # Since subband_samples is a float array, it needs to be cast to unsigned integers.
            subband_samples_quantized = subband_samples.astype('uint32')

            # Forming output bitsream and appending it to the output file.
            bitstream_formatting(outputmp3file,
                                 params,
                                 subband_bit_allocation,
                                 scfindices,
                                 subband_samples_quantized)
        self.pb["value"] = 100
        # Display pop up window when finished
        messagebox.showinfo("Finished", "File saved at " + outputmp3file)


def record():
    # If audio was already previously recorded, clear it
    if frames:
        frames.clear()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK))
    print("Finished recording.")
    stream.stop_stream()
    stream.close()


def save_audio():
    # If no audio was recorded, display error message box
    if not frames:
        messagebox.showerror("Error", "No audio to save.")
        return

    audio.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    frames.clear()
    messagebox.showinfo("Success", f"Audio saved to {WAVE_OUTPUT_FILENAME}")


def stop_recording():
    # If the user wants to end the recording early, save accumulated frames and terminate the audio
    if len(frames) == 0:
        print("No audio to save.")
        return
    audio.terminate()


def play_audio():
    # Play audio from the frames that were recorded, not from a file
    if len(frames) == 0:
        # Display error message box
        messagebox.showerror("Error", "No audio to play.")
        return
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    for frame in frames:
        stream.write(frame)
    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
