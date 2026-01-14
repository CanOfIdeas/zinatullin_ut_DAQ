import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import TerminalConfiguration

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import DE_alg_RRC

DEV = "Dev5"
TRIG_SRC = "PFI0"
AI_CH = ["ai0", "ai1"]
AO_CH = ["ao0"]
T_TO_CHRG = 0.05
SAMPLING_RATE = 100000
PRETRIGGER_SAMPLES_RATIO = 0.01
NO_OF_SAMPLES = int(T_TO_CHRG * SAMPLING_RATE)
AO_VS = 5

samples_buf = None #Initialize a buffer to hold the acquired samples in.



def initialize_device(dev, ai_channels, ao_channels, trig_src, no_of_samples, sampling_rate):
    global samples_buf

    # Initialize buffer for samples. This creates a 2D numpy array: a row for each of the input channels, each row holds no_of_samples data points.
    samples_buf = np.zeros((len(ai_channels), no_of_samples))

    ai_task = nidaqmx.Task("AI TASK") # Create a task, named "AI TASK"

    # Iterate over all input channels and add them to the ai_task
    for i in range(len(ai_channels)):
        ai_task.ai_channels.add_ai_voltage_chan(dev + "/" + ai_channels[i], ai_channels[i], terminal_config = TerminalConfiguration.RSE) # Voltage channel must be in the form <Device name>/<Analouge in channel>. More info: https://nidaqmx-python.readthedocs.io/en/latest/_modules/nidaqmx/_task_modules/ai_channel_collection.html#AIChannelCollection.add_ai_voltage_chan

    ai_task.timing.cfg_samp_clk_timing(sampling_rate, samps_per_chan=no_of_samples) # Set timing for task.

    ai_task.in_stream.auto_start = False # Forces to explicitly start task.

    reader = stream_readers.AnalogMultiChannelReader(ai_task.in_stream) # Creates a reader instance

    ao_task = nidaqmx.Task("AO TASK")
    for ao_channel in ao_channels:
        ao_task.ao_channels.add_ao_voltage_chan(dev + "/" + ao_channel, ao_channel)

    return ai_task, ao_task, reader

def start_tasks(ai_task, ao_task, reader, ao_Vs, t_to_dischrg=1):
    global samples_buf

    ai_task.start()

    ao_task.start()
    ao_task.write(ao_Vs)

    reader.read_many_sample(samples_buf, nidaqmx.constants.READ_ALL_AVAILABLE)
    ai_task.stop()
    time.sleep(t_to_dischrg)
    ao_task.write(0)
    time.sleep(t_to_dischrg)
    ao_task.stop()



ai_task, ao_task, reader = initialize_device(DEV, AI_CH, AO_CH, TRIG_SRC, NO_OF_SAMPLES, sampling_rate = SAMPLING_RATE) # Initialize the DAQ device


plt.ion()
xdata = np.linspace(0, T_TO_CHRG, NO_OF_SAMPLES)
line, = plt.plot(xdata, np.zeros_like(xdata))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

while 1:
    start_tasks(ai_task, ao_task, reader, AO_VS, T_TO_CHRG) # Start the tasks
    # Generate x-axis values: a numpy array with whole numbers in the range of [0, <length of samples buffer>)
    # divided by sampling rate gives us the point in time each sample was measured relative to the time the first sample was measured.
    xdata = (np.linspace(0, samples_buf.shape[1], samples_buf.shape[1]))/SAMPLING_RATE

    capacitor_chraging_curve = samples_buf[0] # Assume that the first measurment in the buffer is the measurment of voltage on the capacitor.

    # Differential evolution parameters
    popsize = 15
    its = 150
    mut = 0.3
    crossp=0.5
    bounds = [(22E3, 22E3), (0, 100000), (1E-6, 1E-6)]

    de_result = DE_alg_RRC.main(xdata, capacitor_chraging_curve, [popsize, its, mut, crossp, bounds, AO_VS]) # Run differential evolution

    # Extract the results of differential evolution
    params = de_result[0]
    # Each row of params contains values for R1, R2 and C. To plot the evolution of each of the values,
    # we transpose the array containing these results.
    param_iterations = params.T
    R1 = np.array([param_iterations[0]])
    R2 = np.array([param_iterations[1]])
    C = np.array([param_iterations[2]])
    err = np.array([de_result[1]])
    opt_curves = de_result[2]
    opt_curve = np.array([opt_curves[-1]]) # Select the last iteration of optimized curve to plot.

    curve_plot_y = np.zeros((samples_buf.shape[0] + 1, samples_buf.shape[1])) # Initialize a numpy array for plotting the optimized curve generated with DE and the actually measured data on the same plot.

    # Assign the elements from the measurement buffer to buffer created in the previous step.
    for i in range(samples_buf.shape[0]):
        curve_plot_y[i] = samples_buf[i]
    curve_plot_y[-1] = opt_curve # Append the optimized curve to the buffer to be plotted

    print('R2: ', R2[-1][-1], ' C: ', C[-1][-1])
    # Plot chargeing curve
    axs = plt.gca()
    plt.title('R2: ' + str(np.round(R2[-1][-1])) + ' C: ' + str(C[-1][-1]))
    line.set_xdata(xdata)
    line.set_ydata(opt_curve[0])
    axs.relim()
    axs.autoscale_view()
    plt.draw()
    plt.pause(0.01)  # small delay to update
    time.sleep(0.01)
    
ai_task.stop()
ao_task.write(0)
ao_task.stop()
ai_task.close()
ao_task.close()

plt.ioff()
plt.show()