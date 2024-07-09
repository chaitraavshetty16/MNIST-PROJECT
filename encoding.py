# imports
from snntorch import spikeplot as splt
from snntorch import spikegen
from IPython.display import HTML
import torch
import matplotlib.pyplot as plt
from network import data_loader

# Code adapted from :https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html

def Spiking():
    # Temporal Dynamics
    num_steps = 100
    # create vector filled with 0.5
    raw_vector = torch.ones(num_steps)*0.5
    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)
    print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
    # Generating a rate-coded sample of data.
    # Iterate through minibatches
    train_loader,_ = data_loader(True)
    data = iter(train_loader)
    data_it, targets_it = next(data)
    # Spiking Data
    spike_data = spikegen.rate(data_it, num_steps=num_steps)
    print(spike_data.size())
    spike_data_sample = spike_data[:, 0, 0]
    print(spike_data_sample.size())
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    # The below is the path where ffmpeg application resides
    plt.rcParams['animation.ffmpeg_path'] = '/Users/Sreeram/Downloads/ffmpeg'
    HTML(anim.to_html5_video())
    anim.save("spike_mnist_test.mp4")
    print(f"The corresponding target is: {targets_it[0]}")
    spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
    spike_data_sample2 = spike_data[:, 0, 0]
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample2, fig, ax)
    HTML(anim.to_html5_video())
    plt.figure(facecolor="w")
    plt.subplot(1,2,1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 1')
    plt.savefig("SpikingData")
