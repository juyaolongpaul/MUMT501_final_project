# -*- coding: utf-8 -*-

'''
 * Copyright (C) 2015  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of pypYIN
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 * If you have any problem about this algorithm, I suggest you to contact: Matthias Mauch
 * m.mauch@qmul.ac.uk who is the original C++ version author of this algorithm
 *
 * If you want to refer this code, please consider this article:
 *
 * M. Mauch and S. Dixon,
 * “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions”,
 * in Proceedings of the IEEE International Conference on Acoustics,
 * Speech, and Signal Processing (ICASSP 2014), 2014.
 *
 * M. Mauch, C. Cannam, R. Bittner, G. Fazekas, J. Salamon, J. Dai, J. Bello and S. Dixon,
 * “Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency”,
 * in Proceedings of the First International Conference on Technologies for
 * Music Notation and Representation, 2015.
'''
from scipy.io.wavfile import read, write
import os, sys
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)
from sklearn.metrics import f1_score
import math
import essentia.standard as ess
import numpy
import matplotlib.pyplot as plt
def FreqtoNote(f, reference):
    if f>0:
        m = 12 * math.log2(f / reference) + 69
    else:
        m = 0
    return m


def MIDI_note_to_one(ori):
    new = numpy.copy(ori)
    for i, item in enumerate(ori):
        if int(item) != 0:
            new[i] = 1
        else:
            new[i] = 0
    return new


def evaluate_per_person(person_name, f1_total, accuracy_total, counter, f_result):
    f1_this_person = []
    accuracy_this_person = []
    for filename_gt in os.listdir('.'):
        #if counter >= 1: break
        filename, file_extension = os.path.splitext(filename_gt)
        #print(file_extension)
        if file_extension != '.pv': continue
        if person_name not in filename: continue
        counter += 1
        filename += '.wav'
        print(filename, 'ID:', counter)
        print(filename, 'ID:', counter, file=f_result)
    # filename = r'/mnt/c/Users/test/Documents/Github/pypYIN/MIR-1K/Wavfile/davidson_1_02.wav'
    # filename_gt = r'/mnt/c/Users/test/Documents/Github/pypYIN/MIR-1K/PitchLabel/davidson_1_02.pv'
        f_gt = open(filename_gt, 'r')
        pitch_values_MIDI_gt = f_gt.readlines()
        for i, item in enumerate(pitch_values_MIDI_gt):
            pitch_values_MIDI_gt[i] = float(item.strip())
        pitch_values_MIDI_gt = numpy.asarray(pitch_values_MIDI_gt)
        #filename = r'/mnt/c/Users/test/Documents/Github/pypYIN/Row_e_right_sie_male.wav'
        fs = 16000
        audio = ess.MonoLoader(filename=os.path.join(dir, 'MIR-1K', 'Wavfile', filename), sampleRate=fs, downmix="right")()
        ess.MonoWriter(sampleRate=fs, filename=os.path.join(dir, 'MIR-1K', 'WavfileVoiceOnly', filename[:-4] + 'VoiceOnly' + '.wav'))(audio)
        # data0 is the data from channel 0.
        write(filename,fs, audio)
        # initialise
        # loader = ess.EqloudLoader(filename=r'/mnt/c/Users/test/Documents/Github/pypYIN/MIR-1K/Wavfile/abjones_1_01.wav', sampleRate=44100)
        # audio = loader()
        #filename1 = '/mnt/c/Users/test/Documents/Github/pypYIN/src/testAudioLong.wav'
        frameSize = 1024
        hopSize = 16
        frameSize_gt = 640
        hopSize_gt = 320



        # frame-wise calculation
        #fs, audio = audio_read(filename1, formatsox=False)


        # rms mean
        # rms = []
        # for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        #     rms.append(RMS(frame, frameSize))
        # rmsMean = np.mean(rms)
        # print 'rmsMean', rmsMean
        # c++ Melodia
        # pitch_extractor = ess.PitchMelodia(frameSize=2048, hopSize=128)
        # pitch_values1, pitch_confidence1 = pitch_extractor(audio)
        # pitch_times = numpy.linspace(0.0, len(audio) / fs, len(pitch_values1))
        # c++ pYIN
        pitch_values2, pitch_confidence2 = ess.PitchYinProbabilistic(frameSize=frameSize, sampleRate=fs, outputUnvoiced="zero", hopSize=hopSize)(audio)
        #pitch_values2, pitch_confidence2 = ess.PitchYin(frameSize=hopSize, sampleRate=fs,
                                                                     # )(audio)
        # print(pitch)
        pitch_values_MIDI = numpy.copy(pitch_values2)
        pitch_times = numpy.linspace(0.0, len(audio) / fs, len(pitch_values2))
        pitch_times_gt = numpy.linspace(0.0, len(audio) / fs, len(pitch_values_MIDI_gt))
        for i, value in enumerate(pitch_values2):
            pitch_values_MIDI[i] = FreqtoNote(value, 440)
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Identified f0 (Hz)')
        plt.grid(True)
        plt.plot(pitch_times_gt, pitch_values_MIDI_gt, 'r', pitch_times, pitch_values_MIDI, 'g--')
        plt.gca().legend(('GT','pYIN'))
        #plt.plot(pitch_times, pitch_values2, 'r')
        plt.show()
        correct_frame_num = 0
        pitch_values_MIDI_aligned = numpy.zeros(len(pitch_values_MIDI_gt))
        # Evaluation
        for i, item in enumerate(pitch_times_gt):
            if (i * int(hopSize_gt/hopSize)) >= len(pitch_values_MIDI):
                pitch_values_MIDI_aligned[i] = pitch_values_MIDI[-1]
                if abs(pitch_values_MIDI_gt[i] - pitch_values_MIDI[-1]) < 1:
                    correct_frame_num += 1
                #else:
                    #print('error')
            else:
                pitch_values_MIDI_aligned[i] = pitch_values_MIDI[i * int(hopSize_gt/hopSize)]
                if abs(pitch_values_MIDI_gt[i] - pitch_values_MIDI[i * int(hopSize_gt/hopSize)]) <1:
                    correct_frame_num += 1
                #else:
                    #print('error')
        print('accuracy is', correct_frame_num/len(pitch_times_gt))
        print('accuracy is', correct_frame_num / len(pitch_times_gt), file=f_result)
        accuracy_total.append(correct_frame_num/len(pitch_times_gt))
        accuracy_this_person.append(correct_frame_num/len(pitch_times_gt))
        pitch_values_MIDI_gt_f1 = MIDI_note_to_one(pitch_values_MIDI_gt)
        pitch_values_MIDI_aligned_f1 = MIDI_note_to_one(pitch_values_MIDI_aligned)
        f1 = f1_score(pitch_values_MIDI_gt_f1, pitch_values_MIDI_aligned_f1, average='binary')
        print('f1 score is:', f1)
        print('f1 score is:', f1, file=f_result)
        f1_total.append(f1)
        f1_this_person.append(f1)
    if person_name == 'geniusturtle':
        print('debug')
    print('----------------------------------------------------', file=f_result)
    print('The overall accuracy for', person_name, 'is:', numpy.mean(accuracy_this_person), '±', numpy.std(accuracy_this_person))
    print('The overall accuracy for', person_name, 'is:', numpy.mean(accuracy_this_person), '±',
          numpy.std(accuracy_this_person), file=f_result)
    print('The overall f1 for ', person_name, 'is:', numpy.mean(f1_this_person), '±', numpy.std(f1_this_person))
    print('The overall f1 for ', person_name, 'is:', numpy.mean(f1_this_person), '±', numpy.std(f1_this_person), file=f_result)
    return f1_total, accuracy_total


def experiment_comnbination(name_of_experiment, male_names, female_names):

    accuracy_total = []
    f1_total = []

    os.chdir(os.path.join(dir, 'MIR-1K', 'PitchLabel'))
    f_result = open(os.path.join(dir, 'result', name_of_experiment + '.txt'), 'w')
    counter = 0
    for person_name in male_names:
        f1_total, accuracy_total = evaluate_per_person(person_name, f1_total, accuracy_total, counter, f_result)
    for person_name in female_names:
        f1_total, accuracy_total = evaluate_per_person(person_name, f1_total, accuracy_total, counter, f_result)
    print('----------------------------------------------------', file=f_result)
    print('overall accuracy is:', numpy.mean(accuracy_total), '±', numpy.std(accuracy_total))
    print('overall accuracy is:', numpy.mean(accuracy_total), '±', numpy.std(accuracy_total), file=f_result)
    print('overall f1 is:', numpy.mean(f1_total), '±', numpy.std(f1_total))
    print('overall f1 is:', numpy.mean(f1_total), '±', numpy.std(f1_total), file=f_result)
    f_result.close()

if __name__ == "__main__":
    male_names1 = ['abjones', 'bobon', 'bug', 'davidson', 'fdps', 'leon', 'stool', 'jmzen', 'geniusturtle', 'khair',
                  'Kenshin']
    male_names2 = ['abjones', 'bobon', 'bug', 'davidson', 'fdps', 'leon', 'stool', 'jmzen', 'khair',
                  'Kenshin']
    female_names1 = ['amy', 'Ani', 'annar', 'ariel', 'heycat', 'tammy', 'titon', 'yifen']
    female_names2 = ['amy', 'Ani', 'annar', 'ariel', 'tammy', 'titon', 'yifen']
    # experiment_comnbination('Male_with_turtle', male_names1, [])
    experiment_comnbination('Male_no_turtle', male_names2, [])
    # experiment_comnbination('Female_with_heycat', female_names1, [])
    # experiment_comnbination('Female_no_heycat', female_names2, [])
    # experiment_comnbination('Overall', male_names1, female_names1)
    # experiment_comnbination('Overall_no_turtle_no_heycat', male_names2, female_names2)
    # experiment_comnbination('Overall_no_turtle_no_heycat', male_names2, female_names2)
    # experiment_comnbination('Overall_no_turtle', male_names2, female_names1)
    #experiment_comnbination('only_geniusturtle', ['geniusturtle'], [])