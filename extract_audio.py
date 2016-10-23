# The simple implementation of obtaining the audio clip of a original video.

import moviepy.editor as mp
import os

def getAudioClip(video_reading_path, audio_storing_path):
    clip = mp.VideoFileClip(video_reading_path)
    print (clip.audio.make_frame)
    clip.audio.write_audiofile(audio_storing_path)



if __name__ == "__main__":

# 1. Set the access path to the original file.
#video_reading_path = "./CS2108-Vine-Dataset/vine/training/1001088152326610944.mp4"

# 2. Set the path to store the extracted audio clip.
#audio_storing_path = "./deeplearning/data/audio/1001088152326610944.wav"

    for file in os.listdir("./CS2108-Vine-Dataset/vine/validation"):
        fileName = file[:len(file)-4]
        try:
            getAudioClip("./CS2108-Vine-Dataset/vine/validation/"+file, "./data/validation_audio/"+fileName+".wav")
        except Exception:
            print Exception
