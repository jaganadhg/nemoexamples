#!/usr/bin/env python

import glob 

import json
import soundfile as sf 
from soundfile import SoundFile


def convert_flac_to_wav(wav_path: str) -> None:
    """ Convert a .flac speech file to .wav file 
        Parameters
        -----------
        :params wav_path: Path to the flac file 
        
        Notes
        -----------
        Open the .flac file using soundfile and write to .wav
        format.
        
        Usage
        -----------
        >>> #TODO
    """
    
    with SoundFile(wav_path,) as wav:
        wav_arr = wav.read()
        
        sample_rate = wav.samplerate
        nframes = wav.frames 
        duration = nframes/sample_rate
        print(f"Wave duration is {duration} .")
        output_paths = wav_path.split(".")
        output_path = output_paths[0] + ".wav"
        wav_id = output_paths[0].split("/")[-1]
        
        sf.write(output_path,
                 wav_arr,
                 sample_rate)

    return (output_path, duration, wav_id)


def create_nemo_manifest(flac_path: str, manifest_path: str) -> None:
    """ Convert flac files in a dierctory and generate manifest fro Nemo
        Parameters
        ------------
        :params flac_path: Path to direcctory with flac files
        :params manifest_path: Path (with filename) to write the manifest
        
        Usage
        -------------
        >>> #TODO
    """
    
    all_flac_file = glob.glob(flac_path + "*.flac")
    # import pdb; pdb.set_trace()
    transcript_path = glob.glob(flac_path + "*.trans.txt")[0]
    
    transcripts = open(transcript_path).readlines()
    
    for file_li in all_flac_file:
        with open(manifest_path, 'a') as manifest:
            meta_one = convert_flac_to_wav(file_li)
            
            transcript = list(filter(lambda x: x.startswith(meta_one[2]),
                                     transcripts))[0].strip()
            transcript = transcript.replace(meta_one[2], "")
            
            metadata = {
                    "audio_filepath": meta_one[0],
                    "duration": meta_one[1],
                    "text": transcript
                }
            
            json.dump(metadata, manifest)
            manifest.write("\n")
            

if __name__ == "__main__":
    saample_file = "/home/jaganadhg/AI_RND/nvidianemo/LibriSpeech/dev-clean/84/121550/84-121550-0035.flac"
    
    # convert_flac_to_wav(saample_file)
    
    flac_path = "/home/jaganadhg/AI_RND/nvidianemo/LibriSpeech/dev-clean/84/121123/"
    meta_apth = "metadata_validation.json"
    
    create_nemo_manifest(flac_path,
                         meta_apth)