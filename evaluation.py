"""
The script uses the Frechet Audio Distance (FAD) and CLAPScore to evaluate the performance of the model.

From the google CAPs dataset - 10 pairs of audio and text data are used to evaluate the model.

With the text data, audio is generated using MusicGen and Riffusion models.

Additionally prompts with a diverse range of genres and moods were generated to evaluate the model.

MusicGen and Riffusion models are used to generate audio from the text data.

"""

from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(model_name = "vggish",sample_rate = 16000, use_pca = False, use_activation = False, verbose = False)

path1 = "CAP_audio"
path2 = "MG_cap"
path3 = "RF_Cap"

print(frechet.score(path1, path2))
print(frechet.score(path1, path3))

### Write FAD scores to a csv file

fad_scores = []
for i in range(10):
    fad_scores.append(frechet.score(f"CAP_audio/cap{i}.wav", f"MG_cap/cap_{i}.wav"))

df = pd.DataFrame({'CAP_audio + MusicGen_audio':fad_scores})

df.to_csv('fad_scores.csv',index=False)




from transformers import ClapModel, ClapProcessor
import librosa
from scipy.spatial.distance import cosine
import pandas as pd

model = ClapModel.from_pretrained("laion/clap-htsat-unfused",force_download=True)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused",force_download=True)


##### CLAP_text + CLAP_audio #####

df_captions = pd.read_csv('captions.csv')
clap_scores = []


for i in range(10):
    
    text = df_captions['caption'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'CAP_audio/cap{i}.wav'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    clap_scores.append(similarity_score)


print("Completed CLAP_text + CLAP_audio")
##### CLAP_text + MusicGen_audio #####

musicgen_scores = []

for i in range(10):        
    text = df_captions['caption'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'MG_cap/cap_{i}.wav'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    musicgen_scores.append(similarity_score)


print("Completed CLAP_text + MusicGen_audio")
##### CLAP_text + Riffusion_audio #####

riffusion_scores = []

for i in range(10):
    text = df_captions['caption'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'RF_Cap/cap_{i}.mp3'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    riffusion_scores.append(similarity_score)

print("Completed CLAP_text + Riffusion_audio")
####### Genre_text + MusicGen_audio #####

df_genre = pd.read_csv('prompts_genres.csv')

genre_scores_mg = []

for i in range(10):
    text = df_genre['prompts'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'MG_genre/genre_{i+1}.wav'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    genre_scores_mg.append(similarity_score)


print("Completed Genre_text + MusicGen_audio")
#### Genre_text + Riffusion_audio #####

genre_scores_rf = []

for i in range(10):
    text = df_genre['prompts'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'RF_genre/genre_{i}.mp3'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    genre_scores_rf.append(similarity_score)

print("Completed Genre_text + Riffusion_audio")
####### Mood_text + MusicGen_audio #####

df_mood = pd.read_csv('mood_prompts.csv')

mood_scores_mg = []

for i in range(10):
    text = df_mood['prompts'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'MG_mood/mood_{i+1}.wav'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    mood_scores_mg.append(similarity_score)

print("Completed Mood_text + MusicGen_audio")
#### Mood_text + Riffusion_audio #####

mood_scores_rf = []

for i in range(10):
    text = df_mood['prompts'][i]
    text_inputs = processor(text=text, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.squeeze()

    audio_path = f'RF_mood/mood_{i}.mp3'
    audio_array, sample_rate = librosa.load(audio_path, sr=None)
    audio_inputs = processor(audios=audio_array, return_tensors="pt",sampling_rate=48000)
    audio_embeddings = model.get_audio_features(**audio_inputs)
    audio_embeddings = audio_embeddings.squeeze()
    
    similarity_score = 1 - cosine(text_embeddings.detach().numpy(), audio_embeddings.detach().numpy())
    mood_scores_rf.append(similarity_score)

print("Completed Mood_text + Riffusion_audio")


### Write all the scores to a csv file

df = pd.DataFrame({'CLAP_text + CLAP_audio':clap_scores,
                     'CLAP_text + MusicGen_audio':musicgen_scores,
                     'CLAP_text + Riffusion_audio':riffusion_scores,
                     'Genre_text + MusicGen_audio':genre_scores_mg,
                     'Genre_text + Riffusion_audio':genre_scores_rf,
                     'Mood_text + MusicGen_audio':mood_scores_mg,
                     'Mood_text + Riffusion_audio':mood_scores_rf})

df.to_csv('evaluation_scores.csv',index=False)
